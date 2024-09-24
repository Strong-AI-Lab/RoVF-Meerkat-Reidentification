import os
import json
import pickle
from helper_functions import *
import argparse
import warnings
import torch
import pickle
from tqdm import tqdm
import os

#Disable TQDM to hide propagation bars from SAM2
os.environ['TQDM_DISABLE'] = 'True'

def extract_patch_embeddings(config, image_list):
    patch_embeddings = dict()

    model, processor = initalise_DINOv2_model(config)
    full_names = [k for k in image_list.keys()]

    with torch.no_grad():
        for i in tqdm(range(0, len(image_list), config["batch_size"]), desc="Extracting patch embeddings", disable=False):
            # Get the current batch of names
            names = [full_names[i + t] for t in range(min(config["batch_size"], len(image_list) - (i+1)))]

            # Get the current batch of images and resize them
            image_batch = [
                cv2.resize(np.asarray(image_list[name]), (224 * config["resize_factor"], 224 * config["resize_factor"]))
                for name in names
            ]

            inputs = processor(images=image_batch, return_tensors="pt").to(config["device"])
            
            outputs = model(**inputs, output_attentions=True)
            outputs = outputs['attentions'][-1].detach().cpu().numpy()
            outputs = outputs[:, :, 1:, 0]
            outputs = outputs.transpose(0,2,1).reshape(-1, config["nhd"])
            outputs = outputs.reshape(len(image_batch), -1, config["nhd"])

            for t in range(len(names)):
                patch_embeddings[names[t]] = outputs[t,...]
            
    if config["checkpoint_path"]:
        with open(os.path.join(config["output_folder"],config["checkpoint_path"]), 'wb') as f:
            pickle.dump(patch_embeddings,f)

    return patch_embeddings


def fit_discriminant_model(config, boxes=None):
    """
    This process is applied to fit a discriminant model, by first extracting the
    patch embeddings for the first frame of each training/validation sample,
    then fitting a PCA or LDA model based on config file.
    """
    #If a checkpoint path is provided, and it exists import embeddings 
    if os.path.exists(os.path.join(config["output_folder"],config["checkpoint_path"])) and not config["restart"]:
        print("Using existing patch embeddings")
        patch_embeddings = dict()
        with open(os.path.join(config["output_folder"],config["checkpoint_path"]), 'rb') as f:
            patch_embeddings.update(pickle.load(f))
    else:
        #Import dataset
        print("Loading datasets, this may take a few minutes")
        train_dataset = import_dataset(os.path.join(config["dataset_dir"],'clips/Train/'), os.path.join(config["dataset_dir"],'Cooccurrences.json'), 1)
        val_dataset = import_dataset(os.path.join(config["dataset_dir"],'clips/Val/'), os.path.join(config["dataset_dir"],'Cooccurrences.json'), 1)

        #Load first frame from each clip to train the LDA model
        images = {train_dataset[i][0]:train_dataset[i][1].detach().cpu().numpy().transpose(1,2,0) for i in range(len(train_dataset))}
        images.update({train_dataset[i][0]:val_dataset[i][1].detach().cpu().numpy().transpose(1,2,0) for i in range(len(val_dataset))})

        #Extract patch embeddings for each image
        patch_embeddings = extract_patch_embeddings(config, images)

    #Reshape patch embeddings to (n_images,nhd)
    num_images = len(patch_embeddings)
    embeddings = np.stack(list(patch_embeddings.values()), axis=0)
    embeddings = embeddings.reshape(-1,config["nhd"])

    # Fit LDA or PCA component to segment the background
    print("Fitting discriminant model")
    if config["use_LDA"]:
        #Read bounding box masks
        masks_full = get_box_masks(boxes, [k for k in patch_embeddings.keys()], (config["rows"], config["cols"])) #Move to LDA?
        masks = masks_full.reshape(-1) #Reshape to (n_images,)

        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        discriminant_model = LDA(n_components=1)
        discriminant_model.fit(embeddings, masks)

        #Save model to pkl file
        with open(os.path.join(config["output_folder"],'LDA_A.pkl'), 'wb') as pickle_file:
            pickle.dump(discriminant_model, pickle_file)
    else:
        from sklearn.decomposition import PCA
        discriminant_model = PCA(n_components=1)
        discriminant_model.fit(embeddings)

        #Save model to pkl file
        with open(os.path.join(config["output_folder"],'PCA_A.pkl'), 'wb') as pickle_file:
            pickle.dump(discriminant_model, pickle_file)

    return discriminant_model

def apply_background_masking(config, discriminant_model, boxes):
    """
    This process is applied when used for inference, after a discriminant model 
    has been fit. Runs through the entire background masking process.

    This version is slightly more optimised compared with the version we use in
    the paper, and instead of reading frames from .jpg files we can pass the 
    tensor directly to SAM 2. This may lead to slightly different results.
    """
    #################
    #Initalise models
    #################
    #Load all frames
    test_dataset = import_dataset(os.path.join(config["dataset_dir"],'clips/Test/'), os.path.join(config["dataset_dir"],'Cooccurrences.json'), config["num_frames"])
    videos = {test_dataset[i][0]:test_dataset[i][1] for i in range(len(test_dataset))}

    #Load all videos from the training, validation, and test sets
    if not config["test_mode"]:
        train_dataset = import_dataset(os.path.join(config["dataset_dir"],'clips/Train/'), os.path.join(config["dataset_dir"],'Cooccurrences.json'), config["num_frames"])
        val_dataset = import_dataset(os.path.join(config["dataset_dir"],'clips/Val/'), os.path.join(config["dataset_dir"],'Cooccurrences.json'), config["num_frames"])
        videos.update({val_dataset[i][0]:val_dataset[i][1].detach().cpu().numpy().transpose(1,2,0) for i in range(len(val_dataset))})
        videos.update({train_dataset[i][0]:train_dataset[i][1].detach().cpu().numpy().transpose(1,2,0) for i in range(len(train_dataset))})

    #Initialise the DINO model
    model, processor = initalise_DINOv2_model(config)
    full_names = [k for k in videos.keys()]

    DINO_masks = dict() #dictionary to store intermediate masks

    #Initialise the SAM model
    if config["SAM_model"]:
        from custom_SAM2 import build_custom_sam2_video_predictor as build_sam
        sam2_checkpoint = "SAM2/checkpoints/sam2_hiera_large.pt"
        model_cfg = "sam2_hiera_l.yaml"
        predictor = build_sam(model_cfg, sam2_checkpoint)
        SAM_masks = dict() #dictionary to store resulting masks
    
    ########################################
    #Applying the background masking process
    ########################################
    with torch.no_grad():
        # For video in dataset
        for i in tqdm(range(len(videos)), desc="Extracting patch embeddings", disable=False):

            # 1. Apply DINOv2 Base model
            name = full_names[i]
            video = videos[name]

            # Can prompt the SAM model here if that option is enabled
            if config["SAM_model"]:
                # Prompt SAM 2
                inference_state = predictor.init_state(video=video)
                predictor.reset_state(inference_state)

            video_detached = video.detach().cpu().numpy().transpose(0,2,3,1)
            embeddings = np.zeros((video.shape[0],config["rows"]*config["cols"],config["nhd"]))

            video_frames = [cv2.resize(np.asarray(video_detached[j,:,:,:]), 
                            (224 * config["resize_factor"], 224 * config["resize_factor"])) 
                            for j in range(video.shape[0])]
            
            for k in range(0, len(video_frames), config["batch_size"]):
                # Get the current batch of images and resize them
                video_batch = [video_frames[k+j] for j in range(min(config["batch_size"], len(video_frames) - k))]
                
                inputs = processor(images=video_batch, return_tensors="pt").to(config["device"])
                
                outputs = model(**inputs, output_attentions=True)
                outputs = outputs['attentions'][-1].detach().cpu().numpy()
                outputs = outputs[:, :, 1:, 0] # Get cls patch attentions [327, 12, 256]
                outputs = outputs.transpose(0,2,1).reshape(-1, config["nhd"])
                outputs = outputs.reshape(len(video_batch), -1, config["nhd"])

                for t in range(len(video_batch)):
                    embeddings[k+t,...] = outputs[t,...]
            
            # 2. Apply LDA/PCA model
            num_frames = len(video_frames)
            embeddings = embeddings.reshape(-1,config["nhd"])
            res_embeddings = discriminant_model.transform(embeddings)

            res_embeddings = res_embeddings.reshape(num_frames,config["rows"],config["cols"])
            res_embeddings[:,0,0] = 0

            # 3. Post-processing
            if config["mask_box"]:
                box_masks = get_single_video_masks(boxes, name, (config["rows"],config["cols"]))

            mid_masks = []
            for frame_num in range(res_embeddings.shape[0]):
                mask = res_embeddings[frame_num,...]

                #Convert to a binary mask
                if config["use_LDA"]:
                    mask[mask < 0.5] = 0
                    mask[mask > 0.5] = 1
                else:
                    #Threshold PCA prediction maps at 0
                    mask[mask < 0] = 0
                    mask[mask > 0] = 1

                # Mask out background of the bounding box
                if config["mask_box"]:
                    mask[box_masks[frame_num] < 0.5] = 0

                # Apply morphological operations
                mask = cv2.morphologyEx(mask, op=cv2.MORPH_CLOSE, kernel=np.ones((5,5), dtype=np.uint8))
                mask = cv2.morphologyEx(mask, op=cv2.MORPH_ERODE, kernel=np.ones((3,3), dtype=np.uint8))
                mask = cv2.resize(mask, (224,224), interpolation=cv2.INTER_LINEAR)
                _, mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)
                mask = mask.astype(np.bool_)

                # These masks are referred to as: DINOv2-LDA/PCA
                if config["save_intermediate_masks"]:
                    mid_masks += [mask]

                # Prompt SAM with masks
                if config["SAM_model"]:
                    if frame_num in config["frame_prompts"]:
                        _, _, _ = predictor.add_new_mask(
                            inference_state=inference_state,
                            frame_idx=frame_num,
                            obj_id=0,
                            mask=mask
                        )
            
            if config["save_intermediate_masks"]:
                DINO_masks[name] = mid_masks
        
            # 4. Propagate masks
            if config["SAM_model"]:
                #Get mask from SAM
                video_segments = {}  # video_segments contains the per-frame segmentation results
                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                    video_segments[out_frame_idx] = {
                        out_obj_id: out_mask_logits[i]
                        for i, out_obj_id in enumerate(out_obj_ids)
                }
                    
                final_masks = []
                for frame_id in range(len(video_segments)):
                    mask = process_mask(video_segments[frame_id][0])
                    mask[cv2.resize(box_masks[frame_id], (224,224), interpolation=cv2.INTER_LINEAR)==0] = 0
                    final_masks.append(mask)

                final_masks = np.asarray(final_masks)
                mid_masks = np.asarray(mid_masks)

                #visualise_masks(image_set, final_masks)
                SAM_masks[name] = final_masks
        
        #Final stage: Save masks for entire dataset
        print("Saving model outputs.")
        if config["save_intermediate_masks"]:
            with open(config["output_folder"]+"DINOv2_masks.pkl", 'wb') as f:
                pickle.dump(DINO_masks,f)
        if config["SAM_model"]:
            with open(config["output_folder"]+"SAM_Masks.pkl", 'wb') as f:
                pickle.dump(SAM_masks,f)

def main():
    parser = argparse.ArgumentParser(description="Set up configuration for the DINOv2-LDA-SAM2 segmentation process")

    #Adding arguments
    #Parameters related to the process configuration and ablation studies.
    parser.add_argument('-m', '--use_LDA', type=bool, default=True, help='Whether to use LDA (True) or PCA (False)')
    parser.add_argument('-fp', '--frame_prompts', type=int, nargs='+', default=[0, 10, 19], help='List of frame prompts to use')
    parser.add_argument('-mb', '--mask_box', type=bool, default=True, help='Whether to mask frames using bounding box during post-processing')
    parser.add_argument('-t', '--test_mode', type=bool, default=True, help='Whether to only apply this to the test set (True) or all sets (False)')
    parser.add_argument('-si', '--save_intermediate_masks', type=bool, default=True, help='Whether to save intermediate masks (e.g., DINOv2-LDA)')
    parser.add_argument('-s', '--apply_SAM', type=bool, default=True, help='Whether to use SAM2 (True) or not (False)')

    #Parameters related to dataset
    parser.add_argument('-o', '--output_folder', type=str, default='results', help='Output folder for results')
    parser.add_argument('-i', '--dataset_dir', type=str, default='polarbears_h5files', help='Dataset directory')
    parser.add_argument('--num_frames', type=int, default=20, help='Number of frames to process (default=20 for meerkats and polar bears)')
    parser.add_argument('--checkpoint_path', type=str, default='DINO_embeddings.pkl', help='Path to save/load embeddings checkpoint')
    parser.add_argument('--restart', type=bool, default=False, help='Whether to restart processing from the beginning')
    
    #Related to resources available
    parser.add_argument('-r', '--resize_factor', type=float, default=4, help='Resize factor for images')
    parser.add_argument('--dino_model', type=str, default="facebook/dinov2-base", help='DINO model to use')
    parser.add_argument('-d', '--device', type=str, default='cuda', help='Device to use (e.g., cuda)')
    parser.add_argument('-b', '--batch_size', type=int, default=4, help='Batch size for processing')

    # Parse the arguments
    args = parser.parse_args()

    if args.dino_model == "facebook/dinov2-base":
        patch_size = 14
        nhd = 12
    else:
        raise ValueError(f"Only one DINOv2 model is supported. facebook/dinov2-base.")

    if not args.apply_SAM and not args.save_intermediate_masks:
        warnings.warn(f"Expected one of 'save_intermediate_masks' and 'use_SAM' to be True. No masks will be saved.", UserWarning)    

    config = {
        "resize_factor":args.resize_factor,
        "image_height":224*args.resize_factor,
        "image_width":224*args.resize_factor,
        "nhd":nhd,
        "rows":(224*args.resize_factor)//patch_size,
        "cols":(224*args.resize_factor)//patch_size,
        "dino_model":args.dino_model,
        "use_LDA":args.use_LDA,
        "device":args.device,
        "batch_size":args.batch_size,
        "checkpoint_path":args.checkpoint_path,
        "output_folder":args.output_folder,
        "dataset_dir":"../Dataset/"+args.dataset_dir+"/",
        "restart":args.restart,
        "num_frames":args.num_frames,
        "frame_prompts":args.frame_prompts,
        "save_intermediate_masks":args.save_intermediate_masks,
        "mask_box":args.mask_box,
        "test_mode":args.test_mode,
        "SAM_model":args.apply_SAM
    }

    #Setup output folder
    if not os.path.exists(config["output_folder"]):
        os.makedirs(config["output_folder"])

    #Import boxes
    with open(os.path.join(config["dataset_dir"], "boxes.json"), 'r') as file:
        boxes = json.load(file)

    #Get discriminant model for patch embedding classification
    if not config["restart"]:
        if config["use_LDA"]:
            model_path = os.path.join(config["dataset_dir"],"LDA_A.pkl")
        else:
            model_path = os.path.join(config["dataset_dir"],"PCA_A.pkl")

        #If a model has already been fit, use that, otherwise create a new one
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                discriminant_model = pickle.load(f)
        else:
            discriminant_model = fit_discriminant_model(config, boxes)
    else:
        discriminant_model = fit_discriminant_model(config, boxes)

    #Apply model to all frames
    apply_background_masking(config, discriminant_model, boxes)

if __name__ == "__main__":
    main()