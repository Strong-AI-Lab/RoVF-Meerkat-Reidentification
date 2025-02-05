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
import gzip

#Disable TQDM to hide propagation bars from SAM2
os.environ['TQDM_DISABLE'] = 'True'

def fit_discriminant_model(config, boxes=None):
    """
    This process is applied to fit a discriminant model, by first extracting the
    patch embeddings for the first frame of each training/validation sample,
    then fitting a PCA or LDA model based on config file.
    """
    #If a checkpoint path is provided, and it exists import embeddings 
    if os.path.exists(os.path.join(config["output_folder"],config["checkpoint_path"])) and not config["restart"]:
        print("Using existing patch embeddings")
        model_outputs = dict()
        with open(os.path.join(config["output_folder"],config["checkpoint_path"]), 'rb') as f:
            model_outputs.update(pickle.load(f))
    else:
        #Import first frame from training and validation sets
        print("Loading datasets, this may take a few minutes")
        train_dataset = import_dataset(os.path.join(config["dataset_dir"],'clips/Train/'), os.path.join(config["dataset_dir"],'Cooccurrences.json'), 1)
        val_dataset = import_dataset(os.path.join(config["dataset_dir"],'clips/Val/'), os.path.join(config["dataset_dir"],'Cooccurrences.json'), 1)

        images = {train_dataset[i][0]:train_dataset[i][1].detach().cpu().numpy().transpose(1,2,0) for i in range(len(train_dataset))}
        images.update({train_dataset[i][0]:val_dataset[i][1].detach().cpu().numpy().transpose(1,2,0) for i in range(len(val_dataset))})

        #Extract patch embeddings for each image
        model_outputs = extract_patch_embeddings(config, images)

    # Fit LDA or PCA component to segment the background
    print("Fitting discriminant model")
    image_names = [k for k in model_outputs.keys()]

    #Reshape patch embeddings to (n_images,nhd)
    model_outputs = np.asarray([model_outputs[k] for k in model_outputs.keys()])
    model_outputs = model_outputs.reshape(-1,config["nhd"])
    
    if config["use_LDA"]:
        #Read bounding box masks
        masks_full = get_video_masks(boxes, image_names, (config["rows"], config["cols"]), frames=[0])
        masks = masks_full.reshape(-1)

        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        discriminant_model = LDA(n_components=1)
        discriminant_model.fit(model_outputs, masks)

        #Save model to pkl file
        with open(os.path.join(config["output_folder"],'LDA_A.pkl'), 'wb') as pickle_file:
            pickle.dump(discriminant_model, pickle_file)
    else:
        from sklearn.decomposition import PCA
        discriminant_model = PCA(n_components=1)
        discriminant_model.fit(model_outputs)

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
    #Load all videos from the training, validation, and test sets
    if config["test_mode"]:
        #Only load the clips needed for testing
        test_dataset = import_dataset(os.path.join(config["dataset_dir"],'clips/Test/'), os.path.join(config["dataset_dir"],'Cooccurrences.json'), config["num_frames"])
        videos = {test_dataset[i][0]:test_dataset[i][1] for i in range(len(test_dataset))}
        if config["dataset"] == "polarbears":
            with gzip.open("ground_truth_masks/polarbear_GT_masks.pkl", 'rb') as f:
                GT = pickle.load(f)
        elif config["dataset"] == "meerkat":
            with gzip.open("ground_truth_masks/meerkat_GT_masks.pkl", 'rb') as f:
                GT = pickle.load(f)
        videos = {key: value for key, value in videos.items() if key in GT}
    else:
        #Load all clips
        test_dataset = import_dataset(os.path.join(config["dataset_dir"],'clips/Test/'), os.path.join(config["dataset_dir"],'Cooccurrences.json'), config["num_frames"])
        train_dataset = import_dataset(os.path.join(config["dataset_dir"],'clips/Train/'), os.path.join(config["dataset_dir"],'Cooccurrences.json'), config["num_frames"])
        val_dataset = import_dataset(os.path.join(config["dataset_dir"],'clips/Val/'), os.path.join(config["dataset_dir"],'Cooccurrences.json'), config["num_frames"])

        videos = {test_dataset[i][0]:test_dataset[i][1] for i in range(len(test_dataset))}
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
        #For video in dataset
        for i in tqdm(range(len(videos)), desc="Extracting patch embeddings", disable=False):

            # 1. Apply DINOv2 Base model
            name = full_names[i]
            video = videos[name]

            # Can prompt the SAM model here if that option is enabled
            if config["SAM_model"]:
                # Prompt SAM 2
                inference_state = predictor.init_state_video(video=videos[name])
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
            embeddings = embeddings.reshape(-1,config["nhd"])
            res_embeddings = discriminant_model.transform(embeddings)

            res_embeddings = res_embeddings.reshape(video.shape[0],config["rows"],config["cols"])
            res_embeddings[:,0,0] = 0

            # 3. Post-processing
            if config["mask_box"]:
                box_masks = get_video_masks(boxes, [name], (config["rows"],config["cols"]))

            mid_masks = []
            for frame_num in range(res_embeddings.shape[0]):
                mask = res_embeddings[frame_num,...]

                #Convert to a binary mask
                if config["use_LDA"]:
                    mask[mask < 0.5] = 0
                    mask[mask >= 0.5] = 1
                else:
                    #Threshold PCA prediction maps at 0
                    mask[mask < 0] = 0
                    mask[mask > 0] = 1

                # Apply morphological operations
                mask = cv2.morphologyEx(mask, op=cv2.MORPH_CLOSE, kernel=np.ones((5,5), dtype=np.uint8))
                mask = cv2.morphologyEx(mask, op=cv2.MORPH_ERODE, kernel=np.ones((3,3), dtype=np.uint8))

                # Mask out background of the bounding box
                if config["mask_box"]:
                    mask[box_masks[frame_num] < 0.5] = 0

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
                mid_masks = np.asarray(mid_masks)
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
                    if config["mask_box"]:
                        mask[cv2.resize(box_masks[frame_id], (224,224), interpolation=cv2.INTER_LINEAR)==0] = 0
                    final_masks.append(mask)

                final_masks = np.asarray(final_masks)
                SAM_masks[name] = final_masks
        
        #Final stage: Save masks for entire dataset
        print("Saving model outputs.")
        if config["save_intermediate_masks"]:
            with open(config["output_folder"]+"/"+config["prefix"]+"DINOv2_masks.pkl", 'wb') as f:
                pickle.dump(DINO_masks,f)
        if config["SAM_model"]:
            with open(config["output_folder"]+"/"+config["prefix"]+"DINOv2_SAM_masks.pkl", 'wb') as f:
                pickle.dump(SAM_masks,f)

def main():
    parser = argparse.ArgumentParser(description="Set up configuration for the DINOv2-LDA-SAM2 segmentation process")

    #Adding arguments
    #Parameters related to the process configuration and ablation studies.
    parser.add_argument('-m', '--use_LDA', type=str, default="True", help='Whether to use LDA (True) or PCA (False)')
    parser.add_argument('-fp', '--frame_prompts', type=str, default="[0, 10, 19]", help='List of frame prompts to use')
    parser.add_argument('-mb', '--mask_box', type=str, default="True", help='Whether to mask frames using bounding box during post-processing')
    parser.add_argument('-t', '--test_mode', type=str, default="True", help='Whether to only apply this to the test set (True) or all sets (False)')
    parser.add_argument('-si', '--save_intermediate_masks', type=str, default="True", help='Whether to save intermediate masks (e.g., DINOv2-LDA)')
    parser.add_argument('-s', '--apply_SAM', type=str, default="True", help='Whether to use SAM2 (True) or not (False)')

    #Parameters related to dataset
    parser.add_argument('-o', '--output_folder', type=str, default='results', help='Output folder for results')
    parser.add_argument('-i', '--prefix', type=str, default='polarbears', help='Prefix for saving output')
    parser.add_argument('-dir', '--dataset_dir', type=str, default='polarbears', help='Dataset folder name (i.e., "polarbears" or "meerkat")')
    parser.add_argument('--num_frames', type=int, default=20, help='Number of frames to process (default=20 for meerkats and polar bears)')
    parser.add_argument('--checkpoint_path', type=str, default='DINO_embeddings.pkl', help='Path to save/load embeddings checkpoint')
    parser.add_argument('--restart', type=str, default="True", help='Whether to restart processing from the beginning')
    
    #Related to resources available
    parser.add_argument('-r', '--resize_factor', type=float, default=4, help='Resize factor for images')
    parser.add_argument('--dino_model', type=str, default="facebook/dinov2-base", help='DINO model to use')
    parser.add_argument('-d', '--device', type=str, default='cuda:0', help='Device to use (e.g., cuda)')
    parser.add_argument('-b', '--batch_size', type=int, default=4, help='Batch size for processing')

    # Parse the arguments
    args = parser.parse_args()

    if args.dino_model == "facebook/dinov2-base":
        patch_size = 14
        nhd = 12
    else:
        raise ValueError(f"Only one DINOv2 model is supported. facebook/dinov2-base.")

    args.restart = check_bool(args.restart, "restart")
    args.use_LDA = check_bool(args.use_LDA, "use_LDA")
    args.save_intermediate_masks = check_bool(args.save_intermediate_masks, "save_intermediate_masks")
    args.mask_box = check_bool(args.mask_box, "mask_box")
    args.test_mode = check_bool(args.test_mode, "test_mode")
    args.apply_SAM = check_bool(args.apply_SAM, "apply_SAM")

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
        "prefix": args.prefix,
        "dataset_dir":"../Dataset/"+args.dataset_dir+"_h5files/",
        "restart":args.restart,
        "num_frames":args.num_frames,
        "frame_prompts":eval(args.frame_prompts),
        "save_intermediate_masks":args.save_intermediate_masks,
        "mask_box":args.mask_box,
        "test_mode":args.test_mode,
        "SAM_model":args.apply_SAM,
        "dataset":args.dataset_dir
    }

    print(config)

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

    print(discriminant_model)

    #Apply model to all frames
    apply_background_masking(config, discriminant_model, boxes)

if __name__ == "__main__":
    main()