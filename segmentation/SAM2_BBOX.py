import argparse
import os
import json
from helper_functions import *
import gzip
from custom_SAM2 import build_custom_sam2_video_predictor as build_sam

def apply_SAM2(config, boxes):
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
        videos = {test_dataset[i][0]:test_dataset[i][1] for i in range(len(test_dataset))}
        train_dataset = import_dataset(os.path.join(config["dataset_dir"],'clips/Train/'), os.path.join(config["dataset_dir"],'Cooccurrences.json'), config["num_frames"])
        val_dataset = import_dataset(os.path.join(config["dataset_dir"],'clips/Val/'), os.path.join(config["dataset_dir"],'Cooccurrences.json'), config["num_frames"])
        videos.update({val_dataset[i][0]:val_dataset[i][1].detach().cpu().numpy().transpose(1,2,0) for i in range(len(val_dataset))})
        videos.update({train_dataset[i][0]:train_dataset[i][1].detach().cpu().numpy().transpose(1,2,0) for i in range(len(train_dataset))})

    full_names = [k for k in videos.keys()]

    sam2_checkpoint = "SAM2/checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    predictor = build_sam(model_cfg, sam2_checkpoint)
    SAM_masks = dict() #dictionary to store resulting masks

    with torch.no_grad():
        # For video in dataset
        for i in tqdm(range(len(videos)), desc="Applying SAM2 to video", disable=False):
            name = full_names[i]
            video = videos[name]

            # Prompt SAM 2
            inference_state = predictor.init_state_video(video=video)
            predictor.reset_state(inference_state)

            box = get_video_box(boxes, [name])
            for frame_num in range(20):
                r = box[frame_num,:]
                _, _, _ = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=frame_num,
                    obj_id=0,
                    box=np.array([r[0], r[2], r[1], r[3]], dtype=np.float32)
                )

            video_segments = {}  # video_segments contains the per-frame segmentation results
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                video_segments[out_frame_idx] = {
                    out_obj_id: out_mask_logits[i]
                    for i, out_obj_id in enumerate(out_obj_ids)
            }

            final_masks = []
            for frame_id in range(len(video_segments)):
                mask = process_mask(video_segments[frame_id][0])
                #visualise_masks(mask, video[frame_id].numpy())
                final_masks.append(mask)

            final_masks = np.asarray(final_masks)

            SAM_masks[name] = final_masks

    with open(config["output_folder"]+"/"+config["prefix"]+"_SAM_box_masks.pkl", 'wb') as f:
        pickle.dump(SAM_masks,f)

def visualise_masks(masks, images):
    images = (255*images.transpose(1,2,0)).astype(np.uint8)
    seg_color = np.asarray([255, 127, 14]).reshape(1,1,-1)
    color_mask = ((255*masks).reshape(224, 224, 1) * seg_color.reshape(1, 1, -1)).astype(np.uint8)
    frame = cv2.addWeighted(images, 1, color_mask, 0.3, 1)

    cv2.imshow("Image", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Set up configuration for the SAM2-bounding box segmentation process")

    #Adding arguments
    #Parameters related to the process configuration and ablation studies.
    parser.add_argument('-t', '--test_mode', type=bool, default=True, help='Whether to only apply this to the test set (True) or all sets (False)')
    parser.add_argument('-fp', '--frame_prompts', type=str, default="[0, 10, 19]", help='List of frame prompts to use')

    #Parameters related to dataset
    parser.add_argument('-o', '--output_folder', type=str, default='results', help='Output folder for results')
    parser.add_argument('-i', '--prefix', type=str, default='meerkat', help='Prefix for saving output')
    parser.add_argument('-dir', '--dataset_dir', type=str, default='meerkat', help='Dataset folder name (i.e., "polarbears" or "meerkat")')
    parser.add_argument('--num_frames', type=int, default=20, help='Number of frames to process (default=20 for meerkats and polar bears)')

    #Related to resources available
    parser.add_argument('-d', '--device', type=str, default='cuda', help='Device to use (e.g., cuda)')
    parser.add_argument('-b', '--batch_size', type=int, default=4, help='Batch size for processing')

    # Parse the arguments
    args = parser.parse_args()

    config = {
        "output_folder":args.output_folder,
        "device":args.device,
        "batch_size":args.batch_size,
        "num_frames":args.num_frames,
        "test_mode":args.test_mode,
        "frame_prompts":eval(args.frame_prompts),
        "prefix": args.prefix,
        "dataset_dir":"../Dataset/"+args.dataset_dir+"_h5files/",
        "dataset":args.dataset_dir
    }

    if not os.path.exists(config["output_folder"]):
        os.makedirs(config["output_folder"])

    #Import boxes
    with open(os.path.join(config["dataset_dir"], "boxes.json"), 'r') as file:
        boxes = json.load(file)

    apply_SAM2(config, boxes)

main()