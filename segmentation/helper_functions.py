
from transformers import AutoImageProcessor, AutoModel
import sys
sys.path.append("..")
from dataloaders.ReID import AnimalClipDataset
import json
import numpy as np
import cv2

def import_dataset(data_dir, cooccurrence_file, num_frames):
    # Load cooccurrence data from the file
    with open(cooccurrence_file, 'r') as file:
        cooccurrences = json.load(file)
    
    # Create the dataset object
    dataset = AnimalClipDataset(data_dir, cooccurrences, mode="video_only", num_frames=num_frames)
    
    return dataset

def get_single_video_masks(boxes, name, out_shape):
    masks = []
    for frame in range(20):
        start_position = [boxes[name]["boxes"][frame][2],boxes[name]["boxes"][frame][3], boxes[name]["boxes"][frame][0],boxes[name]["boxes"][frame][1]]

        start_position = np.asarray(start_position, dtype=np.uint8)

        mask = cv2.rectangle(np.zeros((224,224)), [start_position[0],start_position[2]], [start_position[1],start_position[3]], color=255, thickness=-1)
        masks += [cv2.resize(mask, (out_shape[1], out_shape[0]))]
    
    return masks

def get_box_masks(boxes, names, out_shape, frame = 0):
    masks = []

    for name in names:
        box = boxes[name]["boxes"][frame]
        raw = boxes[name]["raw"][frame]

        # Calculate height and width of the box
        height = box[1] - box[0]
        width = box[3] - box[2]

        # Scaling factors for height and width
        height_scale = 224 / height
        width_scale = 224 / width

        # Calculate adjusted start position relative to the box
        start_position = [
            (raw[0] - box[0]) * height_scale,
            (raw[1] - box[0]) * height_scale,
            (raw[2] - box[2]) * width_scale,
            (raw[3] - box[2]) * width_scale
        ]

        # Convert to uint8 type
        start_position = np.asarray(start_position, dtype=np.uint8)

        # Create mask with rectangle based on scaled start positions
        mask = cv2.rectangle(
            np.zeros((224, 224), dtype=np.uint8), 
            (start_position[0], start_position[2]), 
            (start_position[1], start_position[3]), 
            color=255, thickness=-1
        )

        # Resize mask to output shape
        resized_mask = cv2.resize(mask, (out_shape[1], out_shape[0]))
        masks.append(resized_mask)

    return np.asarray(masks,dtype=np.uint8)//255

def initalise_DINOv2_model(config):
    processor = AutoImageProcessor.from_pretrained(config["dino_model"], force_download=False)

    processor.crop_size = {'height': config["image_height"], 'width':config["image_width"]}
    processor.do_resize = False
    processor.do_rescale = False
    processor.do_center_crop = False

    model = AutoModel.from_pretrained(config["dino_model"], device_map = config["device"], force_download=False)
    model = model.to(config["device"])
    return model, processor

def process_mask(mask_in):
    mask = (mask_in > 0.0).cpu().numpy()
    return mask[0,...]
