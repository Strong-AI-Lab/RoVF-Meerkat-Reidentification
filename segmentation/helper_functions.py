
from transformers import AutoImageProcessor, AutoModel
import sys
sys.path.append("..")
from dataloaders.ReID import AnimalClipDataset
import json
import numpy as np
import cv2
import pickle
from tqdm import tqdm
import os
import torch
import matplotlib.pyplot as plt

def extract_patch_embeddings(config, image_list):
    model_outputs = dict()

    model, processor = initalise_DINOv2_model(config)
    full_names = [k for k in image_list.keys()]

    with torch.no_grad():
        for i in tqdm(range(0, len(image_list), config["batch_size"]), desc="Extracting patch embeddings", disable=False):
            # Get the current batch of names
            names = [full_names[i + t] for t in range(min(config["batch_size"], len(image_list) - i))]

            # Get the current batch of images and resize them
            image_batch = [cv2.resize(np.asarray(image_list[name]), (224 * config["resize_factor"], 224 * config["resize_factor"])) for name in names]
            inputs = processor(images=image_batch, return_tensors="pt").to(config["device"])
            
            outputs = model(**inputs, output_attentions=True)
            outputs = outputs['attentions'][-1].detach().cpu().numpy()
            outputs = outputs[:, :, 1:, 0]
            outputs = outputs.transpose(0,2,1).reshape(-1, config["nhd"])
            outputs = outputs.reshape(len(image_batch), -1, config["nhd"])

            for t in range(len(names)):
                model_outputs[names[t]] = outputs[t,...]
            
    if config["checkpoint_path"]:
        with open(os.path.join(config["output_folder"],config["checkpoint_path"]), 'wb') as f:
            pickle.dump(model_outputs,f)

    return model_outputs

def import_dataset(data_dir, cooccurrence_file, num_frames):
    # Load cooccurrence data from the file
    with open(cooccurrence_file, 'r') as file:
        cooccurrences = json.load(file)
    
    # Create the dataset object
    dataset = AnimalClipDataset(data_dir, cooccurrences, mode="video_only", num_frames=num_frames)
    
    return dataset

def save_video(video):
    for i in range(20):
        cv2.imwrite("temp_video/"+str(i).zfill(3)+".jpg", (255*video[i].numpy().transpose(1,2,0)).astype(np.uint8))

def check_bool(arg, arg_name):
    if arg.lower() == 'true':
        return True
    elif arg.lower() == 'false':
        return False
    else:
        raise ValueError(f"Invalid value for boolean argument '--"+arg_name+"': {arg}. Must be 'True' or 'False'.")

def get_video_masks(boxes, names, out_shape, frames=list(range(20))):
    masks = []
    for name in names:
        for frame in frames:
            r = boxes[name]["boxes"][frame]
            p = boxes[name]["raw"][frame]

            if p == [0,224,0,224]:
                x1, x2, y1, y2 = r
            else:
                original_dims = [r[1]-r[0],r[3]-r[2]]

                scale_x = 224/original_dims[0]
                scale_y = 224/original_dims[1]

                x1 = int((p[0] - r[0]) * scale_x) 
                x2 = int((p[1] - r[0]) * scale_x)
                y1 = int((p[2] - r[2]) * scale_y)
                y2 = int((p[3] - r[2]) * scale_y)

            mask = cv2.rectangle(np.zeros((224,224)), [x1,y1], [x2,y2], color=1, thickness=-1)

            # Resize mask to output shape
            resized_mask = cv2.resize(mask, (out_shape[1], out_shape[0]))
            masks.append(resized_mask)
    
    return (np.asarray(masks)).astype(np.uint8)

def get_video_box(boxes, names, frames=list(range(20))):
    box_array = []
    for name in names:
        for frame in frames:
            r = boxes[name]["boxes"][frame]
            p = boxes[name]["raw"][frame]

            if p == [0,224,0,224]:
                x1, x2, y1, y2 = r
            else:
                original_dims = [r[1]-r[0],r[3]-r[2]]

                scale_x = 224/original_dims[0]
                scale_y = 224/original_dims[1]

                x1 = int((p[0] - r[0]) * scale_x) 
                x2 = int((p[1] - r[0]) * scale_x)
                y1 = int((p[2] - r[2]) * scale_y)
                y2 = int((p[3] - r[2]) * scale_y)

            box_array.append([x1,x2,y1,y2])
    
    return (np.asarray(box_array)).astype(np.float32)

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
