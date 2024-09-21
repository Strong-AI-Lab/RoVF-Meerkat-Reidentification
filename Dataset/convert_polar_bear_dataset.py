import os
import shutil
import zipfile
import argparse
import tempfile
import h5py
import numpy as np
import pandas as pd
import cv2
import json

def save_h5(filename, clip_set):
    with h5py.File(filename, 'w') as f:
        for j, (key,value) in enumerate(clip_set.items()):
            clip = np.zeros((20, 224, 224, 3), dtype=np.uint8)
            for frame in range(len(value)):
                clip[frame,...] = value[frame]
            f.create_dataset(key, data=clip[:,:,:,(2,1,0)])

def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        print(f"Extracted zip file to {extract_to}")

def process_extracted_data(extract_to):
    print(f"Processing data in {extract_to}, this may take a few minutes")

    #Create new directories
    output_dir = "polarbears_h5files"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(os.path.join(output_dir, "clips")):
        os.makedirs(os.path.join(output_dir, "clips"))
    if not os.path.exists(os.path.join(output_dir, "clips", "Train")):
        os.makedirs(os.path.join(output_dir, "clips", "Train"))
    if not os.path.exists(os.path.join(output_dir, "clips", "Val")):
        os.makedirs(os.path.join(output_dir, "clips", "Val"))
    if not os.path.exists(os.path.join(output_dir, "clips", "Test")):
        os.makedirs(os.path.join(output_dir, "clips", "Test"))

    #Read dataset, sample frames, and save as h5 files
    test_percent = 0.8
    val_percent = 0.7

    tracks = pd.read_csv(extract_to+"/track_info.csv")

    split_index = {i: tracks[tracks["id"] == i]["tracklet"].nunique() for i in range(13)}

    counter = {i:0 for i in range(13)}
    lengths = []
    train_clips = dict()
    test_clips = dict()
    val_clips = dict()

    for _, row in tracks.iterrows():
        total_frames = row.iloc[4]-row.iloc[3]+1
        lengths += [total_frames]
        clip = np.zeros((total_frames,224,224,3))
        for i, k in enumerate(range(row.iloc[3], row.iloc[4]+1)):
            filename = extract_to+"/"+str(row["id"]).zfill(3)+"/"+str(row["id"]).zfill(3)+"C"+str(row["cam"]).zfill(2)+"T"+str(row["tracklet"]).zfill(3)+"F"+str(k).zfill(3)+".jpg"
            im = cv2.resize(cv2.imread(filename), (224,224))
            clip[i,...] = im
        
        interval = np.linspace(0, total_frames - 1, 20, dtype=int)
        clip = clip[interval,...]
        if counter[row["id"]] >= split_index[row["id"]]*test_percent:
            test_clips[str(row["id"]).zfill(3)+"_"+str(row["tracklet"]).zfill(3)] = clip
        elif counter[row["id"]] >= split_index[row["id"]]*val_percent:
            val_clips[str(row["id"]).zfill(3)+"_"+str(row["tracklet"]).zfill(3)] = clip
        else:
            train_clips[str(row["id"]).zfill(3)+"_"+str(row["tracklet"]).zfill(3)] = clip
        counter[row["id"]] += 1

    cocc = {i:[k for k in range(13) if k != i] for i in range(13)}

    print("Number for sets:", len(train_clips)+len(val_clips)+len(test_clips), len(train_clips), len(val_clips), len(test_clips))


    save_h5(output_dir+'/clips/Train/PolarBears_train.h5', train_clips)
    save_h5(output_dir+'/clips/Val/PolarBears_val.h5', val_clips)
    save_h5(output_dir+'/clips/Test/PolarBears_test.h5', test_clips)

    with open(output_dir+"/Cooccurrences.json", 'w') as file:
        json.dump(cocc, file, indent=4)

def clean_up_extracted_data(extract_to):
    """
    Deletes the extracted data directory.
    """
    if os.path.exists(extract_to):
        shutil.rmtree(extract_to)
        print(f"Deleted extracted data from {extract_to}")

def main(zip_path):
    # Create a temporary directory for extraction
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory {temp_dir}")
        
        # Extract zip file
        extract_zip(zip_path, temp_dir)
        
        # Process extracted data
        process_extracted_data(temp_dir)
        
        # Clean up is automatically done by TemporaryDirectory context manager
        clean_up_extracted_data(temp_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert the PolarBearVidID dataset to our format from the zip file.")
    parser.add_argument("zip_path", help="Path to the dataset zip file.")
    args = parser.parse_args()

    main(args.zip_path)