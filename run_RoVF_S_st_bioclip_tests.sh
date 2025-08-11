#!/bin/bash

model_dir="results/hyperparameter_search/RoVF_S_st_bioclip"
df_meerkat="Dataset/meerkat_h5files/Precomputed_test_examples_meerkat.csv"
df_polarbear="Dataset/polarbears_h5files/Precomputed_test_examples_polarbear.csv"
main_py="main.py"
device="cuda"

find "$model_dir" -type f -name "checkpoint_epoch_*.pt" | while read -r ckpt; do
    df="$df_meerkat"
    mask_file="Dataset/meerkat_h5files/masks/meerkat_masks.pkl"
    echo "Testing checkpoint with mask: $ckpt"
    CUDA_VISIBLE_DEVICES=0 python "$main_py" test '' -cp "$ckpt" -df "$df" -d "$device" -m "$mask_file"
    echo "Testing checkpoint without mask: $ckpt"
    CUDA_VISIBLE_DEVICES=0 python "$main_py" test '' -cp "$ckpt" -df "$df" -d "$device"
done
