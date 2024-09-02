'''
Helper functions to help loading models used for training and evaluation.
'''

import torch
import torch.nn as nn

import sys
sys.path.append("..")

from models.dinov2_wrapper import DINOv2VideoWrapper
from models.recurrent_wrapper import RecurrentWrapper
from models.perceiver_wrapper import CrossAttention, TransformerEncoder, Perceiver

import transformers
from transformers import AutoModel

def dino_model_load(
    dino_model_name="facebook/dinov2-base", output_dim=None, forward_strat: str="cat", sequence_length=None, num_frames: int=1, dropout_rate=0.1
):

    # #model_name: str, dropout_rate: float=0.0, strategy: str="average"
    dino = DINOv2VideoWrapper(
         dino_model_name=dino_model_name, output_dim=output_dim, forward_strat=forward_strat, 
         sequence_length=sequence_length, num_frames=num_frames, dropout_rate=dropout_rate
    )
    return dino

def recurrent_model_perceiver_load(
    perceiver_config, dino_model_name="facebook/dinov2-base", dropout_rate=0.1, freeze_image_model=True
):
    #perceiver_config = {
    #    "input_dim": input_dim,
    #    "latent_dim": latent_dim,
    #    "num_heads": num_heads,
    #    "num_latents": num_latents,
    #    "num_transformer_layers": num_tf_layers,
    #    "dropout": dropout_rate,
    #    "output_dim": output_dim
    #}

    recurrent_model = RecurrentWrapper(
        perceiver_config=perceiver_config, model_name=dino_model_name, 
        dropout_rate=dropout_rate, freeze_image_model=freeze_image_model
    )
    return recurrent_model

if __name__ == "__main__": # TODO: test below
    # test both models
    dino_model = dino_model_load()
    recurrent_model = recurrent_model_perceiver_load()

    # run through example video
    video = torch.randn(2, 8, 3, 224, 224)
    dino_output = dino_model(video)
    recurrent_output = recurrent_model(video)

    print(f"dino_output[-1].size(): {dino_output[-1].size()}")
    print(f"recurrent_output[-1].size(): {recurrent_output[-1].size()}")