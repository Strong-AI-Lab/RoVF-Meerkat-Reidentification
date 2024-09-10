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
from models.recurrent_decoder import RecurrentDecoder

import transformers
from transformers import AutoModel

import yaml

def set_layernorm_eps_recursive(module, eps=1e-5, level=0):
    """
    Recursively set the eps value for all LayerNorm layers in a module, including submodules.
    Args:
        module (nn.Module): The model or submodule to update.
        eps (float): The new epsilon value to set.
        level (int): Used for tracking the depth of recursion (for hierarchy display purposes).
    """
    for name, submodule in module.named_children():
        # Check if the submodule is a LayerNorm layer
        if isinstance(submodule, nn.LayerNorm):
            submodule.eps = eps
            print(f"{'  ' * level}Updated LayerNorm at {name} to eps={eps}")
        else:
            # Recursively apply the same operation to submodules
            print(f"{'  ' * level}Descending into {name}")
            set_layernorm_eps_recursive(submodule, eps, level+1)

def set_dropout_p_recursive(module, p=0.5, level=0):
    """
    Recursively set the dropout probability (p) for all Dropout layers in a module, including submodules.
    Args:
        module (nn.Module): The model or submodule to update.
        p (float): The new dropout probability to set.
        level (int): Used for tracking the depth of recursion (for hierarchy display purposes).
    """
    for name, submodule in module.named_children():
        # Check if the submodule is a Dropout layer
        if isinstance(submodule, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            submodule.p = p
            print(f"{'  ' * level}Updated Dropout at {name} to p={p}")
        else:
            # Recursively apply the same operation to submodules
            print(f"{'  ' * level}Descending into {name}")
            set_dropout_p_recursive(submodule, p, level+1)


def dino_model_load(
    dino_model_name="facebook/dinov2-base", output_dim=None, forward_strat: str="cat", 
    sequence_length=None, num_frames: int=1, dropout_rate=0.1
):
    dino = DINOv2VideoWrapper(
         dino_model_name=dino_model_name, output_dim=output_dim, forward_strat=forward_strat, 
         sequence_length=sequence_length, num_frames=num_frames, dropout_rate=dropout_rate
    )
    return dino

def recurrent_model_perceiver_load(
    perceiver_config, dino_model_name="facebook/dinov2-base", dropout_rate=0.1, freeze_image_model=True
):
    recurrent_model = RecurrentWrapper(
        perceiver_config=perceiver_config, model_name=dino_model_name, 
        dropout_rate=dropout_rate, freeze_image_model=freeze_image_model
    )
    return recurrent_model

def load_model_from_checkpoint(checkpoint_path: str):
    
    def convert_none_str_to_none(value):
        if isinstance(value, str):
            if value.lower() == "none":
                return None
        return value

    # Load the checkpoint file
    checkpoint = torch.load(checkpoint_path)
    
    # Convert the 'metadata' (YAML string) back to a dictionary
    yaml_str = checkpoint['metadata']
    config = yaml.safe_load(yaml_str)
    
    # Extract model details
    model_details = config['model_details']
    model_type = model_details['model_type']
    if model_type == 'dino':
        # Extract DINO model specific parameters from YAML
        dino_model_name = convert_none_str_to_none(model_details.get('dino_model_name', 'facebook/dinov2-base'))
        output_dim = convert_none_str_to_none(model_details.get('output_dim', None))
        forward_strat = convert_none_str_to_none(model_details.get('strategy', 'cat'))
        sequence_length = convert_none_str_to_none(model_details.get('sequence_length', None))
        num_frames = convert_none_str_to_none(model_details.get('num_frames', 1))
        dropout_rate = convert_none_str_to_none(model_details.get('dropout_rate', 0.1))

        # Call dino_model_load with the extracted parameters
        model = dino_model_load(
            dino_model_name=dino_model_name, 
            output_dim=output_dim, 
            forward_strat=forward_strat, 
            sequence_length=sequence_length, 
            num_frames=num_frames, 
            dropout_rate=dropout_rate
        )
        
    elif model_type == 'recurrent' or model_type == "recurrent_perceiver":
        # Extract recurrent model specific parameters
        #perceiver_config = model_details.get('perceiver_config', {})  # assuming this is nested somewhere
        perceiver_config = {
            "input_dim": convert_none_str_to_none(model_details.get('input_dim', 768)),
            "latent_dim": convert_none_str_to_none(model_details.get('latent_dim', 768)),
            "num_heads": convert_none_str_to_none(model_details.get('num_heads', 12)),
            "num_latents": convert_none_str_to_none(model_details.get('num_latents', 64)),
            "num_transformer_layers": convert_none_str_to_none(model_details.get('num_tf_layers', 2)),
            "dropout": convert_none_str_to_none(model_details.get('dropout_rate', 0.1)),
            "output_dim": convert_none_str_to_none(model_details.get('output_dim', 768))
        }
        dino_model_name = model_details.get('dino_model_name', 'facebook/dinov2-base')
        dropout_rate = model_details.get('dropout_rate', 0.1)
        freeze_image_model = model_details.get('freeze_image_model', True)
        #print(f"perceiver_config: {perceiver_config}")
        # Call recurrent_model_perceiver_load with the extracted parameters
        model = recurrent_model_perceiver_load(
            perceiver_config=perceiver_config, 
            dino_model_name=dino_model_name, 
            dropout_rate=dropout_rate, 
            freeze_image_model=freeze_image_model
        )
    elif model_type == 'recurrent_decoder':
        print(f"model_details: {model_details}")
        model = RecurrentDecoder(
            v_size=convert_none_str_to_none(model_details.get("v_size")),
            d_model=convert_none_str_to_none(model_details.get("d_model")),
            nhead=convert_none_str_to_none(model_details.get("nhead")),
            num_layers=convert_none_str_to_none(model_details.get("num_layers")),
            dim_feedforward=convert_none_str_to_none(model_details.get("dim_feedforward")),
            dropout=convert_none_str_to_none(model_details.get("dropout_rate")), 
            activation=convert_none_str_to_none(model_details.get("activation")),
            temperature=convert_none_str_to_none(model_details.get("temperature")),
            image_model_name=convert_none_str_to_none(model_details.get("image_model_name")),
            freeze_image_model=convert_none_str_to_none(model_details.get("freeze_image_model"))
        )


    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load the model state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

if __name__ == "__main__": # TODO: test below
    # test both models
    #dino_model = dino_model_load()
    #recurrent_model = recurrent_model_perceiver_load()

    # run through example video
    #video = torch.randn(2, 8, 3, 224, 224)
    #dino_output = dino_model(video)
    #recurrent_output = recurrent_model(video)

    #print(f"dino_output[-1].size(): {dino_output[-1].size()}")
    #print(f"recurrent_output[-1].size(): {recurrent_output[-1].size()}")

    model_type = "recurrent"
    if model_type == "recurrent" or "recurrent_perceiver":
        ckpt_path = "/home/kkno604/github/meerkat-repos/RoVF-meerkat-reidentification/results/hyperparameter_search/reproducibility_test/rovf_margin_1_adamw_transformations/checkpoint_epoch_1.pt"
        model = load_model_from_checkpoint(ckpt_path)

        #example video 
        video = torch.randn(2, 8, 3, 224, 224)
        output = model(video)[-1]
        print(f"output.size(): {output.size()}")
    elif model_type == "dino":
        ckpt_path = "results/hyperparameter_search/reproducibility_test/rovf_margin_1_adamw_clip1/checkpoint_epoch_16.pt"
        model = load_model_from_checkpoint(ckpt_path)

        #example video 
        video = torch.randn(2, 8, 3, 224, 224)
        output = model(video)
        print(f"output.size(): {output.size()}")
    else:
        raise ValueError(f"Unknown model type: {model_type}")