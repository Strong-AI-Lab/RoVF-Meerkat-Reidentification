'''
Helper functions to help loading models used for training and evaluation.
'''

import torch
import torch.nn as nn

import sys
sys.path.append("..")

from models.dinov2_wrapper import DINOv2VideoWrapper
from models.recurrent_wrapper import RecurrentWrapper
from models.recurrent_decoder import RecurrentDecoder
from models.bioCLIP_wrapper import BioCLIPVideoWrapper
from models.MegaDescriptor_wrapper import MegaDescriptorVideoWrapper
from models.ViViT_wrapper import ViViTWrapper
from models.TimeSformer_wrapper import TimeSformerWrapper

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

def megadescriptors_model_load(
    model_name="facebook/dino-vits16", output_dim=None, forward_strat: str="cat",
    sequence_length=None, num_frames: int=1, dropout_rate=0.1
):
    megadescriptor = DINOv2VideoWrapper(
         dino_model_name=dino_model_name, output_dim=output_dim, forward_strat=forward_strat, 
         sequence_length=sequence_length, num_frames=num_frames, dropout_rate=dropout_rate
    )

def recurrent_model_perceiver_load(
    perceiver_config, dino_model_name="facebook/dinov2-base", dropout_rate=0.1, freeze_image_model=True, is_append_avg_emb=False
):
    from models.perceiver_wrapper import CrossAttention, TransformerEncoder, TransformerDecoder, Perceiver
    recurrent_model = RecurrentWrapper(
        perceiver_config=perceiver_config, model_name=dino_model_name, 
        dropout_rate=dropout_rate, freeze_image_model=freeze_image_model,
        is_append_avg_emb=is_append_avg_emb, type_="v1"
    )
    return recurrent_model

def recurrent_model_perceiver_loadv2(
    perceiver_config, dino_model_name="facebook/dinov2-base", dropout_rate=0.1, freeze_image_model=True, is_append_avg_emb=False
):
    from models.perceiver_wrapper import CrossAttention, TransformerEncoder, TransformerDecoder, PerceiverV2 as Perceiver
    recurrent_model = RecurrentWrapper(
        perceiver_config=perceiver_config, model_name=dino_model_name, 
        dropout_rate=dropout_rate, freeze_image_model=freeze_image_model,
        is_append_avg_emb=is_append_avg_emb, type_="v2"
    )
    return recurrent_model

def LSTM_model_load(
    perceiver_config, dino_model_name="facebook/dinov2-base", dropout_rate=0.1, freeze_image_model=True, is_append_avg_emb=False
):
    recurrent_model = RecurrentWrapper(
        perceiver_config=perceiver_config, model_name=dino_model_name, 
        dropout_rate=dropout_rate, freeze_image_model=freeze_image_model,
        is_append_avg_emb=is_append_avg_emb, type_="", recurrent_type="lstm"
    )
    return recurrent_model

def GRU_model_load(
    perceiver_config, dino_model_name="facebook/dinov2-base", dropout_rate=0.1, freeze_image_model=True, is_append_avg_emb=False
):
    recurrent_model = RecurrentWrapper(
        perceiver_config=perceiver_config, model_name=dino_model_name, 
        dropout_rate=dropout_rate, freeze_image_model=freeze_image_model,
        is_append_avg_emb=is_append_avg_emb, type_="", recurrent_type="gru"
    )
    return recurrent_model


def bioclip_model_load(
    model_name="hf-hub:imageomics/bioclip", output_dim=None, forward_strat: str="cls", 
    sequence_length=None, num_frames: int=1, dropout_rate=0.1, checkpoint_path=None
):
    bioclip = BioCLIPVideoWrapper(
        model_name=model_name, output_dim=output_dim, forward_strat=forward_strat, 
        sequence_length=sequence_length, num_frames=num_frames, dropout_rate=dropout_rate, 
        checkpoint_path=checkpoint_path
    )
    return bioclip

def megadescriptors_model_load(
    model_name="hf-hub:BVRA/MegaDescriptor-T-224", output_dim=None, forward_strat: str="cat",
    sequence_length=None, num_frames: int=1, dropout_rate=0.1, checkpoint_path=None
):
    megadescriptor = MegaDescriptorVideoWrapper(
        model_name=model_name, output_dim=output_dim, forward_strat=forward_strat, 
        sequence_length=sequence_length, num_frames=num_frames, dropout_rate=dropout_rate, 
        checkpoint_path=checkpoint_path
    )
    return megadescriptor

def vivit_model_load(
    model_name="google/vivit-b-16x2-kinetics400", output_dim=None, dropout_rate=0.1
):
    vivit = ViViTWrapper(
        vivit_model_name=model_name, output_dim=output_dim, dropout_rate=dropout_rate
    )
    return vivit

def timesformer_model_load(
    model_name="facebook/timesformer-base-finetuned-k400", output_dim=None, dropout_rate=0.1
):
    timesformer = TimeSformerWrapper(
        timesformer_model_name=model_name, output_dim=output_dim, dropout_rate=dropout_rate
    )
    return timesformer

def image_model_load(model_type, embed_dim, training=False):
    import torchvision.models as models
    if model_type == 'VGG-16':
        import torchvision.models as models
        model = models.vgg16()
    elif model_type == 'ResNet18':
        import torchvision.models as models
        model = models.resnet18()
    elif model_type == 'ResNet50':
        import torchvision.models as models
        model = models.resnet50()
    elif model_type == 'ResNet152':
        model = models.resnet152()
    
    # Initialise weights of FC layer
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    if "ResNet" in model_type:
        if training:
            # Freeze all the layers in the model
            for param in model.parameters():
                param.requires_grad = False
        
        model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 2048),nn.ReLU(),nn.Linear(2048,embed_dim))

        if training:
            init_weights(model.fc)
    else:
        if training:
            for param in model.features.parameters():
                param.requires_grad = False

        model.classifier = nn.Sequential(
            model.classifier[0], 
            model.classifier[1], 
            model.classifier[2], 
            nn.Linear(model.classifier[3].in_features, 2048),
            nn.ReLU(),
            nn.Linear(2048,embed_dim)
        )

        if training:
            init_weights(model.classifier)
    return model

def load_model_from_checkpoint(checkpoint_path: str):
    
    def convert_none_str_to_none(value):
        if isinstance(value, str):
            if value.lower() == "none":
                return None
        return value

    def get_with_print(dictionary, key, default=None):
        if key not in dictionary:
            print(f"Key '{key}' not found. Using default value: {default}")
            return default
        return dictionary[key]

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
        dino_model_name = convert_none_str_to_none(get_with_print(model_details, 'dino_model_name', 'facebook/dinov2-base'))
        output_dim = convert_none_str_to_none(get_with_print(model_details, 'output_dim', None))
        forward_strat = convert_none_str_to_none(get_with_print(model_details, 'forward_strat', 'cat'))
        sequence_length = convert_none_str_to_none(get_with_print(model_details, 'sequence_length', None))
        num_frames = convert_none_str_to_none(get_with_print(model_details, 'num_frames', 1))
        dropout_rate = convert_none_str_to_none(get_with_print(model_details, 'dropout_rate', 0.1))

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
        perceiver_config = {
            "raw_input_dim": convert_none_str_to_none(get_with_print(model_details, 'raw_input_dim', 384)),
            "embedding_dim": convert_none_str_to_none(get_with_print(model_details, 'embedding_dim', 384)),
            "latent_dim": convert_none_str_to_none(get_with_print(model_details, 'latent_dim', 384)),
            "num_heads": convert_none_str_to_none(get_with_print(model_details, 'num_heads', 12)),
            "num_latents": convert_none_str_to_none(get_with_print(model_details, 'num_latents', 512)),
            "num_transformer_layers": convert_none_str_to_none(get_with_print(model_details, 'num_tf_layers', 2)),
            "dropout": convert_none_str_to_none(get_with_print(model_details, 'dropout_rate', 0.1)),
            "output_dim": convert_none_str_to_none(get_with_print(model_details, 'output_dim', 384)),
            "use_raw_input": convert_none_str_to_none(get_with_print(model_details, 'use_raw_input', True)),
            "use_embeddings": convert_none_str_to_none(get_with_print(model_details, 'use_embeddings', True)),
            "flatten_channels": convert_none_str_to_none(get_with_print(model_details, 'flatten_channels', False)),
        }

        dino_model_name = convert_none_str_to_none(get_with_print(model_details, 'dino_model_name', 'facebook/dinov2-small'))
        dropout_rate = get_with_print(model_details, 'dropout_rate', 0.1)
        freeze_image_model = get_with_print(model_details, 'freeze_image_model', True)
        is_append_avg_emb = get_with_print(model_details, 'is_append_avg_emb', False)

        model = recurrent_model_perceiver_load(
            perceiver_config=perceiver_config, 
            dino_model_name=dino_model_name, 
            dropout_rate=dropout_rate, 
            freeze_image_model=freeze_image_model,
            is_append_avg_emb=is_append_avg_emb
        )
    elif model_type == "recurrent_perceiverv2":
        # Extract recurrent model specific parameters
        perceiver_config = {
            "raw_input_dim": convert_none_str_to_none(get_with_print(model_details, 'raw_input_dim', 384)),
            "embedding_dim": convert_none_str_to_none(get_with_print(model_details, 'embedding_dim', 384)),
            "latent_dim": convert_none_str_to_none(get_with_print(model_details, 'latent_dim', 384)),
            "num_heads": convert_none_str_to_none(get_with_print(model_details, 'num_heads', 12)),
            "num_latents": convert_none_str_to_none(get_with_print(model_details, 'num_latents', 512)),
            "num_transformer_layers": convert_none_str_to_none(get_with_print(model_details, 'num_tf_layers', 2)),
            "dropout": convert_none_str_to_none(get_with_print(model_details, 'dropout_rate', 0.1)),
            "output_dim": convert_none_str_to_none(get_with_print(model_details, 'output_dim', 384)),
            "use_raw_input": convert_none_str_to_none(get_with_print(model_details, 'use_raw_input', True)),
            "use_embeddings": convert_none_str_to_none(get_with_print(model_details, 'use_embeddings', True)),
            "flatten_channels": convert_none_str_to_none(get_with_print(model_details, 'flatten_channels', False)),
        }

        dino_model_name = convert_none_str_to_none(get_with_print(model_details, 'dino_model_name', 'facebook/dinov2-small'))
        dropout_rate = get_with_print(model_details, 'dropout_rate', 0.1)
        freeze_image_model = get_with_print(model_details, 'freeze_image_model', True)
        is_append_avg_emb = get_with_print(model_details, 'is_append_avg_emb', False)

        model = recurrent_model_perceiver_loadv2(
            perceiver_config=perceiver_config, 
            dino_model_name=dino_model_name, 
            dropout_rate=dropout_rate, 
            freeze_image_model=freeze_image_model,
            is_append_avg_emb=is_append_avg_emb
        )
    elif model_type == "LSTM":
        # Extract recurrent model specific parameters
        perceiver_config = {
            "raw_input_dim": convert_none_str_to_none(get_with_print(model_details, 'raw_input_dim', 384)),
            "embedding_dim": convert_none_str_to_none(get_with_print(model_details, 'embedding_dim', 384)),
            "latent_dim": convert_none_str_to_none(get_with_print(model_details, 'latent_dim', 384)),
            "num_heads": convert_none_str_to_none(get_with_print(model_details, 'num_heads', 12)),
            "num_latents": convert_none_str_to_none(get_with_print(model_details, 'num_latents', 512)),
            "num_transformer_layers": convert_none_str_to_none(get_with_print(model_details, 'num_tf_layers', 2)),
            "dropout": convert_none_str_to_none(get_with_print(model_details, 'dropout_rate', 0.1)),
            "output_dim": convert_none_str_to_none(get_with_print(model_details, 'output_dim', 384)),
            "use_raw_input": convert_none_str_to_none(get_with_print(model_details, 'use_raw_input', True)),
            "use_embeddings": convert_none_str_to_none(get_with_print(model_details, 'use_embeddings', True)),
            "flatten_channels": convert_none_str_to_none(get_with_print(model_details, 'flatten_channels', False)),
        }

        dino_model_name = convert_none_str_to_none(get_with_print(model_details, 'dino_model_name', 'facebook/dinov2-small'))
        dropout_rate = get_with_print(model_details, 'dropout_rate', 0.1)
        freeze_image_model = get_with_print(model_details, 'freeze_image_model', True)
        is_append_avg_emb = get_with_print(model_details, 'is_append_avg_emb', False)

        model = LSTM_model_load(
            perceiver_config=perceiver_config, 
            dino_model_name=dino_model_name, 
            dropout_rate=dropout_rate, 
            freeze_image_model=freeze_image_model,
            is_append_avg_emb=is_append_avg_emb
        )
    elif model_type == "GRU":
        # Extract recurrent model specific parameters
        perceiver_config = {
            "raw_input_dim": convert_none_str_to_none(get_with_print(model_details, 'raw_input_dim', 384)),
            "embedding_dim": convert_none_str_to_none(get_with_print(model_details, 'embedding_dim', 384)),
            "latent_dim": convert_none_str_to_none(get_with_print(model_details, 'latent_dim', 384)),
            "num_heads": convert_none_str_to_none(get_with_print(model_details, 'num_heads', 12)),
            "num_latents": convert_none_str_to_none(get_with_print(model_details, 'num_latents', 512)),
            "num_transformer_layers": convert_none_str_to_none(get_with_print(model_details, 'num_tf_layers', 2)),
            "dropout": convert_none_str_to_none(get_with_print(model_details, 'dropout_rate', 0.1)),
            "output_dim": convert_none_str_to_none(get_with_print(model_details, 'output_dim', 384)),
            "use_raw_input": convert_none_str_to_none(get_with_print(model_details, 'use_raw_input', True)),
            "use_embeddings": convert_none_str_to_none(get_with_print(model_details, 'use_embeddings', True)),
            "flatten_channels": convert_none_str_to_none(get_with_print(model_details, 'flatten_channels', False)),
        }

        dino_model_name = convert_none_str_to_none(get_with_print(model_details, 'dino_model_name', 'facebook/dinov2-small'))
        dropout_rate = get_with_print(model_details, 'dropout_rate', 0.1)
        freeze_image_model = get_with_print(model_details, 'freeze_image_model', True)
        is_append_avg_emb = get_with_print(model_details, 'is_append_avg_emb', False)

        model = GRU_model_load(
            perceiver_config=perceiver_config, 
            dino_model_name=dino_model_name, 
            dropout_rate=dropout_rate, 
            freeze_image_model=freeze_image_model,
            is_append_avg_emb=is_append_avg_emb
        )
    elif model_type == "bioclip":
        # Extract BioCLIP model specific parameters from YAML
        model_name = convert_none_str_to_none(get_with_print(model_details, 'model_name', 'hf-hub:imageomics/bioclip'))
        output_dim = convert_none_str_to_none(get_with_print(model_details, 'output_dim', None))
        forward_strat = convert_none_str_to_none(get_with_print(model_details, 'forward_strat', 'cls'))
        sequence_length = convert_none_str_to_none(get_with_print(model_details, 'sequence_length', None))
        num_frames = convert_none_str_to_none(get_with_print(model_details, 'num_frames', 1))
        dropout_rate = convert_none_str_to_none(get_with_print(model_details, 'dropout_rate', 0.1))
        checkpoint_path = convert_none_str_to_none(get_with_print(model_details, 'checkpoint_path', None))

        # Call bioclip_model_load with the extracted parameters
        model = bioclip_model_load(
            model_name=model_name, 
            output_dim=output_dim, 
            forward_strat=forward_strat, 
            sequence_length=sequence_length, 
            num_frames=num_frames, 
            dropout_rate=dropout_rate, 
            checkpoint_path=checkpoint_path
        )
    elif model_type == "megadescriptor":
        # Extract MegaDescriptor model specific parameters from YAML
        model_name = convert_none_str_to_none(get_with_print(model_details, 'model_name', 'hf-hub:BVRA/MegaDescriptor-T-224'))
        output_dim = convert_none_str_to_none(get_with_print(model_details, 'output_dim', None))
        forward_strat = convert_none_str_to_none(get_with_print(model_details, 'forward_strat', 'cat'))
        sequence_length = convert_none_str_to_none(get_with_print(model_details, 'sequence_length', None))
        num_frames = convert_none_str_to_none(get_with_print(model_details, 'num_frames', 1))
        dropout_rate = convert_none_str_to_none(get_with_print(model_details, 'dropout_rate', 0.1))
        checkpoint_path = convert_none_str_to_none(get_with_print(model_details, 'checkpoint_path', None))

        # Call megadescriptors_model_load with the extracted parameters
        model = megadescriptors_model_load(
            model_name=model_name, 
            output_dim=output_dim, 
            forward_strat=forward_strat, 
            sequence_length=sequence_length, 
            num_frames=num_frames, 
            dropout_rate=dropout_rate, 
            checkpoint_path=checkpoint_path
        )
    elif model_type == "vivit":
        # Extract ViViT model specific parameters from YAML
        vivit_model_name = convert_none_str_to_none(get_with_print(model_details, 'vivit_model_name', 'google/vivit-b-16x2-kinetics400'))
        output_dim = convert_none_str_to_none(get_with_print(model_details, 'output_dim', None))
        dropout_rate = convert_none_str_to_none(get_with_print(model_details, 'dropout_rate', 0.1))

        # Call vivit_model_load with the extracted parameters
        model = vivit_model_load(
            vivit_model_name=vivit_model_name, 
            output_dim=output_dim, 
            dropout_rate=dropout_rate
        )
    elif model_type == "timesformer":
        # Extract TimeSformer model specific parameters from YAML
        timesformer_model_name = convert_none_str_to_none(get_with_print(model_details, 'timesformer_model_name', 'facebook/timesformer-base-finetuned-k400'))
        output_dim = convert_none_str_to_none(get_with_print(model_details, 'output_dim', None))
        dropout_rate = convert_none_str_to_none(get_with_print(model_details, 'dropout_rate', 0.1))

        # Call timesformer_model_load with the extracted parameters
        model = timesformer_model_load(
            timesformer_model_name=timesformer_model_name, 
            output_dim=output_dim, 
            dropout_rate=dropout_rate
        )
    elif model_type == 'recurrent_decoder':
        print(f"model_details: {model_details}")
        model = RecurrentDecoder(
            v_size=convert_none_str_to_none(get_with_print(model_details, "v_size")),
            d_model=convert_none_str_to_none(get_with_print(model_details, "d_model")),
            nhead=convert_none_str_to_none(get_with_print(model_details, "nhead")),
            num_layers=convert_none_str_to_none(get_with_print(model_details, "num_layers")),
            dim_feedforward=convert_none_str_to_none(get_with_print(model_details, "dim_feedforward")),
            dropout=convert_none_str_to_none(get_with_print(model_details, "dropout_rate")), 
            activation=convert_none_str_to_none(get_with_print(model_details, "activation")),
            temperature=convert_none_str_to_none(get_with_print(model_details, "temperature")),
            image_model_name=convert_none_str_to_none(get_with_print(model_details, "image_model_name")),
            freeze_image_model=convert_none_str_to_none(get_with_print(model_details, "freeze_image_model"))
        )
    elif model_type in ["ResNet18", "ResNet50", "ResNet152", "VGG-16"]:
        model = image_model_load(model_type, get_with_print(model_details, 'embedding_dim', 256))
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