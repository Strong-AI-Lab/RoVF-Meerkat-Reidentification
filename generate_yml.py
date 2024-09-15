import os
import copy
import yaml

def create_yaml_config_rovf_emb(
    mask_type=["mask", "no_mask", "50/50"],
    fps=[0.5, 1], 
    aug=["no_aug", "aug"],
):
    #with open(yml_path, 'w') as file:
    #    yaml.dump(config, file)

    filepath = "training_scripts/exp_metadata/hyperparameter_search/rovf_emb/"
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    config_orig = {
        'model_details': {
            "model_type": "recurrent_perceiver",
            "raw_input_dim": None,
            "embedding_dim": 384,
            "latent_dim": 384,
            "num_heads": 8,
            "num_latents": 257,
            "num_tf_layers": 2,
            "dropout_rate": 0.1,
            "output_dim": 384,
            "use_raw_input": False,
            "use_embeddings": True,
            "flatten_channels": False,
            "dino_model_name": "facebook/dinov2-small",
            "freeze_image_model": True
        },
        'dataloader_details': {
            'transformations': None,
            'cooccurrences_filepath': 'Dataset/meerkat_h5files/Cooccurrences.json',
            'clips_directory': 'Dataset/meerkat_h5files/clips/',
            'total_frames': 20,
            'num_frames': 10,
            'mode': 'positive_negative',
            'K': 20,
            'zfill_num': 4,
            'is_override': True,
            'override_value': 5000,
            'mask_path': 'Dataset/meerkat_h5files/masks/meerkat_masks.pkl',
            'apply_mask_percentage': 1.0
        },
        'training_details': {
            'epochs': 1,
            'batch_size': 30,
            'accumulation_steps': 1,
            'log_directory': '',
            'anchor_dino_model': None,
            'criterion_details': {
                'name': 'triplet_margin_loss',
                'margin': 1.0,
                'p': 2
            },
            'optimizer_details': {
                'name': 'adamw',
                'lr': 0.0001,
                'weight_decay': 0.01
            },
            'scheduler_details': {
                'name': 'warmup_cosine_decay_scheduler',
                'warmup_steps': 250,
                'decay_steps': 4750,
                'start_lr': 0.001,
                'max_lr': 0.005,
                'end_lr': 0.0001
            },
            'anchor_function_details': {
                'similarity_measure': 'euclidean_distance',
                'type': 'hard'
            },
            'clip_value': None
        }
    }

    for mask in mask_type:
        for f in fps:
            for aug_type in aug:
                config = copy.deepcopy(config_orig)
                mask_filename = None
                if mask == "mask":
                    config['dataloader_details']['mask_path'] = 'Dataset/meerkat_h5files/masks/meerkat_masks.pkl'
                    config['dataloader_details']['apply_mask_percentage'] = 1.0
                    mask_filename = 'mask'
                elif mask == "no_mask":
                    config['dataloader_details']['mask_path'] = ''
                    config['dataloader_details']['apply_mask_percentage'] = 1.0 # this can stay as 1.0
                    mask_filename = 'no_mask'
                elif mask == "50/50":
                    config['dataloader_details']['mask_path'] = 'Dataset/meerkat_h5files/masks/meerkat_masks.pkl'
                    config['dataloader_details']['apply_mask_percentage'] = 0.5
                    mask_filename = '50_50'
                else:
                    raise ValueError("Invalid mask type")
                f_filename = None
                if f == 0.5:
                    config['dataloader_details']['total_frames'] = 20
                    config['dataloader_details']['num_frames'] = 5
                    fps_filename = '0p5_fps'
                elif f == 1:
                    config['dataloader_details']['total_frames'] = 20
                    config['dataloader_details']['num_frames'] = 10
                    fps_filename = '1_fps'
                else:
                    raise ValueError("Invalid fps")
                aug_filename = None
                if aug_type == "no_aug":
                    config['dataloader_details']['transformations'] = None
                    aug_filename = 'no_aug'
                elif aug_type == "aug":
                    config['dataloader_details']['transformations'] = [
                        'random_resized_crop',
                        'horizontal_flip',
                        'gaussian_blur',
                        'color_jitter'
                    ]
                    aug_filename = 'aug'
                else:
                    raise ValueError("Invalid augmentation type")
                
                config['training_details']['log_directory'] = f"results/hyperparameter_search/rovf_emb/{mask_filename}/{fps_filename}/{aug_filename}/"

                # in results/hyperparameter_search/rovf/ create a folder of this name
                directory = f"results/hyperparameter_search/rovf_emb/{mask_filename}/{fps_filename}/{aug_filename}/"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                
                # save to yaml file with the following name
                filename = f"{filepath}{mask_filename}_{fps_filename}_{aug_filename}.yml"
                with open(filename, 'w') as file:
                    yaml.dump(config, file)

def create_yaml_config_rovf_perc(
    mask_type=["mask", "no_mask", "50/50"],
    fps=[0.5, 1], 
    aug=["no_aug", "aug"],
):
    #with open(yml_path, 'w') as file:
    #    yaml.dump(config, file)

    filepath = "training_scripts/exp_metadata/hyperparameter_search/rovf_perc/"
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    config_orig = {
        'model_details': {
            "model_type": "recurrent_perceiver",
            "raw_input_dim": None,
            "embedding_dim": 3,
            "latent_dim": 384,
            "num_heads": 8,
            "num_latents": 257,
            "num_tf_layers": 2,
            "dropout_rate": 0.1,
            "output_dim": 384,
            "use_raw_input": True,
            "use_embeddings": False,
            "flatten_channels": False,
            "dino_model_name": "facebook/dinov2-small",
            "freeze_image_model": True
        },
        'dataloader_details': {
            'transformations': None,
            'cooccurrences_filepath': 'Dataset/meerkat_h5files/Cooccurrences.json',
            'clips_directory': 'Dataset/meerkat_h5files/clips/',
            'total_frames': 20,
            'num_frames': 10,
            'mode': 'positive_negative',
            'K': 20,
            'zfill_num': 4,
            'is_override': True,
            'override_value': 5000,
            'mask_path': 'Dataset/meerkat_h5files/masks/meerkat_masks.pkl',
            'apply_mask_percentage': 1.0
        },
        'training_details': {
            'epochs': 1,
            'batch_size': 6,
            'accumulation_steps': 5,
            'log_directory': '',
            'anchor_dino_model': None,
            'criterion_details': {
                'name': 'triplet_margin_loss',
                'margin': 1.0,
                'p': 2
            },
            'optimizer_details': {
                'name': 'adamw',
                'lr': 0.0001,
                'weight_decay': 0.01
            },
            'scheduler_details': {
                'name': 'warmup_cosine_decay_scheduler',
                'warmup_steps': 250,
                'decay_steps': 4750,
                'start_lr': 0.001,
                'max_lr': 0.005,
                'end_lr': 0.0001
            },
            'anchor_function_details': {
                'similarity_measure': 'euclidean_distance',
                'type': 'hard'
            },
            'clip_value': None
        }
    }

    for mask in mask_type:
        for f in fps:
            for aug_type in aug:
                config = copy.deepcopy(config_orig)
                mask_filename = None
                if mask == "mask":
                    config['dataloader_details']['mask_path'] = 'Dataset/meerkat_h5files/masks/meerkat_masks.pkl'
                    config['dataloader_details']['apply_mask_percentage'] = 1.0
                    mask_filename = 'mask'
                elif mask == "no_mask":
                    config['dataloader_details']['mask_path'] = ''
                    config['dataloader_details']['apply_mask_percentage'] = 1.0 # this can stay as 1.0
                    mask_filename = 'no_mask'
                elif mask == "50/50":
                    config['dataloader_details']['mask_path'] = 'Dataset/meerkat_h5files/masks/meerkat_masks.pkl'
                    config['dataloader_details']['apply_mask_percentage'] = 0.5
                    mask_filename = '50_50'
                else:
                    raise ValueError("Invalid mask type")
                f_filename = None
                if f == 0.5:
                    config['dataloader_details']['total_frames'] = 20
                    config['dataloader_details']['num_frames'] = 5
                    fps_filename = '0p5_fps'
                elif f == 1:
                    config['dataloader_details']['total_frames'] = 20
                    config['dataloader_details']['num_frames'] = 10
                    fps_filename = '1_fps'
                else:
                    raise ValueError("Invalid fps")
                aug_filename = None
                if aug_type == "no_aug":
                    config['dataloader_details']['transformations'] = None
                    aug_filename = 'no_aug'
                elif aug_type == "aug":
                    config['dataloader_details']['transformations'] = [
                        'random_resized_crop',
                        'horizontal_flip',
                        'gaussian_blur',
                        'color_jitter'
                    ]
                    aug_filename = 'aug'
                else:
                    raise ValueError("Invalid augmentation type")
                
                config['training_details']['log_directory'] = f"results/hyperparameter_search/rovf_perc/{mask_filename}/{fps_filename}/{aug_filename}/"

                # in results/hyperparameter_search/rovf/ create a folder of this name
                directory = f"results/hyperparameter_search/rovf_perc/{mask_filename}/{fps_filename}/{aug_filename}/"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                
                # save to yaml file with the following name
                filename = f"{filepath}{mask_filename}_{fps_filename}_{aug_filename}.yml"
                with open(filename, 'w') as file:
                    yaml.dump(config, file)


def create_yaml_config_dino(
    mask_type=["mask", "no_mask", "50/50"],
    fps=[0.5, 1],
    aug=["no_aug", "aug"],
):
    filepath = "training_scripts/exp_metadata/hyperparameter_search/dino/"
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    config_orig = {
        'model_details': {
            'model_type': 'dino',
            'dino_model_name': 'facebook/dinov2-small',
            'dropout_rate': 0.1,
            'output_dim': None,
            'forward_strat': 'average',
            'sequence_length': None,
            'num_frames': 10
        },
        'dataloader_details': {
            'transformations': None,
            'cooccurrences_filepath': 'Dataset/meerkat_h5files/Cooccurrences.json',
            'clips_directory': 'Dataset/meerkat_h5files/clips/',
            'total_frames': 20,
            'num_frames': 10,
            'mode': 'positive_negative',
            'K': 20,
            'zfill_num': 4,
            'is_override': True,
            'override_value': 5000,
            'mask_path': 'Dataset/meerkat_h5files/masks/meerkat_masks.pkl',
            'apply_mask_percentage': 1.0
        },
        'training_details': {
            'epochs': 1,
            'batch_size': 30,
            'accumulation_steps': 1,
            'log_directory': '',
            'anchor_dino_model': None,
            'criterion_details': {
                'name': 'triplet_margin_loss',
                'margin': 1.0,
                'p': 2
            },
            'optimizer_details': {
                'name': 'adamw',
                'lr': 0.0001,
                'weight_decay': 0.01
            },
            'scheduler_details': {
                'name': 'warmup_cosine_decay_scheduler',
                'warmup_steps': 250,
                'decay_steps': 4750,
                'start_lr': 0.001,
                'max_lr': 0.005,
                'end_lr': 0.0001
            },
            'anchor_function_details': {
                'similarity_measure': 'euclidean_distance',
                'type': 'hard'
            },
            'clip_value': None
        }
    }

    for mask in mask_type:
        for f in fps:
            for aug_type in aug:
                config = copy.deepcopy(config_orig)
                
                # Mask configuration
                if mask == "mask":
                    config['dataloader_details']['mask_path'] = 'Dataset/meerkat_h5files/masks/meerkat_masks.pkl'
                    config['dataloader_details']['apply_mask_percentage'] = 1.0
                    mask_filename = 'mask'
                elif mask == "no_mask":
                    config['dataloader_details']['mask_path'] = ''
                    config['dataloader_details']['apply_mask_percentage'] = 1.0
                    mask_filename = 'no_mask'
                elif mask == "50/50":
                    config['dataloader_details']['mask_path'] = 'Dataset/meerkat_h5files/masks/meerkat_masks.pkl'
                    config['dataloader_details']['apply_mask_percentage'] = 0.5
                    mask_filename = '50_50'
                else:
                    raise ValueError("Invalid mask type")
                
                # FPS configuration
                if f == 0.5:
                    config['dataloader_details']['total_frames'] = 20
                    config['dataloader_details']['num_frames'] = 5
                    config['model_details']['num_frames'] = 5
                    fps_filename = '0p5_fps'
                elif f == 1:
                    config['dataloader_details']['total_frames'] = 20
                    config['dataloader_details']['num_frames'] = 10
                    config['model_details']['num_frames'] = 10
                    fps_filename = '1_fps'
                else:
                    raise ValueError("Invalid fps")
                
                # Augmentation configuration
                if aug_type == "no_aug":
                    config['dataloader_details']['transformations'] = None
                    aug_filename = 'no_aug'
                elif aug_type == "aug":
                    config['dataloader_details']['transformations'] = [
                        'random_resized_crop',
                        'horizontal_flip',
                        'gaussian_blur',
                        'color_jitter'
                    ]
                    aug_filename = 'aug'
                else:
                    raise ValueError("Invalid augmentation type")
                
                config['training_details']['log_directory'] = f"results/hyperparameter_search/dino/{mask_filename}/{fps_filename}/{aug_filename}/"

                # Create directory
                directory = f"results/hyperparameter_search/dino/{mask_filename}/{fps_filename}/{aug_filename}/"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                
                # Save to YAML file
                filename = f"{filepath}{mask_filename}_{fps_filename}_{aug_filename}.yml"
                with open(filename, 'w') as file:
                    yaml.dump(config, file)

if __name__ == "__main__":
    create_yaml_config_dino()
    create_yaml_config_rovf_emb()
    create_yaml_config_rovf_perc()