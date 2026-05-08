# Models Guide

This directory contains the model wrappers used by `main.py`, checkpoint loading, embedding generation, and the training/evaluation helpers. Most experiments select a model through `model_details.model_type` in a YAML file under `training_scripts/exp_metadata/`.

## Wrapper Map

- `dinov2_wrapper.py`: DINOv2 image/video embedding wrapper.
- `bioCLIP_wrapper.py`: BioCLIP image/video embedding wrapper.
- `MegaDescriptor_wrapper.py`: MegaDescriptor image/video embedding wrapper.
- `ViViT_wrapper.py`: HuggingFace ViViT video wrapper.
- `TimeSformer_wrapper.py`: HuggingFace TimeSformer video wrapper.
- `recurrent_wrapper.py`: RoVF recurrent wrapper around an image model plus a Perceiver, LSTM, or GRU recurrent head.
- `perceiver_wrapper.py`: reusable Perceiver blocks and the two Perceiver recurrent-head implementations.
- `recurrent_decoder.py`: recurrent decoder variant.

`training_functions/load_model_helper.py` is the central factory for constructing these models from YAML metadata or checkpoint metadata.

## YAML Model Types

The key `model_details.model_type` controls which wrapper is instantiated:

- `dino`: `DINOv2VideoWrapper`
- `bioclip`: `BioCLIPVideoWrapper`
- `megadescriptor`: `MegaDescriptorVideoWrapper`
- `vivit`: `ViViTWrapper`
- `timesformer`: `TimeSformerWrapper`
- `recurrent` or `recurrent_perceiver`: `RecurrentWrapper` with `Perceiver` v1
- `recurrent_perceiverv2`: `RecurrentWrapper` with `PerceiverV2`
- `LSTM`: `RecurrentWrapper` with an LSTM recurrent head
- `GRU`: `RecurrentWrapper` with a GRU recurrent head
- `recurrent_decoder`: `RecurrentDecoder`
- `ResNet18`, `ResNet50`, `ResNet152`, `VGG-16`: torchvision image backbones with projectors

The YAML files also define dataloader paths, mask use, frame sampling, optimizer settings, scheduler settings, loss configuration, and logging paths. Checkpoints store the YAML metadata string, so `load_model_from_checkpoint()` can reconstruct the architecture before loading weights.

## RoVF Recurrent Flow

`RecurrentWrapper` processes a video as a sequence of frame embeddings:

1. Each frame is passed through the image model, usually DINOv2.
2. The image-model output sequence is detached when `freeze_image_model=True`.
3. The recurrent head receives each frame embedding in sequence.
4. The final recurrent output is returned, optionally with an average frame embedding added.

For DINOv2-small, the frame embedding dimension is typically `384`. The full-model RoVF-ST configs use `num_frames: 5`, `num_latents: 257`, `latent_dim: 384`, and `output_dim: 384`.

## Perceiver v1 vs v2

There are two Perceiver recurrent-head classes:

- `Perceiver` v1 supports first-step `video_emb` initialization.
- `PerceiverV2` ignores `video_emb` and always starts from learned recurrent latents.

Both implementations now share the same core mechanics:

- learned latent parameters
- learned latent positional embeddings added in `forward`
- cross-attention from latents to frame data
- the same transformer encoder output path
- optional flattened output projection
- cached recurrent latents across frames, reset at video boundaries by `RecurrentWrapper`

The intended behavioral difference is only whether averaged frame/patch embeddings can initialize the first recurrent hidden state. That v1-only behavior is what enables RoVF-af and RoVF-af-st.

## RoVF Variant Mapping

RoVF variants are controlled by the Perceiver implementation and `is_append_avg_emb`.

- **Standard RoVF**
  - `model_type: recurrent_perceiverv2`
  - `is_append_avg_emb: false`
  - Starts from learned recurrent latents and returns the final recurrent output.

- **RoVF-st**
  - `model_type: recurrent_perceiverv2`
  - `is_append_avg_emb: true`
  - Starts from learned recurrent latents, then adds the average frame embedding to the final video embedding.

- **RoVF-af**
  - `model_type: recurrent_perceiver`
  - `is_append_avg_emb: false`
  - Initializes the first recurrent hidden state from averaged frame/patch embeddings and returns the final recurrent output.

- **RoVF-af-st**
  - `model_type: recurrent_perceiver`
  - `is_append_avg_emb: true`
  - Initializes from averaged frame/patch embeddings and also adds the average frame embedding to the final output.

Use `recurrent_perceiverv2` for pure ST experiments. Using v1 with `is_append_avg_emb=True` is AF-ST, not pure ST, because v1 consumes `video_emb` on the first recurrent step.

## Best Published RoVF-ST Config

The full-model YAMLs for the best RoVF-ST setup are:

- `training_scripts/exp_metadata/full_model_training/rovf_s_st_mask_no_aug_0p5_fps_meerkat.yml`
- `training_scripts/exp_metadata/full_model_training/rovf_s_st_mask_no_aug_0p5_fps_polarbear.yml`

Important model fields:

```yaml
model_type: recurrent_perceiverv2
dino_model_name: facebook/dinov2-small
is_append_avg_emb: true
freeze_image_model: true
embedding_dim: 384
latent_dim: 384
output_dim: 384
num_heads: 8
num_latents: 257
num_tf_layers: 2
dropout_rate: 0.1
use_embeddings: true
use_raw_input: false
flatten_channels: false
```

Important dataloader/training fields:

```yaml
num_frames: 5
total_frames: 20
apply_mask_percentage: 1.0
transformations: null
batch_size: 30
epochs: 10
criterion: triplet_margin_loss
optimizer: adamw
```

## Checkpoint Loading

Use `training_functions.load_model_helper.load_model_from_checkpoint()` for trained checkpoints:

```python
from training_functions.load_model_helper import load_model_from_checkpoint

model = load_model_from_checkpoint("path/to/checkpoint_epoch_10.pt")
model.eval()
```

The loader reads checkpoint metadata, rebuilds the corresponding wrapper, and loads `model_state_dict`. If model code changes architectural parameters or state-dict names, old checkpoints may fail to load. Keep constructor signatures and parameter names backward compatible unless a deliberate migration is planned.

### Manual RoVF Checkpoint Loading

Prefer `load_model_from_checkpoint()` for normal use. Manual loading is useful when debugging the model construction path or when you need to customize the wrapper before loading weights. The manually supplied architecture must match the checkpoint exactly.

Example for a RoVF-ST no-mask style checkpoint, equivalent to the recurrent Perceiver branch inside `load_model_from_checkpoint()`:

```python
import torch
import yaml

from models.recurrent_wrapper import RecurrentWrapper

checkpoint_path = "path/to/full_model_training/rovf_st_no_mask_example/checkpoint_epoch_2.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = torch.load(checkpoint_path, map_location="cpu")
metadata = yaml.safe_load(checkpoint["metadata"])
model_details = metadata["model_details"]
perceiver_type = "v2" if model_details["model_type"] == "recurrent_perceiverv2" else "v1"

perceiver_config = {
    "raw_input_dim": model_details.get("raw_input_dim", 384),
    "embedding_dim": model_details.get("embedding_dim", 384),
    "latent_dim": model_details.get("latent_dim", 384),
    "num_heads": model_details.get("num_heads", 8),
    "num_latents": model_details.get("num_latents", 257),
    "num_transformer_layers": model_details.get("num_tf_layers", 2),
    "dropout": model_details.get("dropout_rate", 0.1),
    "output_dim": model_details.get("output_dim", 384),
    "use_raw_input": model_details.get("use_raw_input", False),
    "use_embeddings": model_details.get("use_embeddings", True),
    "flatten_channels": model_details.get("flatten_channels", False),
}

model = RecurrentWrapper(
    perceiver_config=perceiver_config,
    model_name=model_details.get("dino_model_name", "facebook/dinov2-small"),
    dropout_rate=model_details.get("dropout_rate", 0.1),
    freeze_image_model=model_details.get("freeze_image_model", True),
    is_append_avg_emb=model_details.get("is_append_avg_emb", False),
    type_=perceiver_type,
)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()
```

For pure RoVF-ST checkpoints, `model_type` should normally be `recurrent_perceiverv2`, which selects `type_="v2"`. RoVF-AF and RoVF-AF-ST checkpoints use `recurrent_perceiver`, which selects `type_="v1"` because v1 supports averaged frame/patch embedding initialization.

## Development Notes

- Keep YAML `model_details` and checkpoint loading behavior aligned.
- Avoid changing tensor shapes without updating tests and checkpoint-loading expectations.
- Prefer adding explicit tests in `tests/test_perceiver_wrapper.py` and `tests/test_smoke_model_loading.py` when changing recurrent behavior.
- Avoid relying on ad-hoc `__main__` experiments inside model files for verification; use focused tests or small external scripts.
- If a method distinction matters for a paper result, represent it explicitly in YAML and document it here.
