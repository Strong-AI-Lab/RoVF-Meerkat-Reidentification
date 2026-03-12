# Models Folder Guide

This folder contains model wrappers used by `training_functions/load_model_helper.py` and `main.py`.

## Which wrapper to use

- `dinov2_wrapper.py`: frame/video embedding wrapper around DINOv2.
- `bioCLIP_wrapper.py`: frame/video embedding wrapper around BioCLIP.
- `MegaDescriptor_wrapper.py`: frame/video embedding wrapper around MegaDescriptor.
- `ViViT_wrapper.py`: HuggingFace ViViT wrapper.
- `TimeSformer_wrapper.py`: HuggingFace TimeSformer wrapper.
- `recurrent_wrapper.py`: combines an image model with a recurrent head (`Perceiver`, `PerceiverV2`, `LSTM`, `GRU`).
- `recurrent_decoder.py`: recurrent decoder variant.
- `perceiver_wrapper.py`: reusable Perceiver components.

## Perceiver variants

Two Perceiver implementations are available and selected by configuration:

- `Perceiver` (v1): supports optional `video_emb` injection on first step and applies an extra residual around the transformer block.
- `PerceiverV2` (v2): supports positional embeddings via `add_pos_emb` in `forward` and uses the plain transformer output path.

In config terms (`model_details.model_type`):

- `recurrent_perceiver` / `recurrent` -> uses `Perceiver` (v1)
- `recurrent_perceiverv2` -> uses `PerceiverV2` (v2)

## RoVF variants configuration

The Recurrence over Video Frames (RoVF) architecture has several variants depending on how spatial and temporal information is integrated. You can configure these in `RecurrentWrapper` by combining `is_append_avg_emb` with the Perceiver `type_`.

- **RoVF-st:** adds the average frame embedding to the final video embedding.
	- Setup: `is_append_avg_emb=True`
	- Version: works with both `type_="v1"` and `type_="v2"`

- **RoVF-af:** initializes the recurrent hidden state using averaged patch embeddings.
	- Setup: `is_append_avg_emb=False`
	- Version: requires `type_="v1"` (`Perceiver`), because v1 supports first-step `video_emb` latent initialization

- **RoVF-af-st:** combines averaged-patch initialization and final average-embedding appending.
	- Setup: `is_append_avg_emb=True`
	- Version: requires `type_="v1"`

- **Standard RoVF:** baseline setup with learnable latent parameters and no average-frame integration.
	- Setup: `is_append_avg_emb=False`
	- Version: typically configured with `type_="v2"` (`PerceiverV2`)

## Cleanup conventions for this folder

When editing wrappers, keep these constraints:

1. Keep production code separate from ad-hoc tests (move experiments to notebooks or dedicated test files).
2. Avoid `sys.path.append("..")`; use package imports from repo root.
3. Keep constructor args backward compatible with YAML configs unless migration is planned.
4. If you change tensor shape behavior, update both this file and `training_functions/load_model_helper.py` docs/comments.
5. Prefer small, explicit helper methods over large monolithic `forward` blocks.
