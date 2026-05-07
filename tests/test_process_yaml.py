from training_functions.process_yaml import process_yaml_for_training


def test_process_yaml_for_training_merges_nested_list_of_dicts(tmp_path):
    yaml_path = tmp_path / "config.yml"
    yaml_path.write_text(
        """
model_details:
  - model_type: recurrent
  - perceiver:
    - latent_dim: 384
    - num_heads: 12
dataloader_details:
  - num_frames: 5
  - mode: Train
""",
        encoding="utf-8",
    )

    config = process_yaml_for_training(yaml_path)

    assert config == {
        "model_details": {
            "model_type": "recurrent",
            "perceiver": {"latent_dim": 384, "num_heads": 12},
        },
        "dataloader_details": {"num_frames": 5, "mode": "Train"},
    }


def test_process_yaml_for_training_returns_none_for_invalid_yaml(tmp_path):
    yaml_path = tmp_path / "bad.yml"
    yaml_path.write_text("model_details: [unterminated", encoding="utf-8")

    assert process_yaml_for_training(yaml_path) is None
