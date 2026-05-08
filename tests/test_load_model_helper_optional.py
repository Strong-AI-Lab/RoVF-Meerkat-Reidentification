import yaml

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")


@pytest.mark.optional_deps
def test_load_model_from_checkpoint_uses_cpu_checkpoint_and_loader(monkeypatch, tmp_path):
    pytest.importorskip("torchvision")
    pytest.importorskip("transformers")
    pytest.importorskip("timm")
    pytest.importorskip("open_clip")

    from training_functions import load_model_helper

    class DummyDino(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(1, 1)

    dummy = DummyDino()

    def fake_dino_model_load(**kwargs):
        assert kwargs["dino_model_name"] == "dummy-dino"
        assert kwargs["output_dim"] == 1
        return dummy

    monkeypatch.setattr(load_model_helper, "dino_model_load", fake_dino_model_load)

    config = {
        "model_details": {
            "model_type": "dino",
            "dino_model_name": "dummy-dino",
            "output_dim": 1,
            "forward_strat": "cls",
            "sequence_length": None,
            "num_frames": 2,
            "dropout_rate": 0.0,
        }
    }
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "metadata": yaml.safe_dump(config),
            "model_state_dict": dummy.state_dict(),
        },
        checkpoint_path,
    )

    model = load_model_helper.load_model_from_checkpoint(str(checkpoint_path))

    assert model is dummy
