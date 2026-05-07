import yaml

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")


def _device_params():
    params = [pytest.param("cpu", id="cpu")]
    if torch.cuda.is_available():
        params.append(pytest.param("cuda", marks=pytest.mark.gpu, id="cuda"))
    return params


class TinyProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, x):
        return self.linear(x)


def _base_model_details(model_type):
    return {
        "model_type": model_type,
        "dino_model_name": "dummy-dino",
        "model_name": "dummy-model",
        "output_dim": 2,
        "embedding_dim": 2,
        "raw_input_dim": 2,
        "latent_dim": 2,
        "num_heads": 1,
        "num_latents": 2,
        "num_tf_layers": 1,
        "dropout_rate": 0.0,
        "use_raw_input": False,
        "use_embeddings": True,
        "flatten_channels": False,
        "freeze_image_model": True,
        "is_append_avg_emb": False,
        "forward_strat": "cls",
        "sequence_length": None,
        "num_frames": 2,
    }


@pytest.fixture
def load_model_helper():
    pytest.importorskip("torchvision")
    pytest.importorskip("transformers")
    pytest.importorskip("timm")
    pytest.importorskip("open_clip")

    from training_functions import load_model_helper

    return load_model_helper


@pytest.mark.optional_deps
@pytest.mark.parametrize(
    ("model_type", "loader_name"),
    [
        ("dino", "dino_model_load"),
        ("recurrent", "recurrent_model_perceiver_load"),
        ("recurrent_perceiverv2", "recurrent_model_perceiver_loadv2"),
        ("LSTM", "LSTM_model_load"),
        ("GRU", "GRU_model_load"),
        ("ResNet18", "image_model_load"),
    ],
)
@pytest.mark.parametrize("device_name", _device_params())
def test_smoke_load_checkpoint_and_forward(load_model_helper, monkeypatch, tmp_path, model_type, loader_name, device_name):
    checkpoint_model = TinyProjection()
    loaded_model = TinyProjection()

    def fake_loader(*args, **kwargs):
        return loaded_model

    monkeypatch.setattr(load_model_helper, loader_name, fake_loader)

    checkpoint_path = tmp_path / f"{model_type}.pt"
    torch.save(
        {
            "metadata": yaml.safe_dump({"model_details": _base_model_details(model_type)}),
            "model_state_dict": checkpoint_model.state_dict(),
        },
        checkpoint_path,
    )

    model = load_model_helper.load_model_from_checkpoint(str(checkpoint_path))
    model.to(device_name)
    model.eval()

    sample = torch.tensor([[1.0, 2.0]], device=device_name)
    with torch.no_grad():
        output = model(sample)

    assert output.shape == (1, 2)
    assert output.device.type == torch.device(device_name).type
    for expected, actual in zip(checkpoint_model.parameters(), model.cpu().parameters()):
        assert torch.allclose(expected, actual)
