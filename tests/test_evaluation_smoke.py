import pickle
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")


def _device_params():
    params = [pytest.param("cpu", id="cpu")]
    if torch.cuda.is_available():
        params.append(pytest.param("cuda", marks=pytest.mark.gpu, id="cuda"))
    return params


class TinyEmbeddingModel(nn.Module):
    def forward(self, data):
        flattened = data.float().view(data.size(0), -1)
        return flattened[:, :2]


@pytest.fixture
def get_embeddings_module():
    pytest.importorskip("torchvision")
    pytest.importorskip("transformers")
    pytest.importorskip("timm")
    pytest.importorskip("open_clip")

    from evaluation import get_embeddings

    return get_embeddings


@pytest.fixture
def main_module():
    pytest.importorskip("torchvision")
    pytest.importorskip("transformers")
    pytest.importorskip("timm")
    pytest.importorskip("open_clip")

    import main

    return main


def _checkpoint_path(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "metadata": yaml.safe_dump(
                {
                    "dataloader_details": {
                        "num_frames": 2,
                    }
                }
            ),
            "model_state_dict": {},
        },
        checkpoint_path,
    )
    return checkpoint_path


def _write_embeddings(path, embeddings):
    with path.open("wb") as f:
        pickle.dump(embeddings, f)


def _examples_dataframe():
    return pd.DataFrame([["clip_a", "clip_b", "clip_c", "clip_d", "clip_e"]])


def _normal_embeddings(device_name):
    device = torch.device(device_name)
    return {
        "clip_a": torch.tensor([0.0, 0.0], device=device),
        "clip_b": torch.tensor([0.0, 0.1], device=device),
        "clip_c": torch.tensor([5.0, 0.0], device=device),
        "clip_d": torch.tensor([6.0, 0.0], device=device),
        "clip_e": torch.tensor([7.0, 0.0], device=device),
    }


def _image_majority_embeddings(device_name):
    base = _normal_embeddings(device_name)
    return {
        key: [value.clone(), value.clone()]
        for key, value in base.items()
    }


@pytest.mark.optional_deps
@pytest.mark.parametrize("device_name", _device_params())
def test_smoke_get_embeddings_with_fake_checkpoint_and_dataloader(get_embeddings_module, monkeypatch, tmp_path, device_name):
    checkpoint_path = _checkpoint_path(tmp_path)

    def fake_dataloader_creation(**kwargs):
        assert kwargs["num_frames"] == 2
        return [
            (
                ["clip_a", "clip_b"],
                torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            )
        ]

    monkeypatch.setattr(get_embeddings_module, "dataloader_creation", fake_dataloader_creation)
    monkeypatch.setattr(
        get_embeddings_module,
        "load_model_from_checkpoint",
        lambda checkpoint: TinyEmbeddingModel(),
    )

    embeddings = get_embeddings_module.get_embeddings(
        model_ckpt=str(checkpoint_path),
        transformations=None,
        cooccurrences_filepath="unused.json",
        clips_directory="unused/",
        num_frames=99,
        mode="Test",
        K=2,
        total_frames=2,
        zfill_num=4,
        is_override=False,
        override_value=None,
        masks=None,
        apply_mask_percentage=1.0,
        device=device_name,
        img_maj_vote=False,
    )

    assert sorted(embeddings) == ["clip_a", "clip_b"]
    assert embeddings["clip_a"].shape == (2,)
    assert embeddings["clip_a"].device.type == torch.device(device_name).type


def test_get_embeddings_resolve_device_rejects_missing_cuda(get_embeddings_module, monkeypatch):
    monkeypatch.setattr(get_embeddings_module.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA was requested"):
        get_embeddings_module.resolve_device("cuda")


@pytest.mark.parametrize("device_name", _device_params())
def test_smoke_get_metrics_normal_embeddings(tmp_path, device_name):
    from evaluation import get_metrics as metrics_module

    embedding_path = tmp_path / "embeddings.pkl"
    _write_embeddings(embedding_path, _normal_embeddings(device_name))

    metrics = metrics_module.get_metrics(
        [str(embedding_path)],
        _examples_dataframe(),
        img_maj_vote=False,
    )

    assert metrics == [(pytest.approx(1.0), pytest.approx(1.0), pytest.approx(3.0))]


@pytest.mark.parametrize("device_name", _device_params())
def test_smoke_get_metrics_image_majority_vote_embeddings(tmp_path, device_name):
    from evaluation import get_metrics as metrics_module

    embedding_path = tmp_path / "embeddings_imv.pkl"
    _write_embeddings(embedding_path, _image_majority_embeddings(device_name))

    metrics = metrics_module.get_metrics(
        [str(embedding_path)],
        _examples_dataframe(),
        img_maj_vote=True,
    )

    assert metrics == [(pytest.approx(1.0), pytest.approx(1.0), pytest.approx(3.0))]


def test_smoke_main_get_metrics_writes_metrics_file(main_module, tmp_path):
    embedding_path = tmp_path / "embeddings.pkl"
    dataframe_path = tmp_path / "examples.csv"
    _write_embeddings(embedding_path, _normal_embeddings("cpu"))
    _examples_dataframe().to_csv(dataframe_path, index=False)

    main_module.get_metrics(
        SimpleNamespace(
            embedding_path=str(embedding_path),
            dataframe_path=str(dataframe_path),
            img_maj_vote=False,
            mask_path=None,
        )
    )

    metrics_path = tmp_path / "embeddings_metrics.txt"
    assert metrics_path.exists()
    assert "Top-1 Accuracy: 1.0" in metrics_path.read_text(encoding="utf-8")
