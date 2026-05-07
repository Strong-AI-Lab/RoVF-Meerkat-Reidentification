import pytest

torch = pytest.importorskip("torch")

from models.perceiver_wrapper import CrossAttention, Perceiver, PerceiverV2


def _perceiver_kwargs(**overrides):
    kwargs = {
        "raw_input_dim": 3,
        "embedding_dim": 5,
        "latent_dim": 8,
        "num_heads": 2,
        "num_latents": 4,
        "num_transformer_layers": 1,
        "dropout": 0.0,
        "output_dim": 6,
        "use_raw_input": False,
        "use_embeddings": True,
    }
    kwargs.update(overrides)
    return kwargs


def test_cross_attention_accepts_sequence_data():
    attention = CrossAttention(latent_dim=8, data_dim=5, num_heads=2)
    latents = torch.randn(2, 4, 8)
    data = torch.randn(2, 7, 5)

    output = attention(latents, data)

    assert output.shape == (2, 4, 8)


def test_cross_attention_accepts_flat_data():
    attention = CrossAttention(latent_dim=8, data_dim=5, num_heads=2)
    latents = torch.randn(2, 4, 8)
    data = torch.randn(2, 10)

    output = attention(latents, data)

    assert output.shape == (2, 4, 8)


@pytest.mark.parametrize("model_cls", [Perceiver, PerceiverV2])
def test_perceiver_embeddings_only_output_shape(model_cls):
    model = model_cls(**_perceiver_kwargs())
    embeddings = torch.randn(2, 3, 5)

    output = model(embeddings=embeddings)

    assert output.shape == (2, 6)


@pytest.mark.parametrize("model_cls", [Perceiver, PerceiverV2])
def test_perceiver_raw_input_uses_raw_input_dim(model_cls):
    model = model_cls(
        **_perceiver_kwargs(
            raw_input_dim=3,
            embedding_dim=5,
            use_raw_input=True,
            use_embeddings=False,
        )
    )
    raw_input = torch.randn(2, 4, 4, 3)

    output = model(raw_input=raw_input)

    assert output.shape == (2, 6)


@pytest.mark.parametrize("model_cls", [Perceiver, PerceiverV2])
def test_perceiver_reset_latents_clears_cached_state(model_cls):
    model = model_cls(**_perceiver_kwargs())
    embeddings = torch.randn(2, 3, 5)

    model(embeddings=embeddings)
    assert model.latents is not None

    model(embeddings=embeddings, is_reset_latents=True)
    assert model.latents is None


def test_perceiver_requires_embeddings_when_configured_for_embeddings():
    model = Perceiver(**_perceiver_kwargs())

    with pytest.raises(ValueError, match="embeddings is required"):
        model()


def test_perceiver_requires_raw_input_when_configured_for_raw_input():
    model = Perceiver(
        **_perceiver_kwargs(use_raw_input=True, use_embeddings=False)
    )

    with pytest.raises(ValueError, match="raw_input is required"):
        model()


def test_perceiver_rejects_dual_raw_and_embedding_inputs():
    with pytest.raises(AssertionError, match="both cannot be True"):
        Perceiver(**_perceiver_kwargs(use_raw_input=True, use_embeddings=True))
