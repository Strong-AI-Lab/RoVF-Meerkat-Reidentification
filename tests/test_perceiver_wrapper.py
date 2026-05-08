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


def test_perceiver_v1_and_v2_match_without_video_embedding():
    torch.manual_seed(1)
    v1 = Perceiver(**_perceiver_kwargs())
    v2 = PerceiverV2(**_perceiver_kwargs())
    v2.load_state_dict(v1.state_dict())
    embeddings = torch.randn(2, 3, 5)

    v1_output = v1(embeddings=embeddings, is_reset_latents=True)
    v2_output = v2(embeddings=embeddings, is_reset_latents=True)

    assert torch.allclose(v1_output, v2_output)


def test_video_embedding_only_changes_v1_initial_latents():
    torch.manual_seed(1)
    v1 = Perceiver(**_perceiver_kwargs())
    v2 = PerceiverV2(**_perceiver_kwargs())
    v2.load_state_dict(v1.state_dict())
    embeddings = torch.randn(2, 3, 5)
    video_emb = torch.randn(2, 4, 8)

    v1_without_video = v1(embeddings=embeddings, is_reset_latents=True)
    v1_with_video = v1(
        embeddings=embeddings,
        video_emb=video_emb,
        is_reset_latents=True,
    )
    v2_without_video = v2(embeddings=embeddings, is_reset_latents=True)
    v2_with_video = v2(
        embeddings=embeddings,
        video_emb=video_emb,
        is_reset_latents=True,
    )

    assert not torch.allclose(v1_without_video, v1_with_video)
    assert torch.allclose(v2_without_video, v2_with_video)


@pytest.mark.parametrize("model_cls", [Perceiver, PerceiverV2])
def test_perceiver_with_video_embedding_positional_table_keeps_output_shape(model_cls):
    model = model_cls(**_perceiver_kwargs(use_video_emb=True))
    embeddings = torch.randn(2, 3, 5)
    video_emb = torch.randn(2, 4, 8)

    output = model(
        embeddings=embeddings,
        video_emb=video_emb,
        is_reset_latents=True,
    )

    assert output.shape == (2, 6)
