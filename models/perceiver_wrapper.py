import torch
import torch.nn as nn

"""Perceiver-based recurrence building blocks used by recurrent wrappers.

This module intentionally exposes both `Perceiver` (v1) and `PerceiverV2` (v2),
because both variants are used in training/evaluation entrypoints.
"""


class TransformerEncoder(nn.Module):
    def __init__(self, latent_dim, num_heads, num_layers, dropout):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        norm = nn.LayerNorm(normalized_shape=latent_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, norm=norm
        )

    def forward(self, src, mask=None):
        return self.transformer_encoder(src, mask=mask)


class TransformerDecoder(nn.Module):
    def __init__(self, latent_dim, num_heads, num_layers, dropout):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        return self.transformer_decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
        )


class CrossAttention(nn.Module):
    def __init__(self, latent_dim, data_dim, num_heads):
        super().__init__()
        self.data_dim = data_dim
        self.query_proj = nn.Linear(latent_dim, latent_dim)
        self.key_proj = nn.Linear(data_dim, latent_dim)
        self.value_proj = nn.Linear(data_dim, latent_dim)
        self.attention = nn.MultiheadAttention(latent_dim, num_heads, batch_first=True)

    def forward(self, latents, data):
        query = self.query_proj(latents)

        if len(data.size()) == 2:
            data = data.view(data.size(0), -1, self.data_dim)

        key = self.key_proj(data)
        value = self.value_proj(data)

        attn_output, _ = self.attention(query, key, value)
        return attn_output


class Perceiver(nn.Module):
    def __init__(
        self,
        raw_input_dim,
        embedding_dim,
        latent_dim,
        num_heads,
        num_latents,
        num_transformer_layers,
        dropout,
        output_dim,
        use_raw_input=True,
        use_embeddings=True,
        flatten_channels=False,
        use_video_emb=False,
    ):
        super().__init__()

        self.latents_p = nn.Parameter(
            nn.init.xavier_uniform_(torch.randn(num_latents, latent_dim)) * (latent_dim**0.5)
        )
        self.latents = None
        self.video_emb = None

        self.use_video_emb = use_video_emb
        pe_nlatents = num_latents + 1 if use_video_emb else num_latents
        self.positional_embeddings = nn.init.xavier_uniform_(
            nn.Parameter(torch.randn(pe_nlatents, latent_dim))
        )

        self.num_latents = num_latents
        self.raw_input_dim = raw_input_dim
        self.embedding_dim = embedding_dim
        self.use_raw_input = use_raw_input
        self.use_embeddings = use_embeddings
        assert (use_raw_input or use_embeddings) and not (use_raw_input and use_embeddings), (
            "At least one of use_raw_input or use_embeddings must be True and both cannot be True at once"
        )
        self.flatten_channels = flatten_channels

        if use_raw_input:
            self.raw_cross_attention = CrossAttention(latent_dim, raw_input_dim, num_heads)
        if use_embeddings:
            self.embedding_cross_attention = CrossAttention(latent_dim, embedding_dim, num_heads)

        self.transformer = TransformerEncoder(
            latent_dim, num_heads, num_transformer_layers, dropout
        )

        if output_dim is not None:
            self.output_layer = nn.Linear(latent_dim * num_latents, output_dim)
        else:
            self.output_layer = None

        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(latent_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def reset_latents(self):
        self.latents = None
        self.video_emb = None

    def forward(self, raw_input=None, embeddings=None, video_emb=None, is_reset_latents=False):
        if not self.use_raw_input and not self.use_embeddings:
            raise ValueError("At least one of use_raw_input or use_embeddings must be True")

        if self.use_raw_input and raw_input is None:
            raise ValueError("raw_input is required when use_raw_input is True")

        if self.use_embeddings and embeddings is None:
            raise ValueError("embeddings is required when use_embeddings is True")

        batch_size = raw_input.size(0) if raw_input is not None else embeddings.size(0)

        if self.latents is None:
            if video_emb is not None:
                latents = self.layer_norm2(video_emb)
            else:
                latents = self.latents_p.unsqueeze(0).repeat(batch_size, 1, 1)
        else:
            latents = self.latents

        latents = self.dropout1(latents)

        if self.use_raw_input:
            flattened_raw_input = raw_input.view(raw_input.size(0), -1, raw_input.size(-1))
            latents = self.raw_cross_attention(latents, flattened_raw_input) + latents
            latents = self.dropout2(latents)
        if self.use_embeddings:
            embeddings = self.layer_norm1(embeddings)
            latents = self.embedding_cross_attention(latents, embeddings) + latents
            latents = self.dropout3(latents)

        latents_res = latents
        latents = self.transformer(latents)
        latents = latents + latents_res

        if self.output_layer is not None:
            output = self.output_layer(latents.view(latents.size(0), -1))
        else:
            output = latents[:, 0, :]

        if is_reset_latents:
            self.reset_latents()
        else:
            self.latents = latents

        return output


class PerceiverV2(nn.Module):
    def __init__(
        self,
        raw_input_dim,
        embedding_dim,
        latent_dim,
        num_heads,
        num_latents,
        num_transformer_layers,
        dropout,
        output_dim,
        use_raw_input=True,
        use_embeddings=True,
        flatten_channels=False,
        use_video_emb=False,
    ):
        super().__init__()
        self.latents_p = nn.Parameter(
            nn.init.xavier_uniform_(torch.randn(num_latents, latent_dim)) * (latent_dim**0.5)
        )
        self.latents = None

        self.use_video_emb = use_video_emb
        pe_nlatents = num_latents + 1 if use_video_emb else num_latents
        self.positional_embeddings = nn.init.xavier_uniform_(
            nn.Parameter(torch.randn(pe_nlatents, latent_dim))
        )

        self.num_latents = num_latents
        self.raw_input_dim = raw_input_dim
        self.embedding_dim = embedding_dim
        self.use_raw_input = use_raw_input
        self.use_embeddings = use_embeddings
        assert (use_raw_input or use_embeddings) and not (use_raw_input and use_embeddings), (
            "At least one of use_raw_input or use_embeddings must be True and both cannot be True at once"
        )
        self.flatten_channels = flatten_channels

        if use_raw_input:
            self.raw_cross_attention = CrossAttention(latent_dim, raw_input_dim, num_heads)
        if use_embeddings:
            self.embedding_cross_attention = CrossAttention(latent_dim, embedding_dim, num_heads)

        self.transformer = TransformerEncoder(
            latent_dim, num_heads, num_transformer_layers, dropout
        )

        if output_dim is not None:
            self.output_layer = nn.Linear(latent_dim * num_latents, output_dim)
        else:
            self.output_layer = None

        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(latent_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def reset_latents(self):
        self.latents = None

    def forward(
        self,
        raw_input=None,
        embeddings=None,
        video_emb=None,
        add_pos_emb=True,
        is_reset_latents=False,
    ):
        _ = video_emb

        if not self.use_raw_input and not self.use_embeddings:
            raise ValueError("At least one of use_raw_input or use_embeddings must be True")

        if self.use_raw_input and raw_input is None:
            raise ValueError("raw_input is required when use_raw_input is True")

        if self.use_embeddings and embeddings is None:
            raise ValueError("embeddings is required when use_embeddings is True")

        batch_size = raw_input.size(0) if raw_input is not None else embeddings.size(0)

        if self.latents is None:
            latents = self.latents_p.unsqueeze(0).repeat(batch_size, 1, 1)
        else:
            latents = self.latents

        if add_pos_emb:
            latents = latents + self.positional_embeddings.unsqueeze(0).repeat(batch_size, 1, 1)

        latents = self.dropout1(latents)

        if self.use_raw_input:
            flattened_raw_input = raw_input.view(raw_input.size(0), -1, raw_input.size(-1))
            latents = self.raw_cross_attention(latents, flattened_raw_input) + latents
            latents = self.dropout2(latents)
        if self.use_embeddings:
            embeddings = self.layer_norm1(embeddings)
            latents = self.embedding_cross_attention(latents, embeddings) + latents
            latents = self.dropout3(latents)

        latents = self.transformer(latents)

        if self.output_layer is not None:
            output = self.output_layer(latents.view(latents.size(0), -1))
        else:
            output = latents[:, 0, :]

        if is_reset_latents:
            self.reset_latents()
        else:
            self.latents = latents

        return output
