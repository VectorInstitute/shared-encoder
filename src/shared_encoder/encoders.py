"""Wrapper for CLIP model loaded via open_clip library."""

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.distributed
from mmlearn.conf import external_store
from open_clip.model import CLIP
from open_clip.transformer import _expand_token, text_global_pool
from torch import nn


@dataclass
class SharedEncoderConfig:
    """Configuration for shared encoder."""

    modality_embed_mode: str = "token_wise"
    modality_embed_dim: int = 20
    num_tokens: int = 1
    vision_modality_embed_dim: int = 768
    language_modality_embed_dim: int = 768


class MMCLIP(nn.Module):
    """Wrapper around the `CLIP` model loaded via open_clip."""

    def __init__(self) -> None:
        super().__init__()
        self.model_config = {
            "embed_dim": 512,
            "vision_cfg": {
                "image_size": 224,
                "patch_size": 16,
                "width": 768,
                "layers": 12,
            },
            "text_cfg": {
                "context_length": 77,
                "vocab_size": 49408,
                "width": 768,
                "heads": 12,
                "layers": 12,
            },
        }


@external_store(group="modules/encoders", provider="shared_encoder")
class MMCLIPText(MMCLIP):
    """Wrapper around the `CLIP` text model loaded via open_clip.

    Parameters
    ----------
    model_config_kwargs : Optional[dict[str, Any]], default=None
        Additional model configuration arguments.
    shared_encoder_config : Optional[SharedEncoderConfig], default=None
        Configuration for shared encoder.
    """

    def __init__(
        self,
        model_config_kwargs: Optional[dict[str, Any]] = None,
        shared_encoder_config: Optional[SharedEncoderConfig] = None,
    ) -> None:
        """Initialize the model."""
        super().__init__()
        self.model_config["text_cfg"] = model_config_kwargs
        clip_model = CLIP(
            **self.model_config,
        )
        del clip_model.visual
        self.model = clip_model
        # need a reference to the transformer for sharing layers
        self.transformer = self.model.transformer

        # modality tokens
        self.modality_embed_mode = (
            shared_encoder_config.modality_embed_mode
            if shared_encoder_config is not None
            else ""
        )
        if self.modality_embed_mode == "token_wise":
            modality_embed_dim = shared_encoder_config.modality_embed_dim
            self.language_token = torch.nn.Parameter(torch.randn(modality_embed_dim))

            # update text encoder
            self.model.token_embedding = nn.Embedding(
                self.model.token_embedding.weight.shape[0],
                self.model.token_embedding.weight.shape[1] - modality_embed_dim,
            )
        elif self.modality_embed_mode == "modality_token":
            language_modality_embed_dim = (
                shared_encoder_config.language_modality_embed_dim
            )
            num_tokens = shared_encoder_config.num_tokens
            self.language_token = torch.nn.Parameter(
                torch.randn(num_tokens, language_modality_embed_dim)
            )

            self.model.positional_embedding = nn.Parameter(
                torch.empty(
                    self.model.positional_embedding.shape[0] + num_tokens,
                    self.model.positional_embedding.shape[1],
                )
            )
            nn.init.normal_(self.model.positional_embedding, std=0.01)

            mask = torch.empty(
                self.model.positional_embedding.shape[0],
                self.model.positional_embedding.shape[0],
            )
            mask.fill_(float("-inf"))
            mask.triu_(1)  # zero out the lower diagonal
            self.model.register_buffer("attn_mask", mask, persistent=False)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the forward pass.

        Parameters
        ----------
        input_ids : torch.Tensor
            The input data.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The projected text embeddings and encoder output.
        """
        cast_dtype = self.model.transformer.get_cast_dtype()
        x: torch.Tensor = self.model.token_embedding(input_ids).to(cast_dtype)

        if self.modality_embed_mode == "token_wise":
            mod_tokens = self.language_token.repeat(x.shape[0], x.shape[1], 1).to(
                x.device
            )
            x = torch.cat([mod_tokens, x], dim=2)
        elif self.modality_embed_mode == "modality_token":
            mod_tokens = self.language_token.repeat(x.shape[0], 1, 1).to(x.device)
            x = torch.cat([x[:, 0:1, :], mod_tokens, x[:, 1:, :]], dim=1)

        x = x + self.model.positional_embedding.to(cast_dtype)
        x = self.model.transformer(x, attn_mask=self.model.attn_mask)
        encoder_output = x
        x = self.model.ln_final(x)  # [batch_size, n_ctx, transformer.width]

        if self.modality_embed_mode == "modality_token":
            # pad text with any token
            input_ids = torch.cat([input_ids[:, 0:1], input_ids], dim=1)

        x, _ = text_global_pool(x, input_ids, self.model.text_pool_type)
        if self.model.text_projection is not None:
            if isinstance(self.model.text_projection, nn.Linear):
                x = self.model.text_projection(x)
            else:
                x = x @ self.model.text_projection

        return (x, encoder_output)


@external_store(group="modules/encoders", provider="shared_encoder")
class MMCLIPVision(MMCLIP):
    """Wrapper around the `CLIP` vision model loaded via open_clip.

    Parameters
    ----------
    model_config_kwargs : Optional[dict[str, Any]], default=None
        Additional model configuration arguments.
    shared_encoder_config : Optional[SharedEncoderConfig]], default=None
        Configuration for shared encoder.
    """

    def __init__(
        self,
        model_config_kwargs: Optional[dict[str, Any]] = None,
        shared_encoder_config: Optional[SharedEncoderConfig] = None,
    ) -> None:
        """Initialize the model."""
        super().__init__()
        self.model_config["vision_cfg"] = model_config_kwargs
        clip_model = CLIP(
            **self.model_config,
        )
        del clip_model.transformer
        del clip_model.token_embedding
        self.model = clip_model
        # need a reference to the transformer for sharing layers
        self.transformer = self.model.visual.transformer

        # adding modality tokens
        self.modality_embed_mode = (
            shared_encoder_config.modality_embed_mode
            if shared_encoder_config is not None
            else ""
        )
        if self.modality_embed_mode == "token_wise":
            modality_embed_dim = shared_encoder_config.modality_embed_dim
            self.vision_token = torch.nn.Parameter(torch.randn(modality_embed_dim))

            # update vision encoder
            self.model.visual.conv1 = nn.Conv2d(
                in_channels=self.model.visual.conv1.in_channels,
                out_channels=self.model.visual.conv1.out_channels - modality_embed_dim,
                kernel_size=self.model.visual.conv1.kernel_size,
                stride=self.model.visual.conv1.stride,
                bias=self.model.visual.conv1.bias is not None,
            )

        elif self.modality_embed_mode == "modality_token":
            vision_modality_embed_dim = shared_encoder_config.vision_modality_embed_dim
            num_tokens = shared_encoder_config.num_tokens
            self.vision_token = torch.nn.Parameter(
                torch.randn(num_tokens, vision_modality_embed_dim)
            )

            scale = self.model.visual.positional_embedding.shape[1] ** -0.5
            self.model.visual.positional_embedding = nn.Parameter(
                scale
                * torch.randn(
                    self.model.visual.positional_embedding.shape[0] + num_tokens,
                    self.model.visual.positional_embedding.shape[1],
                )
            )

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the forward pass.

        Parameters
        ----------
        input_ids : torch.Tensor
            The input data.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The projected vision embeddings and encoder output.
        """
        x: torch.Tensor = self.model.visual.conv1(input_ids)

        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width] [-1, 196, 768]

        # class embeddings and positional embeddings
        if self.modality_embed_mode == "token_wise":
            mod_toekns = self.vision_token.repeat(x.shape[0], x.shape[1], 1).to(x.dtype)
            x = torch.cat([mod_toekns, x], dim=2)
        elif self.modality_embed_mode == "modality_token":
            mod_toekns = self.vision_token.repeat(x.shape[0], 1, 1).to(x.dtype)
            x = torch.cat([mod_toekns, x], dim=1)

        x = torch.cat(
            [
                _expand_token(self.model.visual.class_embedding, x.shape[0]).to(
                    x.dtype
                ),
                x,
            ],
            dim=1,
        )
        # shape = [*, grid ** 2 + 1, width]
        x = x + self.model.visual.positional_embedding.to(x.dtype)

        x = self.model.visual.patch_dropout(x)

        x = self.model.visual.ln_pre(x)
        x = self.model.visual.transformer(x)
        encoder_output = x

        if self.model.visual.attn_pool is not None:
            if self.model.visual.attn_pool_contrastive is not None:
                # This is untested, WIP pooling that should match paper
                x = self.model.visual.ln_post(
                    x
                )  # TBD LN first or separate one after each pool?
                tokens: torch.Tensor = self.model.visual.attn_pool(x)
                if self.model.visual.attn_pool_type == "parallel":
                    pooled: torch.Tensor = self.model.visual.attn_pool_contrastive(x)
                else:
                    assert self.model.visual.attn_pool_type == "cascade"
                    pooled = self.model.visual.attn_pool_contrastive(tokens)
            else:
                # this is the original OpenCLIP CoCa setup, does not match paper
                x = self.model.visual.attn_pool(x)
                x = self.model.visual.ln_post(x)
                pooled, tokens = self.model.visual._global_pool(x)
        elif self.model.visual.final_ln_after_pool:
            pooled, tokens = self.model.visual._global_pool(x)
            pooled = self.model.visual.ln_post(pooled)
        else:
            x = self.model.visual.ln_post(x)
            pooled, tokens = self.model.visual._global_pool(x)

        if self.model.visual.proj is not None:
            pooled = pooled @ self.model.visual.proj

        if self.model.visual.output_tokens:
            return pooled, tokens

        return (pooled, encoder_output)
