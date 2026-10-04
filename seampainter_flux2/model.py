"""FLUX.2 conditioning helpers for SeamPainter."""

from __future__ import annotations

import torch

from diffsynth.pipelines.flux2_image import model_fn_flux2


def build_mask_condition(
    seam_tokens: torch.Tensor,
    image_ids: torch.Tensor,
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode the seam mask as edit-image tokens with a distinct time id."""
    if seam_tokens.ndim != 3 or seam_tokens.shape[-1] != 1:
        raise ValueError(f"Expected seam tokens shaped [B, N, 1], got {tuple(seam_tokens.shape)}")
    mask_latents = (seam_tokens * 2.0 - 1.0).repeat(1, 1, 128) * scale
    mask_ids = image_ids.clone()
    mask_ids[..., 0] = 30
    return mask_latents, mask_ids


def add_seampainter_conditions(
    inputs: dict,
    seam_tokens: torch.Tensor,
    quality_tokens: torch.Tensor,
    seam_feature_scale: float,
    quality_feature_scale: float,
    mask_token_scale: float,
) -> dict:
    """Enhance edit features hierarchically and append seam-mask tokens."""
    from .masks import build_feature_weight

    edit_latents = inputs.get("edit_latents")
    edit_image_ids = inputs.get("edit_image_ids")
    if edit_latents is None or edit_image_ids is None:
        raise RuntimeError("FLUX.2 edit-image conditioning was not prepared by the pipeline.")
    if edit_latents.shape[1] != seam_tokens.shape[1]:
        raise RuntimeError(
            "SeamPainter expects exactly one edit image whose token count matches the target image. "
            f"Got edit={edit_latents.shape[1]} and mask={seam_tokens.shape[1]}."
        )

    feature_weight = build_feature_weight(
        seam_tokens,
        quality_tokens,
        seam_scale=seam_feature_scale,
        quality_scale=quality_feature_scale,
    ).to(device=edit_latents.device, dtype=edit_latents.dtype)
    enhanced_edit_latents = edit_latents * feature_weight
    mask_latents, mask_ids = build_mask_condition(
        seam_tokens.to(device=edit_latents.device, dtype=edit_latents.dtype),
        edit_image_ids,
        scale=mask_token_scale,
    )

    inputs["edit_latents"] = torch.cat([enhanced_edit_latents, mask_latents], dim=1)
    inputs["edit_image_ids"] = torch.cat([edit_image_ids, mask_ids], dim=1)
    inputs["seampainter_seam_tokens"] = seam_tokens
    inputs["seampainter_quality_tokens"] = quality_tokens
    return inputs


def model_fn_flux2_seampainter(*args, **kwargs):
    """Call the official FLUX.2 model function after removing private metadata."""
    kwargs.pop("seampainter_seam_tokens", None)
    kwargs.pop("seampainter_quality_tokens", None)
    kwargs.pop("reference_latents", None)
    return model_fn_flux2(*args, **kwargs)
