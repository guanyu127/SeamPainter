"""Mask utilities shared by FLUX.2 training and inference."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def normalize_mask_pair(
    seam_mask: Image.Image,
    quality_mask: Image.Image,
    size: tuple[int, int] | None = None,
) -> tuple[Image.Image, Image.Image]:
    """Return binary masks and enforce quality_mask as a subset of seam_mask."""
    seam_mask = seam_mask.convert("L")
    quality_mask = quality_mask.convert("L")
    if size is not None:
        seam_mask = seam_mask.resize(size, Image.Resampling.NEAREST)
        quality_mask = quality_mask.resize(size, Image.Resampling.NEAREST)

    seam = np.asarray(seam_mask, dtype=np.uint8) >= 128
    quality = (np.asarray(quality_mask, dtype=np.uint8) >= 128) & seam
    return (
        Image.fromarray((seam.astype(np.uint8) * 255), mode="L"),
        Image.fromarray((quality.astype(np.uint8) * 255), mode="L"),
    )


def mask_to_tokens(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Convert a pixel-space binary mask to one scalar per FLUX.2 image token."""
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    mask = mask.to(dtype=torch.float32)
    mask = F.adaptive_max_pool2d(mask, (height, width))
    return (mask > 0.5).to(dtype=mask.dtype).flatten(2).transpose(1, 2)


def build_feature_weight(
    seam_tokens: torch.Tensor,
    quality_tokens: torch.Tensor,
    seam_scale: float,
    quality_scale: float,
) -> torch.Tensor:
    """Build outside/seam/low-quality hierarchical feature weights."""
    quality_tokens = quality_tokens * seam_tokens
    seam_only = (seam_tokens - quality_tokens).clamp_min(0.0)
    return 1.0 + seam_only * (seam_scale - 1.0) + quality_tokens * (quality_scale - 1.0)


def build_supervision_weight(
    seam_tokens: torch.Tensor,
    quality_tokens: torch.Tensor,
    quality_weight: float,
) -> torch.Tensor:
    """Build seam-only=1 and low-quality=quality_weight supervision weights."""
    quality_tokens = quality_tokens * seam_tokens
    seam_only = (seam_tokens - quality_tokens).clamp_min(0.0)
    return seam_only + quality_tokens * quality_weight
