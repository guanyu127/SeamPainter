from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _as_grayscale_array(mask: Image.Image) -> np.ndarray:
    return np.asarray(mask.convert("L"), dtype=np.uint8)


def normalize_mask_pair(
    seam_mask: Image.Image,
    quality_mask: Image.Image,
    size: Tuple[int, int] | None = None,
    threshold: int = 25,
) -> tuple[Image.Image, Image.Image, int]:
    """Binarize both masks and enforce quality_mask as a subset of seam_mask.

    Returns the normalized masks and the number of quality-mask pixels that
    were outside the seam mask before clipping.
    """
    if size is None:
        size = seam_mask.size
    seam = _as_grayscale_array(seam_mask.resize(size, Image.Resampling.NEAREST)) > threshold
    quality = _as_grayscale_array(quality_mask.resize(size, Image.Resampling.NEAREST)) > threshold
    outside = int(np.logical_and(quality, np.logical_not(seam)).sum())
    quality = np.logical_and(quality, seam)
    seam_image = Image.fromarray((seam.astype(np.uint8) * 255), mode="L")
    quality_image = Image.fromarray((quality.astype(np.uint8) * 255), mode="L")
    return seam_image, quality_image, outside


def pil_mask_to_tensor(
    mask: Image.Image,
    spatial_size: tuple[int, int],
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    threshold: float = 0.1,
) -> torch.Tensor:
    """Convert a PIL mask to a binary ``[1, 1, H, W]`` tensor."""
    height, width = spatial_size
    resized = mask.convert("L").resize((width, height), Image.Resampling.NEAREST)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).to(device=device, dtype=dtype)[None, None]
    return (tensor > threshold).to(dtype=dtype)


def tensor_mask_like(
    mask: torch.Tensor,
    reference: torch.Tensor,
    threshold: float = 0.1,
) -> torch.Tensor:
    """Resize, binarize and broadcast a tensor mask to a reference tensor."""
    if mask.ndim == 2:
        mask = mask[None, None]
    elif mask.ndim == 3:
        mask = mask[:, None]
    if mask.shape[-2:] != reference.shape[-2:]:
        mask = F.interpolate(mask.float(), size=reference.shape[-2:], mode="nearest")
    mask = (mask > threshold).to(device=reference.device, dtype=reference.dtype)
    if mask.shape[0] == 1 and reference.shape[0] != 1:
        mask = mask.expand(reference.shape[0], -1, -1, -1)
    if mask.shape[1] == 1 and reference.shape[1] != 1:
        mask = mask.expand(-1, reference.shape[1], -1, -1)
    return mask


def build_feature_weight(
    seam_mask: torch.Tensor,
    quality_mask: torch.Tensor,
    reference: torch.Tensor,
    seam_scale: float,
    quality_scale: float,
    threshold: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create the three-level feature weight map used by SeamPainter."""
    seam = tensor_mask_like(seam_mask, reference, threshold)
    quality = tensor_mask_like(quality_mask, reference, threshold) * seam
    weight = torch.ones_like(reference)
    weight = weight + (seam - quality) * (seam_scale - 1.0)
    weight = weight + quality * (quality_scale - 1.0)
    return weight, seam, quality


def blend_generated_region(
    input_image: Image.Image,
    generated_image: Image.Image,
    seam_mask: Image.Image,
) -> Image.Image:
    """Copy generated pixels only inside the expanded seam mask."""
    size = input_image.size
    generated = generated_image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    mask = seam_mask.convert("L").resize(size, Image.Resampling.NEAREST)
    mask_np = (np.asarray(mask, dtype=np.uint8) > 127)[..., None]
    input_np = np.asarray(input_image.convert("RGB"), dtype=np.uint8)
    generated_np = np.asarray(generated, dtype=np.uint8)
    result = np.where(mask_np, generated_np, input_np)
    return Image.fromarray(result.astype(np.uint8), mode="RGB")
