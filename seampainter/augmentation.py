"""Aligned image augmentations shared by SeamPainter training pipelines."""

from __future__ import annotations

import random
from collections.abc import Sequence

from PIL import Image


_QUARTER_TURNS = (
    Image.Transpose.ROTATE_90,
    Image.Transpose.ROTATE_180,
    Image.Transpose.ROTATE_270,
)


def validate_probability(probability: float) -> float:
    probability = float(probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("random_rotate_90_probability must be in [0, 1]")
    return probability


def maybe_rotate_aligned_90(
    images: Sequence[Image.Image | None],
    *,
    enabled: bool,
    probability: float,
) -> tuple[Image.Image | None, ...]:
    """Rotate aligned images together by a random non-zero quarter turn."""

    probability = validate_probability(probability)
    images = tuple(images)
    if not enabled or random.random() >= probability:
        return images
    method = random.choice(_QUARTER_TURNS)
    return tuple(image.transpose(method) if image is not None else None for image in images)


__all__ = ["maybe_rotate_aligned_90", "validate_probability"]
