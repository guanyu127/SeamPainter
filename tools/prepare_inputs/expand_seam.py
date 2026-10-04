from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def binary_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return (mask > 127).astype(np.uint8)


def expand_seam_array(
    raw_seam: np.ndarray,
    iterations: int = 20,
    threshold: float = 0.1,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Expand a one-pixel seam using the convolution rule from SeamPainter."""
    current = binary_mask(raw_seam).astype(np.float32)
    kernel = np.ones((3, 3), dtype=np.float32) / 9.0
    for _ in range(iterations):
        current = (cv2.filter2D(current, -1, kernel) > threshold).astype(np.float32)
    expanded = current.astype(np.uint8)
    if valid_mask is not None:
        expanded &= binary_mask(valid_mask)
    return expanded * 255


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand a raw seam mask.")
    parser.add_argument("--raw-seam", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--valid-mask", type=Path)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.1)
    args = parser.parse_args()

    raw = cv2.imread(str(args.raw_seam), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise FileNotFoundError(args.raw_seam)
    valid = None
    if args.valid_mask is not None:
        valid = cv2.imread(str(args.valid_mask), cv2.IMREAD_GRAYSCALE)
        if valid is None:
            raise FileNotFoundError(args.valid_mask)
    expanded = expand_seam_array(raw, args.iterations, args.threshold, valid)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), expanded)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
