from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from tools.prepare_inputs.expand_seam import binary_mask, expand_seam_array


def masked_patch_ssim(a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> float:
    """Reproduce the research script's masked 3x3 Gaussian SSIM.

    Only Gaussian windows whose pixels belong to the overlap are averaged. The
    implementation uses NumPy/OpenCV instead of requiring PyTorch during data
    preparation.
    """
    x = a.astype(np.float64)
    y = b.astype(np.float64)
    valid = valid.astype(bool)
    if x.shape != y.shape or x.shape != valid.shape:
        raise ValueError("SSIM patches and their validity mask must share a shape.")

    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    coordinates = np.arange(3, dtype=np.float64) - 1.0
    gaussian = np.exp(-0.5 * (coordinates / 1.5) ** 2)
    gaussian /= gaussian.sum()
    kernel = gaussian[:, None] * gaussian[None, :]

    # The original script replaced zero-valued masked pixels with NaN before
    # convolution, so zero intensity in either warp is excluded as well.
    valid = valid & (x != 0) & (y != 0)
    x = np.where(valid, x, 0.0)
    y = np.where(valid, y, 0.0)
    mean_x = cv2.filter2D(x, cv2.CV_64F, kernel, borderType=cv2.BORDER_CONSTANT)
    mean_y = cv2.filter2D(y, cv2.CV_64F, kernel, borderType=cv2.BORDER_CONSTANT)
    mean_x_sq = mean_x**2
    mean_y_sq = mean_y**2
    mean_xy = mean_x * mean_y
    var_x = cv2.filter2D(x**2, cv2.CV_64F, kernel, borderType=cv2.BORDER_CONSTANT) - mean_x_sq
    var_y = cv2.filter2D(y**2, cv2.CV_64F, kernel, borderType=cv2.BORDER_CONSTANT) - mean_y_sq
    covariance = cv2.filter2D(x * y, cv2.CV_64F, kernel, borderType=cv2.BORDER_CONSTANT) - mean_xy

    numerator = (2 * mean_xy + c1) * (2 * covariance + c2)
    denominator = (mean_x_sq + mean_y_sq + c1) * (var_x + var_y + c2)
    ssim_map = numerator / np.maximum(denominator, 1e-12)

    # A 3x3 Gaussian response is valid only when the complete support lies in
    # the overlap, matching the NaN-masked convolution in the research code.
    support = cv2.filter2D(
        valid.astype(np.uint8),
        cv2.CV_16U,
        np.ones((3, 3), dtype=np.uint8),
        borderType=cv2.BORDER_CONSTANT,
    )
    scores = ssim_map[support == 9]
    return float(scores.mean()) if scores.size else 0.0


def estimate_quality_masks(
    warp1: np.ndarray,
    warp2: np.ndarray,
    mask1: np.ndarray,
    mask2: np.ndarray,
    raw_seam: np.ndarray,
    expanded_seam: np.ndarray,
    *,
    patch_size: int = 21,
    ssim_threshold: float = 0.75,
    expansion_iterations: int = 20,
    expansion_threshold: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate Mqe from local cross-warp similarity along a cutting seam."""
    if patch_size < 3 or patch_size % 2 == 0:
        raise ValueError("patch_size must be an odd integer >= 3")
    shapes = {
        image.shape[:2]
        for image in (warp1, warp2, mask1, mask2, raw_seam, expanded_seam)
    }
    if len(shapes) != 1:
        raise ValueError(f"All warps and masks must share a size, got {sorted(shapes)}")
    gray1 = cv2.cvtColor(warp1, cv2.COLOR_BGR2GRAY) if warp1.ndim == 3 else warp1
    gray2 = cv2.cvtColor(warp2, cv2.COLOR_BGR2GRAY) if warp2.ndim == 3 else warp2
    overlap = binary_mask(mask1).astype(bool) & binary_mask(mask2).astype(bool)
    seam = binary_mask(raw_seam).astype(bool)
    radius = patch_size // 2
    height, width = seam.shape
    poor = np.zeros_like(seam, dtype=np.uint8)

    for y, x in np.argwhere(seam):
        if y < radius or x < radius or y >= height - radius or x >= width - radius:
            continue
        ys = slice(y - radius, y + radius + 1)
        xs = slice(x - radius, x + radius + 1)
        valid = overlap[ys, xs]
        score = masked_patch_ssim(gray1[ys, xs], gray2[ys, xs], valid)
        if score < ssim_threshold:
            poor[y, x] = 255

    quality = expand_seam_array(
        poor,
        iterations=expansion_iterations,
        threshold=expansion_threshold,
        valid_mask=expanded_seam,
    )
    return poor, quality


def _read(path: Path, flag: int):
    image = cv2.imread(str(path), flag)
    if image is None:
        raise FileNotFoundError(path)
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate expanded seam-quality mask (Mqe).")
    parser.add_argument("--warp1", type=Path, required=True)
    parser.add_argument("--warp2", type=Path, required=True)
    parser.add_argument("--mask1", type=Path, required=True)
    parser.add_argument("--mask2", type=Path, required=True)
    parser.add_argument("--raw-seam", type=Path, required=True)
    parser.add_argument("--seam-mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path)
    parser.add_argument("--patch-size", type=int, default=21)
    parser.add_argument("--ssim-threshold", type=float, default=0.75)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--expansion-threshold", type=float, default=0.1)
    args = parser.parse_args()

    poor, quality = estimate_quality_masks(
        _read(args.warp1, cv2.IMREAD_COLOR),
        _read(args.warp2, cv2.IMREAD_COLOR),
        _read(args.mask1, cv2.IMREAD_GRAYSCALE),
        _read(args.mask2, cv2.IMREAD_GRAYSCALE),
        _read(args.raw_seam, cv2.IMREAD_GRAYSCALE),
        _read(args.seam_mask, cv2.IMREAD_GRAYSCALE),
        patch_size=args.patch_size,
        ssim_threshold=args.ssim_threshold,
        expansion_iterations=args.iterations,
        expansion_threshold=args.expansion_threshold,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), quality)
    if args.raw_output is not None:
        args.raw_output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.raw_output), poor)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
