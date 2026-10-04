from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from tools.prepare_inputs.estimate_seam_quality import estimate_quality_masks
from tools.prepare_inputs.expand_seam import binary_mask, expand_seam_array


def read(path: Path, flag: int):
    image = cv2.imread(str(path), flag)
    if image is None:
        raise FileNotFoundError(path)
    return image


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the three ready inputs consumed by SeamPainter inference."
    )
    parser.add_argument("--stitched-result", type=Path, required=True)
    parser.add_argument("--raw-seam", type=Path, required=True)
    parser.add_argument("--warp1", type=Path, required=True)
    parser.add_argument("--warp2", type=Path, required=True)
    parser.add_argument("--mask1", type=Path, required=True)
    parser.add_argument("--mask2", type=Path, required=True)
    parser.add_argument("--stitched-mask", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seam-iterations", type=int, default=20)
    parser.add_argument("--seam-threshold", type=float, default=0.1)
    parser.add_argument("--patch-size", type=int, default=21)
    parser.add_argument("--ssim-threshold", type=float, default=0.75)
    args = parser.parse_args()

    warp1 = read(args.warp1, cv2.IMREAD_COLOR)
    warp2 = read(args.warp2, cv2.IMREAD_COLOR)
    mask1 = read(args.mask1, cv2.IMREAD_GRAYSCALE)
    mask2 = read(args.mask2, cv2.IMREAD_GRAYSCALE)
    raw_seam = read(args.raw_seam, cv2.IMREAD_GRAYSCALE)
    stitched_result = read(args.stitched_result, cv2.IMREAD_COLOR)
    valid = (
        read(args.stitched_mask, cv2.IMREAD_GRAYSCALE)
        if args.stitched_mask is not None
        else ((binary_mask(mask1) | binary_mask(mask2)) * 255)
    )
    shapes = {
        image.shape[:2]
        for image in (warp1, warp2, mask1, mask2, raw_seam, stitched_result, valid)
    }
    if len(shapes) != 1:
        raise ValueError(f"All input images and masks must share a size, got {sorted(shapes)}")
    seam_mask = expand_seam_array(
        raw_seam,
        iterations=args.seam_iterations,
        threshold=args.seam_threshold,
        valid_mask=valid,
    )
    _, quality_mask = estimate_quality_masks(
        warp1,
        warp2,
        mask1,
        mask2,
        raw_seam,
        seam_mask,
        patch_size=args.patch_size,
        ssim_threshold=args.ssim_threshold,
        expansion_iterations=args.seam_iterations,
        expansion_threshold=args.seam_threshold,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output_dir / "stitched_result.png"), stitched_result)
    cv2.imwrite(str(args.output_dir / "seam_mask.png"), seam_mask)
    cv2.imwrite(str(args.output_dir / "seam_quality_mask.png"), quality_mask)
    print(f"Prepared SeamPainter input: {args.output_dir}")


if __name__ == "__main__":
    main()
