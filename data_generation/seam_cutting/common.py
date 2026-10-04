from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from tools.prepare_inputs.expand_seam import binary_mask, expand_seam_array


@dataclass
class SeamCutResult:
    stitched_result: np.ndarray
    raw_seam: np.ndarray
    seam_mask: np.ndarray
    left_mask: np.ndarray
    right_mask: np.ndarray
    stitched_mask: np.ndarray


def validate_inputs(warp1, warp2, mask1, mask2) -> None:
    shapes = {tuple(image.shape[:2]) for image in (warp1, warp2, mask1, mask2)}
    if len(shapes) != 1:
        raise ValueError(f"All warps and masks must share a size, got {sorted(shapes)}")


def result_from_partition(
    warp1: np.ndarray,
    warp2: np.ndarray,
    mask1: np.ndarray,
    mask2: np.ndarray,
    take_first: np.ndarray,
    raw_seam: np.ndarray,
    *,
    expansion_iterations: int = 20,
    expansion_threshold: float = 0.1,
) -> SeamCutResult:
    valid1 = binary_mask(mask1).astype(bool)
    valid2 = binary_mask(mask2).astype(bool)
    union = valid1 | valid2
    overlap = valid1 & valid2
    take_first = take_first.astype(bool)

    left = (valid1 & (~overlap | take_first)).astype(np.uint8)
    right = (valid2 & (~overlap | ~take_first)).astype(np.uint8)
    right[left.astype(bool)] = 0
    missing = union & ~(left.astype(bool) | right.astype(bool))
    left[missing] = 1

    stitched = (
        warp1.astype(np.float32) * left[..., None]
        + warp2.astype(np.float32) * right[..., None]
    ).clip(0, 255).astype(np.uint8)
    seam_mask = expand_seam_array(
        raw_seam,
        iterations=expansion_iterations,
        threshold=expansion_threshold,
        valid_mask=union.astype(np.uint8) * 255,
    )
    return SeamCutResult(
        stitched_result=stitched,
        raw_seam=raw_seam.astype(np.uint8),
        seam_mask=seam_mask,
        left_mask=left * 255,
        right_mask=right * 255,
        stitched_mask=union.astype(np.uint8) * 255,
    )


def save_result(result: SeamCutResult, output_root: Path, sample_id: str) -> None:
    paths = {
        "stitched_result": output_root / "stitched_result" / f"{sample_id}.jpg",
        "seam_mask_yuan": output_root / "seam_mask_yuan" / f"{sample_id}.png",
        "seam_mask": output_root / "seam_mask" / f"{sample_id}.png",
        "stitched_mask": output_root / "stitched_mask" / f"{sample_id}.png",
        "left": output_root / "split_mask" / sample_id / "left.png",
        "right": output_root / "split_mask" / sample_id / "right.png",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(paths["stitched_result"]), result.stitched_result)
    cv2.imwrite(str(paths["seam_mask_yuan"]), result.raw_seam)
    cv2.imwrite(str(paths["seam_mask"]), result.seam_mask)
    cv2.imwrite(str(paths["stitched_mask"]), result.stitched_mask)
    cv2.imwrite(str(paths["left"]), result.left_mask)
    cv2.imwrite(str(paths["right"]), result.right_mask)


def load_sample(sample_dir: Path):
    candidates = {
        "warp1": ("warped_image1.jpg", "warped_image_1.jpg"),
        "warp2": ("warped_image2.jpg", "warped_image_2.jpg"),
        "mask1": ("mask1.jpg", "mask_image_1.jpg"),
        "mask2": ("mask2.jpg", "mask_image_2.jpg"),
    }
    loaded = {}
    for key, names in candidates.items():
        path = next((sample_dir / name for name in names if (sample_dir / name).exists()), None)
        if path is None:
            raise FileNotFoundError(f"Missing {key} in {sample_dir}")
        flag = cv2.IMREAD_COLOR if key.startswith("warp") else cv2.IMREAD_GRAYSCALE
        loaded[key] = cv2.imread(str(path), flag)
        if loaded[key] is None:
            raise ValueError(f"Cannot read {path}")
    validate_inputs(loaded["warp1"], loaded["warp2"], loaded["mask1"], loaded["mask2"])
    return loaded["warp1"], loaded["warp2"], loaded["mask1"], loaded["mask2"]
