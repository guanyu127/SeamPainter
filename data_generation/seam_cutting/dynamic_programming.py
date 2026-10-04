from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from tools.prepare_inputs.expand_seam import binary_mask
from .common import SeamCutResult, load_sample, result_from_partition, save_result, validate_inputs


def seam_energy(warp1: np.ndarray, warp2: np.ndarray) -> np.ndarray:
    gray1 = cv2.cvtColor(warp1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray2 = cv2.cvtColor(warp2, cv2.COLOR_BGR2GRAY).astype(np.float32)
    color = np.abs(gray1 - gray2)
    grad1_x = cv2.Sobel(gray1, cv2.CV_32F, 1, 0, ksize=3)
    grad1_y = cv2.Sobel(gray1, cv2.CV_32F, 0, 1, ksize=3)
    grad2_x = cv2.Sobel(gray2, cv2.CV_32F, 1, 0, ksize=3)
    grad2_y = cv2.Sobel(gray2, cv2.CV_32F, 0, 1, ksize=3)
    geometry = np.abs(grad1_x - grad2_x) + np.abs(grad1_y - grad2_y)
    return color + geometry


def _vertical_shortest_path(energy: np.ndarray, allowed: np.ndarray) -> list[tuple[int, int]]:
    height, width = energy.shape
    cost = np.full((height, width), np.inf, dtype=np.float64)
    parent = np.zeros((height, width), dtype=np.int8)
    cost[0, allowed[0]] = energy[0, allowed[0]]
    for y in range(1, height):
        for x in np.flatnonzero(allowed[y]):
            lo, hi = max(0, x - 1), min(width, x + 2)
            candidates = cost[y - 1, lo:hi]
            if np.isfinite(candidates).any():
                offset = int(np.argmin(candidates))
                previous_x = lo + offset
                cost[y, x] = energy[y, x] + cost[y - 1, previous_x]
                parent[y, x] = previous_x - x
    valid_end = np.flatnonzero(np.isfinite(cost[-1]))
    if valid_end.size == 0:
        raise RuntimeError("The overlap is disconnected; DP could not find a full seam.")
    x = int(valid_end[np.argmin(cost[-1, valid_end])])
    path = []
    for y in range(height - 1, -1, -1):
        path.append((y, x))
        if y:
            x += int(parent[y, x])
    return list(reversed(path))


def dynamic_programming_seam_cut(
    warp1: np.ndarray,
    warp2: np.ndarray,
    mask1: np.ndarray,
    mask2: np.ndarray,
    *,
    expansion_iterations: int = 20,
    expansion_threshold: float = 0.1,
) -> SeamCutResult:
    validate_inputs(warp1, warp2, mask1, mask2)
    overlap = binary_mask(mask1).astype(bool) & binary_mask(mask2).astype(bool)
    ys, xs = np.where(overlap)
    if ys.size == 0:
        raise ValueError("The two warped images do not overlap.")
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    roi_allowed = overlap[y0:y1, x0:x1]
    roi_energy = seam_energy(warp1, warp2)[y0:y1, x0:x1]
    vertical = roi_allowed.shape[0] >= roi_allowed.shape[1]
    if vertical:
        path = _vertical_shortest_path(roi_energy, roi_allowed)
    else:
        path = [(x, y) for y, x in _vertical_shortest_path(roi_energy.T, roi_allowed.T)]

    raw_seam = np.zeros(overlap.shape, dtype=np.uint8)
    take_first = np.zeros(overlap.shape, dtype=bool)
    if vertical:
        for local_y, local_x in path:
            y, x = local_y + y0, local_x + x0
            raw_seam[y, x] = 255
            take_first[y, : x + 1] = True
    else:
        for local_y, local_x in path:
            y, x = local_y + y0, local_x + x0
            raw_seam[y, x] = 255
            take_first[: y + 1, x] = True
    take_first &= overlap
    return result_from_partition(
        warp1,
        warp2,
        mask1,
        mask2,
        take_first,
        raw_seam,
        expansion_iterations=expansion_iterations,
        expansion_threshold=expansion_threshold,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch DP seam cutting for warped image pairs.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.1)
    args = parser.parse_args()
    sample_dirs = sorted(path for path in args.input_root.iterdir() if path.is_dir())
    for index, sample_dir in enumerate(sample_dirs, 1):
        try:
            result = dynamic_programming_seam_cut(
                *load_sample(sample_dir),
                expansion_iterations=args.iterations,
                expansion_threshold=args.threshold,
            )
            save_result(result, args.output_root, sample_dir.name)
            print(f"[{index}/{len(sample_dirs)}] {sample_dir.name}")
        except Exception as error:
            print(f"[{index}/{len(sample_dirs)}] failed {sample_dir.name}: {error}")


if __name__ == "__main__":
    main()
