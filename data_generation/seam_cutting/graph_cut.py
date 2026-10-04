from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import maxflow
import numpy as np

from tools.prepare_inputs.expand_seam import binary_mask
from .common import SeamCutResult, load_sample, result_from_partition, save_result, validate_inputs


def graph_cut_seam(
    warp1: np.ndarray,
    warp2: np.ndarray,
    mask1: np.ndarray,
    mask2: np.ndarray,
    *,
    expansion_iterations: int = 20,
    expansion_threshold: float = 0.1,
) -> SeamCutResult:
    """Cut the overlap with a binary s-t graph and return stitching products."""
    validate_inputs(warp1, warp2, mask1, mask2)
    valid1 = binary_mask(mask1).astype(bool)
    valid2 = binary_mask(mask2).astype(bool)
    overlap = valid1 & valid2
    ys, xs = np.where(overlap)
    if ys.size == 0:
        raise ValueError("The two warped images do not overlap.")
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    roi_overlap = overlap[y0:y1, x0:x1]
    height, width = roi_overlap.shape

    gray1 = cv2.cvtColor(warp1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray2 = cv2.cvtColor(warp2, cv2.COLOR_BGR2GRAY).astype(np.float32)
    difference = np.abs(gray1 - gray2)
    gradient1 = cv2.magnitude(
        cv2.Sobel(gray1, cv2.CV_32F, 1, 0),
        cv2.Sobel(gray1, cv2.CV_32F, 0, 1),
    )
    gradient2 = cv2.magnitude(
        cv2.Sobel(gray2, cv2.CV_32F, 1, 0),
        cv2.Sobel(gray2, cv2.CV_32F, 0, 1),
    )
    energy = difference + np.abs(gradient1 - gradient2) + 1.0
    roi_energy = energy[y0:y1, x0:x1]

    graph = maxflow.GraphFloat()
    nodes = graph.add_grid_nodes((height, width))
    horizontal = roi_energy.copy()
    horizontal[:, :-1] += roi_energy[:, 1:]
    vertical = roi_energy.copy()
    vertical[:-1, :] += roi_energy[1:, :]
    graph.add_grid_edges(
        nodes,
        weights=horizontal,
        structure=np.array([[0, 0, 1]], dtype=np.int32),
        symmetric=True,
    )
    graph.add_grid_edges(
        nodes,
        weights=vertical,
        structure=np.array([[0], [0], [1]], dtype=np.int32),
        symmetric=True,
    )

    exclusive1 = valid1 & ~valid2
    exclusive2 = valid2 & ~valid1
    kernel = np.ones((3, 3), dtype=np.uint8)
    source_boundary = cv2.dilate(exclusive1.astype(np.uint8), kernel).astype(bool) & overlap
    sink_boundary = cv2.dilate(exclusive2.astype(np.uint8), kernel).astype(bool) & overlap
    source_roi = source_boundary[y0:y1, x0:x1]
    sink_roi = sink_boundary[y0:y1, x0:x1]
    # Identical or fully nested validity masks have no exclusive boundary.
    # Anchor opposite sides of the overlap so the s-t cut is still defined.
    if not source_roi.any() or not sink_roi.any():
        fallback_source = np.zeros_like(roi_overlap)
        fallback_sink = np.zeros_like(roi_overlap)
        if height >= width:
            for row in range(height):
                columns = np.flatnonzero(roi_overlap[row])
                if columns.size:
                    fallback_source[row, columns[0]] = True
                    fallback_sink[row, columns[-1]] = True
        else:
            for column in range(width):
                rows = np.flatnonzero(roi_overlap[:, column])
                if rows.size:
                    fallback_source[rows[0], column] = True
                    fallback_sink[rows[-1], column] = True
        if not source_roi.any():
            source_roi = fallback_source
        if not sink_roi.any():
            sink_roi = fallback_sink
    if np.logical_and(source_roi, sink_roi).any():
        raise ValueError("The overlap is too thin to place distinct GraphCut terminals.")
    hard = float(roi_energy.sum() + 1.0)
    graph.add_grid_tedges(nodes[source_roi], hard, 0.0)
    graph.add_grid_tedges(nodes[sink_roi], 0.0, hard)
    graph.maxflow()
    segments = graph.get_grid_segments(nodes)
    take_first_roi = np.logical_not(segments) & roi_overlap

    take_first = np.zeros_like(overlap)
    take_first[y0:y1, x0:x1] = take_first_roi
    first_dilated = cv2.dilate(take_first.astype(np.uint8), np.ones((3, 3), np.uint8))
    second = overlap & ~take_first
    raw_seam = ((first_dilated.astype(bool) & second) * 255).astype(np.uint8)
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
    parser = argparse.ArgumentParser(description="Batch GraphCut seam cutting for warped image pairs.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.1)
    args = parser.parse_args()
    sample_dirs = sorted(path for path in args.input_root.iterdir() if path.is_dir())
    for index, sample_dir in enumerate(sample_dirs, 1):
        try:
            result = graph_cut_seam(
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
