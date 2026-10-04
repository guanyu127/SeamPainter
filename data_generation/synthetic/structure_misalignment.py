from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def structural_distortion(
    image: np.ndarray,
    operation_mask: np.ndarray,
    raw_seam: np.ndarray,
    scale: float = 1.2,
) -> np.ndarray:
    result = image.copy()
    seam_coords = np.argwhere(raw_seam > 127)
    if seam_coords.size == 0:
        return result
    vertical = len(np.unique(seam_coords[:, 0])) >= len(np.unique(seam_coords[:, 1]))
    if vertical:
        for y in np.unique(seam_coords[:, 0]):
            xs = np.flatnonzero(operation_mask[y] > 127)
            if xs.size < 2:
                continue
            x0, x1 = xs[0], xs[-1] + 1
            row = image[y : y + 1, x0:x1]
            stretched = cv2.resize(row, (max(1, round(row.shape[1] * scale)), 1), interpolation=cv2.INTER_NEAREST)
            result[y : y + 1, x0:x1] = stretched[:, : x1 - x0]
    else:
        for x in np.unique(seam_coords[:, 1]):
            ys = np.flatnonzero(operation_mask[:, x] > 127)
            if ys.size < 2:
                continue
            y0, y1 = ys[0], ys[-1] + 1
            column = image[y0:y1, x : x + 1]
            stretched = cv2.resize(column, (1, max(1, round(column.shape[0] * scale))), interpolation=cv2.INTER_NEAREST)
            result[y0:y1, x : x + 1] = stretched[: y1 - y0]
    return result


def generate(root: Path, scale: float = 1.2) -> None:
    output = root / "structure_misalignment"
    output.mkdir(parents=True, exist_ok=True)
    images = sorted((root / "color_jitter").glob("*.*"))
    kernel = np.ones((3, 3), np.uint8)
    for image_path in tqdm(images, desc="Structure misalignment"):
        sample_id = image_path.stem
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        seam = cv2.imread(str(root / "inpaint_mask" / f"{sample_id}.png"), cv2.IMREAD_GRAYSCALE)
        raw = cv2.imread(str(root / "seam_mask_yuan" / f"{sample_id}.png"), cv2.IMREAD_GRAYSCALE)
        left = cv2.imread(str(root / "split_mask" / sample_id / "left.png"), cv2.IMREAD_GRAYSCALE)
        if any(value is None for value in (image, seam, raw, left)):
            continue
        operation = cv2.bitwise_and(seam, left)
        operation = cv2.erode(operation, kernel, iterations=2)
        result = structural_distortion(image, operation, raw, scale)
        cv2.imwrite(str(output / f"{sample_id}.jpg"), result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Add structural seam misalignment.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=1.2)
    args = parser.parse_args()
    generate(args.root, args.scale)


if __name__ == "__main__":
    main()
