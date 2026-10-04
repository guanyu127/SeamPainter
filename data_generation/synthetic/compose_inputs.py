from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def compose(color_image: np.ndarray, structure_image: np.ndarray, quality_mask: np.ndarray) -> np.ndarray:
    mask = (quality_mask > 127)[..., None]
    return np.where(mask, structure_image, color_image).astype(np.uint8)


def generate(root: Path) -> None:
    output = root / "input_img"
    output.mkdir(parents=True, exist_ok=True)
    images = sorted((root / "color_jitter").glob("*.*"))
    for color_path in tqdm(images, desc="Compose training inputs"):
        sample_id = color_path.stem
        color = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
        structure = cv2.imread(str(root / "structure_misalignment" / f"{sample_id}.jpg"), cv2.IMREAD_COLOR)
        quality = cv2.imread(str(root / "seam_quality_mask" / f"{sample_id}.png"), cv2.IMREAD_GRAYSCALE)
        if color is None or quality is None:
            continue
        if structure is None:
            structure = color
        cv2.imwrite(str(output / f"{sample_id}.jpg"), compose(color, structure, quality))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compose final corrupted ControlNet inputs.")
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    generate(args.root)


if __name__ == "__main__":
    main()
