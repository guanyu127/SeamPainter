from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def sample_quality_region(
    seam_mask: np.ndarray,
    rng: random.Random,
    np_rng: np.random.Generator,
    mean: float = 0.5,
    std: float = 0.2,
) -> np.ndarray:
    seam = seam_mask > 127
    height, _ = seam.shape
    result = np.zeros_like(seam, dtype=np.uint8)
    rows = np.flatnonzero(seam.any(axis=1))
    base = max(1, int(height * 0.1))
    low, high = max(1, int(base * 0.9)), max(1, int(base * 1.1))
    index = 0
    while index < len(rows):
        start = int(rows[index])
        length = rng.randint(low, high)
        end = min(height, start + length)
        score = float(np.clip(np_rng.normal(mean, std), 0.0, 1.0))
        if score < mean:
            result[start:end] = seam[start:end]
        index += max(1, np.searchsorted(rows[index:], end, side="left"))
    return result * 255


def generate(root: Path, seed: int = 42, mean: float = 0.5, std: float = 0.2) -> None:
    output = root / "seam_quality_mask"
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    masks = sorted((root / "inpaint_mask").glob("*.*"))
    for mask_path in tqdm(masks, desc="Quality regions"):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            cv2.imwrite(str(output / f"{mask_path.stem}.png"), sample_quality_region(mask, rng, np_rng, mean, std))


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample synthetic low-quality seam regions.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mean", type=float, default=0.5)
    parser.add_argument("--std", type=float, default=0.2)
    args = parser.parse_args()
    generate(args.root, args.seed, args.mean, args.std)


if __name__ == "__main__":
    main()
