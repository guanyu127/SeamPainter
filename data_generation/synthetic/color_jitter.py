from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from tqdm import tqdm


def jitter_region(image: np.ndarray, mask: np.ndarray, rng: random.Random) -> np.ndarray:
    source = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    mask_image = Image.fromarray(mask)
    coords = np.argwhere(mask > 127)
    if coords.size == 0:
        return image.copy()
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    box = (int(x0), int(y0), int(x1), int(y1))
    region = source.crop(box)
    region = ImageEnhance.Brightness(region).enhance(rng.uniform(0.85, 1.15))
    region = ImageEnhance.Contrast(region).enhance(rng.uniform(0.85, 1.15))
    region = ImageEnhance.Color(region).enhance(rng.uniform(0.85, 1.15))
    source.paste(region, box, mask_image.crop(box))
    return cv2.cvtColor(np.asarray(source), cv2.COLOR_RGB2BGR)


def generate(root: Path, seed: int = 42) -> None:
    output = root / "color_jitter"
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    images = sorted((root / "natural_img").glob("*.*"))
    for image_path in tqdm(images, desc="Color jitter"):
        sample_id = image_path.stem
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(root / "split_mask" / sample_id / "left.png"), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            continue
        cv2.imwrite(str(output / f"{sample_id}.jpg"), jitter_region(image, mask, rng))


def main() -> None:
    parser = argparse.ArgumentParser(description="Add one-sided color inconsistency.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate(args.root, args.seed)


if __name__ == "__main__":
    main()
