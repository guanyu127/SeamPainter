from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def resize_to_multiple(image: np.ndarray, multiple: int = 16, max_pixels: int = 1024 * 1024):
    height, width = image.shape[:2]
    if height * width > max_pixels:
        scale = (max_pixels / (height * width)) ** 0.5
        width, height = max(multiple, int(width * scale)), max(multiple, int(height * scale))
    width = max(multiple, width // multiple * multiple)
    height = max(multiple, height // multiple * multiple)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def read_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    return ((mask > 127).astype(np.uint8) * 255)


def find_image(folder: Path, stem: str) -> Path:
    match = next((path for path in folder.glob(f"{stem}.*") if path.suffix.lower() in IMAGE_SUFFIXES), None)
    if match is None:
        raise FileNotFoundError(f"Cannot find {stem} in {folder}")
    return match


def prepare_dataset_base(
    natural_dir: Path,
    template_root: Path,
    output_root: Path,
    *,
    seed: int = 42,
    limit: int | None = None,
    max_pixels: int = 1024 * 1024,
) -> None:
    natural_files = sorted(path for path in natural_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    template_ids = sorted(path.stem for path in (template_root / "seam_mask").iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if limit is not None:
        natural_files = natural_files[:limit]
    if not natural_files or not template_ids:
        raise ValueError("Natural images and seam templates must both be non-empty.")
    random.Random(seed).shuffle(template_ids)

    for folder in ("natural_img", "inpaint_mask", "seam_mask_yuan", "stitched_mask", "split_mask"):
        (output_root / folder).mkdir(parents=True, exist_ok=True)

    for index, natural_path in enumerate(tqdm(natural_files, desc="Prepare natural images"), 1):
        template_id = template_ids[(index - 1) % len(template_ids)]
        image = cv2.imread(str(natural_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        image = resize_to_multiple(image, max_pixels=max_pixels)
        size = (image.shape[1], image.shape[0])
        stitched_mask = read_mask(find_image(template_root / "stitched_mask", template_id), size)
        seam_mask = read_mask(find_image(template_root / "seam_mask", template_id), size)
        raw_seam = read_mask(find_image(template_root / "seam_mask_yuan", template_id), size)
        split_dir = template_root / "split_mask" / template_id
        left = read_mask(find_image(split_dir, "left"), size)
        right = read_mask(find_image(split_dir, "right"), size)
        gt = (image.astype(np.float32) * (stitched_mask[..., None] / 255.0)).astype(np.uint8)
        sample_id = f"{index:05d}"
        cv2.imwrite(str(output_root / "natural_img" / f"{sample_id}.jpg"), gt)
        cv2.imwrite(str(output_root / "inpaint_mask" / f"{sample_id}.png"), seam_mask)
        cv2.imwrite(str(output_root / "seam_mask_yuan" / f"{sample_id}.png"), raw_seam)
        cv2.imwrite(str(output_root / "stitched_mask" / f"{sample_id}.png"), stitched_mask)
        sample_split = output_root / "split_mask" / sample_id
        sample_split.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(sample_split / "left.png"), left)
        cv2.imwrite(str(sample_split / "right.png"), right)


def main() -> None:
    parser = argparse.ArgumentParser(description="Map seam templates onto natural images.")
    parser.add_argument("--natural-dir", type=Path, required=True)
    parser.add_argument("--template-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-pixels", type=int, default=1024 * 1024)
    args = parser.parse_args()
    prepare_dataset_base(**vars(args))


if __name__ == "__main__":
    main()
