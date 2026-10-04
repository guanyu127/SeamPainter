"""Run FLUX.2 SeamPainter on prepared example folders."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from PIL import Image, ImageOps
from safetensors.torch import load_file
from tqdm import tqdm

from seampainter_flux2.constants import (
    DEFAULT_LORA_TARGET_MODULES,
    DEFAULT_MASK_TOKEN_SCALE,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_PROMPT,
    DEFAULT_QUALITY_FEATURE_SCALE,
    DEFAULT_QUALITY_LOSS_WEIGHT,
    DEFAULT_SEAM_FEATURE_SCALE,
    DEFAULT_TOKENIZER_CONFIG,
)
from seampainter_flux2.masks import normalize_mask_pair
from seampainter_flux2.module import SeamPainterFlux2Module


def _find(folder: Path, stem: str) -> Path:
    matches = sorted(path for path in folder.glob(f"{stem}.*") if path.is_file())
    if not matches:
        raise FileNotFoundError(f"Missing {stem} image in {folder}")
    return matches[0]


def _open(path: Path, mode: str) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert(mode)


def _target_size(width: int, height: int, max_pixels: int) -> tuple[int, int]:
    scale = min(1.0, math.sqrt(max_pixels / float(width * height)))
    return (
        max(16, int(width * scale) // 16 * 16),
        max(16, int(height * scale) // 16 * 16),
    )


def _load_sample(folder: Path, max_pixels: int) -> dict:
    image = _open(_find(folder, "stitched_result"), "RGB")
    seam = _open(_find(folder, "seam_mask"), "L")
    quality = _open(_find(folder, "seam_quality_mask"), "L")
    if len({image.size, seam.size, quality.size}) != 1:
        raise ValueError(f"Images are not aligned in {folder}")
    size = _target_size(*image.size, max_pixels)
    if image.size != size:
        image = image.resize(size, Image.Resampling.BICUBIC)
        seam = seam.resize(size, Image.Resampling.NEAREST)
        quality = quality.resize(size, Image.Resampling.NEAREST)
    seam, quality = normalize_mask_pair(seam, quality, image.size)
    return {
        "input_image": image,
        "edit_image": image.copy(),
        "inpaint_mask": seam,
        "seam_quality_mask": quality,
        "prompt": DEFAULT_PROMPT,
        "sample_id": folder.name,
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("examples"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_flux2"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-paths", default=None)
    parser.add_argument("--model-id-with-origin-paths", default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--tokenizer-path", default=DEFAULT_TOKENIZER_CONFIG)
    parser.add_argument("--lora-target-modules", default=DEFAULT_LORA_TARGET_MODULES)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument(
        "--seam-feature-scale", type=float, default=DEFAULT_SEAM_FEATURE_SCALE
    )
    parser.add_argument(
        "--quality-feature-scale", type=float, default=DEFAULT_QUALITY_FEATURE_SCALE
    )
    parser.add_argument(
        "--quality-loss-weight", type=float, default=DEFAULT_QUALITY_LOSS_WEIGHT
    )
    parser.add_argument("--mask-token-scale", type=float, default=DEFAULT_MASK_TOKEN_SCALE)
    parser.add_argument("--max-pixels", type=int, default=1024 * 1024)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--embedded-guidance", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    return parser


def main():
    args = build_parser().parse_args()
    folders = sorted(path for path in args.input_dir.iterdir() if path.is_dir())
    if not folders:
        raise ValueError(f"No sample folders found in {args.input_dir}")
    device = torch.device(args.device)
    model = SeamPainterFlux2Module(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        seam_feature_scale=args.seam_feature_scale,
        quality_feature_scale=args.quality_feature_scale,
        quality_loss_weight=args.quality_loss_weight,
        mask_token_scale=args.mask_token_scale,
        use_gradient_checkpointing=False,
        device=device,
    )
    result = model.load_state_dict(load_file(args.checkpoint, device="cpu"), strict=False)
    missing = sorted(
        name for name in model.trainable_param_names() if name in result.missing_keys
    )
    if result.unexpected_keys or missing:
        raise RuntimeError(
            f"Invalid checkpoint: unexpected={result.unexpected_keys[:10]}, "
            f"missing={missing[:10]}"
        )
    model.to(device)
    model.pipe.device = device
    model.set_trainable_mode(False)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for index, folder in enumerate(tqdm(folders, desc="FLUX.2 SeamPainter")):
        sample = _load_sample(folder, args.max_pixels)
        repaired = model.generate(
            sample,
            seed=args.seed + index,
            num_inference_steps=args.num_inference_steps,
            embedded_guidance=args.embedded_guidance,
        )
        repaired.save(args.output_dir / f"{folder.name}.png")
    print(f"Saved {len(folders)} result(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
