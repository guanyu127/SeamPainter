from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from seampainter.constants import DEFAULT_PROMPT
from seampainter.controlnet import SeamPainterControlNetInput
from seampainter.masks import blend_generated_region, normalize_mask_pair
from seampainter.pipeline import create_pipeline


def compatible_size(width: int, height: int) -> tuple[int, int]:
    return ((width + 15) // 16 * 16, (height + 15) // 16 * 16)


def find_file(sample_dir: Path, stem: str) -> Path:
    for suffix in (".png", ".jpg", ".jpeg", ".webp"):
        path = sample_dir / f"{stem}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing {stem} image in {sample_dir}")


def load_sample(sample_dir: Path):
    input_image = Image.open(find_file(sample_dir, "stitched_result")).convert("RGB")
    seam_mask = Image.open(find_file(sample_dir, "seam_mask")).convert("L")
    quality_mask = Image.open(find_file(sample_dir, "seam_quality_mask")).convert("L")
    seam_mask, quality_mask, outside = normalize_mask_pair(
        seam_mask,
        quality_mask,
        size=input_image.size,
    )
    if outside:
        print(
            f"Warning: clipped {outside} seam-quality pixels outside the seam mask "
            f"for {sample_dir.name}."
        )
    return input_image, seam_mask, quality_mask


def save_debug_grid(images: list[Image.Image], output_path: Path) -> None:
    height = max(image.height for image in images)
    resized = []
    for index, image in enumerate(images):
        width = round(image.width * height / image.height)
        interpolation = Image.Resampling.NEAREST if index in (1, 2) else Image.Resampling.LANCZOS
        resized.append(image.convert("RGB").resize((width, height), interpolation))
    canvas = Image.new("RGB", (sum(image.width for image in resized), height))
    offset = 0
    for image in resized:
        canvas.paste(image, (offset, 0))
        offset += image.width
    canvas.save(output_path)


@torch.no_grad()
def run_sample(pipe, sample_dir: Path, output_path: Path, args) -> Path:
    input_image, seam_mask, quality_mask = load_sample(sample_dir)
    original_size = input_image.size
    model_size = compatible_size(*original_size)

    model_input = input_image.resize(model_size, Image.Resampling.LANCZOS)
    model_seam = seam_mask.resize(model_size, Image.Resampling.NEAREST).convert("RGB")
    model_quality = quality_mask.resize(model_size, Image.Resampling.NEAREST).convert("RGB")

    generated = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
        input_image=model_input,
        inpaint_mask=model_seam,
        denoising_strength=args.denoising_strength,
        blockwise_controlnet_inputs=[
            SeamPainterControlNetInput(
                image=model_input,
                inpaint_mask=model_seam,
                seam_quality_mask=model_quality,
            )
        ],
        height=model_size[1],
        width=model_size[0],
        num_inference_steps=args.num_inference_steps,
    ).convert("RGB")

    final = blend_generated_region(model_input, generated, model_seam)
    final = final.resize(original_size, Image.Resampling.LANCZOS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final.save(output_path)

    if args.save_debug_grid:
        debug_path = output_path.with_name(f"{output_path.stem}_debug.jpg")
        save_debug_grid(
            [
                model_input,
                model_seam,
                model_quality,
                generated,
                final.resize(model_size, Image.Resampling.LANCZOS),
            ],
            debug_path,
        )
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SeamPainter on ready seam-cutting results and two expanded masks."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="One sample directory.")
    source.add_argument("--input-dir", type=Path, help="Directory containing sample subdirectories.")
    parser.add_argument("--output", type=Path, help="Output path for one sample.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--lora-path", type=str, required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--denoising-strength", type=float, default=1.0)
    parser.add_argument("--seam-feature-scale", type=float, default=1.003)
    parser.add_argument("--quality-feature-scale", type=float, default=1.006)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-debug-grid", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input is not None and args.output is None:
        args.output = args.output_dir / f"{args.input.name}.png"

    pipe = create_pipeline(
        device=args.device,
        lora_path=args.lora_path,
        seam_feature_scale=args.seam_feature_scale,
        quality_feature_scale=args.quality_feature_scale,
    )

    if args.input is not None:
        result = run_sample(pipe, args.input, args.output, args)
        print(f"Saved: {result}")
        return

    sample_dirs = sorted(path for path in args.input_dir.iterdir() if path.is_dir())
    if not sample_dirs:
        raise FileNotFoundError(f"No sample directories found in {args.input_dir}")
    for sample_dir in sample_dirs:
        result = run_sample(pipe, sample_dir, args.output_dir / f"{sample_dir.name}.png", args)
        print(f"Saved: {result}")


if __name__ == "__main__":
    main()
