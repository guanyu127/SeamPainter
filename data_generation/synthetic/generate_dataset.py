from __future__ import annotations

import argparse
from pathlib import Path

from .build_metadata import build_metadata
from .color_jitter import generate as generate_color
from .compose_inputs import generate as compose_inputs
from .prepare_natural_images import prepare_dataset_base
from .quality_regions import generate as generate_quality
from .structure_misalignment import generate as generate_structure


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete synthetic data pipeline.")
    parser.add_argument("--natural-dir", type=Path, required=True)
    parser.add_argument("--template-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-pixels", type=int, default=1024 * 1024)
    args = parser.parse_args()

    prepare_dataset_base(
        args.natural_dir,
        args.template_root,
        args.output_root,
        seed=args.seed,
        limit=args.limit,
        max_pixels=args.max_pixels,
    )
    generate_color(args.output_root, args.seed)
    generate_structure(args.output_root)
    generate_quality(args.output_root, args.seed)
    compose_inputs(args.output_root)
    build_metadata(args.output_root)


if __name__ == "__main__":
    main()
