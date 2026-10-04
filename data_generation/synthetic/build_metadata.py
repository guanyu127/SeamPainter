from __future__ import annotations

import argparse
import csv
from pathlib import Path

from seampainter.constants import DEFAULT_PROMPT


def build_metadata(root: Path, output: Path | None = None, prompt: str = DEFAULT_PROMPT) -> Path:
    output = output or root / "metadata.csv"
    folders = {
        "image": root / "natural_img",
        "blockwise_controlnet_image": root / "input_img",
        "blockwise_controlnet_inpaint_mask": root / "inpaint_mask",
        "seam_quality_mask": root / "seam_quality_mask",
    }
    ids = sorted(path.stem for path in folders["image"].glob("*.*"))
    rows = []
    for sample_id in ids:
        paths = {}
        for key, folder in folders.items():
            match = next(folder.glob(f"{sample_id}.*"), None)
            if match is None:
                break
            paths[key] = match.relative_to(root).as_posix()
        if len(paths) == len(folders):
            rows.append({**paths, "prompt": prompt})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*folders, "prompt"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} samples to {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DiffSynth training metadata.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args()
    build_metadata(args.root, args.output, args.prompt)


if __name__ == "__main__":
    main()
