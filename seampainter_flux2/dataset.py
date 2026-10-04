"""Aligned training dataset for FLUX.2 SeamPainter."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from PIL import Image, ImageOps
from torch.utils.data import Dataset

from .constants import DEFAULT_PROMPT
from .masks import normalize_mask_pair


class SeamPainterFlux2Dataset(Dataset):
    """Load GT, stitched input, seam mask and seam-quality mask.

    The stitched image is passed to FLUX.2 unchanged. In particular, pixels in
    the white seam region are not cleared before edit-image VAE encoding.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        metadata_path: str | Path = "metadata.csv",
        *,
        repeat: int = 1,
        max_pixels: int = 1024 * 1024,
        height: int | None = None,
        width: int | None = None,
        random_horizontal_flip: bool = False,
        augment_seed: int = 20260904,
        require_target: bool = True,
        skip_empty_masks: bool = True,
    ):
        self.root = Path(dataset_root).resolve()
        metadata = Path(metadata_path)
        self.metadata_path = metadata if metadata.is_absolute() else self.root / metadata
        self.records = self._read_records(self.metadata_path)
        if not self.records:
            raise ValueError(f"Metadata contains no samples: {self.metadata_path}")
        if repeat <= 0:
            raise ValueError("repeat must be positive")
        if (height is None) != (width is None):
            raise ValueError("height and width must be specified together")
        if height is not None and (height % 16 or width % 16):
            raise ValueError("height and width must be divisible by 16")
        self.repeat = int(repeat)
        self.max_pixels = int(max_pixels)
        self.height = height
        self.width = width
        self.random_horizontal_flip = bool(random_horizontal_flip)
        self.augment_seed = int(augment_seed)
        self.require_target = bool(require_target)
        self.indices = list(range(len(self.records)))
        if skip_empty_masks:
            self.indices = [index for index in self.indices if self._mask_is_nonempty(index)]
            if not self.indices:
                raise ValueError(f"All seam masks are empty: {self.metadata_path}")

    @staticmethod
    def _read_records(path: Path) -> list[dict]:
        if path.suffix.lower() == ".json":
            records = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(records, list):
                raise ValueError(f"JSON metadata must be a list: {path}")
            return records
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    @staticmethod
    def _field(record: dict, *names: str, required: bool = True) -> str | None:
        for name in names:
            value = record.get(name)
            if value:
                return value
        if required:
            raise KeyError(f"Metadata record is missing one of: {', '.join(names)}")
        return None

    def _open(self, relative_path: str, mode: str) -> Image.Image:
        path = (self.root / relative_path).resolve()
        if path != self.root and self.root not in path.parents:
            raise ValueError(f"Dataset path escapes its root: {relative_path}")
        with Image.open(path) as image:
            return ImageOps.exif_transpose(image).convert(mode)

    def _mask_is_nonempty(self, index: int) -> bool:
        record = self.records[index]
        path = self._field(
            record,
            "inpaint_mask",
            "blockwise_controlnet_inpaint_mask",
        )
        return self._open(path, "L").getextrema()[1] >= 128

    def _target_size(self, width: int, height: int) -> tuple[int, int]:
        if self.width is not None:
            return self.width, self.height
        if self.max_pixels <= 0:
            raise ValueError("max_pixels must be positive")
        scale = min(1.0, math.sqrt(self.max_pixels / float(width * height)))
        return (
            max(16, int(width * scale) // 16 * 16),
            max(16, int(height * scale) // 16 * 16),
        )

    def __len__(self) -> int:
        return len(self.indices) * self.repeat

    def __getitem__(self, index: int) -> dict:
        record_index = self.indices[index % len(self.indices)]
        record = self.records[record_index]
        target_path = self._field(record, "image", required=self.require_target)
        input_path = self._field(record, "input_image", "blockwise_controlnet_image")
        edit_path = self._field(
            record,
            "edit_image",
            "input_image",
            "blockwise_controlnet_image",
        )
        seam_path = self._field(
            record,
            "inpaint_mask",
            "blockwise_controlnet_inpaint_mask",
        )
        quality_path = self._field(record, "seam_quality_mask")

        input_image = self._open(input_path, "RGB")
        edit_image = self._open(edit_path, "RGB")
        target = self._open(target_path, "RGB") if target_path else None
        seam_mask = self._open(seam_path, "L")
        quality_mask = self._open(quality_path, "L")
        sizes = {input_image.size, edit_image.size, seam_mask.size, quality_mask.size}
        if target is not None:
            sizes.add(target.size)
        if len(sizes) != 1:
            raise ValueError(f"Record {record_index} contains unaligned image sizes: {sizes}")

        target_size = self._target_size(*input_image.size)
        if input_image.size != target_size:
            input_image = input_image.resize(target_size, Image.Resampling.BICUBIC)
            edit_image = edit_image.resize(target_size, Image.Resampling.BICUBIC)
            seam_mask = seam_mask.resize(target_size, Image.Resampling.NEAREST)
            quality_mask = quality_mask.resize(target_size, Image.Resampling.NEAREST)
            if target is not None:
                target = target.resize(target_size, Image.Resampling.BICUBIC)
        seam_mask, quality_mask = normalize_mask_pair(
            seam_mask, quality_mask, input_image.size
        )

        do_flip = self.random_horizontal_flip and (
            (self.augment_seed + index * 1_000_003) % 2 == 1
        )
        if do_flip:
            input_image = ImageOps.mirror(input_image)
            edit_image = ImageOps.mirror(edit_image)
            seam_mask = ImageOps.mirror(seam_mask)
            quality_mask = ImageOps.mirror(quality_mask)
            if target is not None:
                target = ImageOps.mirror(target)

        sample_id = Path(input_path).stem or f"{record_index:05d}"
        return {
            "image": target,
            "input_image": input_image,
            # Keep a separate key for the official FLUX.2 edit-image interface,
            # but deliberately keep every input pixel in the masked region.
            "edit_image": edit_image,
            "inpaint_mask": seam_mask,
            "seam_quality_mask": quality_mask,
            "prompt": record.get("prompt") or DEFAULT_PROMPT,
            "sample_id": sample_id,
            "dataset_index": record_index,
        }


__all__ = ["SeamPainterFlux2Dataset"]
