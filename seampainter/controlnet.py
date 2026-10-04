from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from diffsynth.pipelines.flux_image_new import ControlNetInput
from diffsynth.pipelines.qwen_image import QwenImageUnit_BlockwiseControlNet

from .constants import (
    DEFAULT_MASK_THRESHOLD,
    DEFAULT_QUALITY_FEATURE_SCALE,
    DEFAULT_SEAM_FEATURE_SCALE,
)
from .masks import build_feature_weight, pil_mask_to_tensor


@dataclass
class SeamPainterControlNetInput(ControlNetInput):
    """ControlNet input extended with the core seam-quality region."""

    seam_quality_mask: Image.Image | None = None


class SeamPainterBlockwiseControlNetUnit(QwenImageUnit_BlockwiseControlNet):
    """Blockwise ControlNet preprocessing used by SeamPainter.

    Compared with DiffSynth-Studio v1.1.9 this unit:
    1. preserves pixels inside ``inpaint_mask`` instead of painting them black;
    2. applies a three-level weight to the encoded conditioning latent;
    3. retains the official appended inpaint-mask channel.
    """

    def __init__(
        self,
        seam_feature_scale: float = DEFAULT_SEAM_FEATURE_SCALE,
        quality_feature_scale: float = DEFAULT_QUALITY_FEATURE_SCALE,
        mask_threshold: float = DEFAULT_MASK_THRESHOLD,
    ) -> None:
        super().__init__()
        if seam_feature_scale < 0 or quality_feature_scale < 0:
            raise ValueError("Feature scales must be non-negative.")
        self.seam_feature_scale = float(seam_feature_scale)
        self.quality_feature_scale = float(quality_feature_scale)
        self.mask_threshold = float(mask_threshold)

    def apply_controlnet_mask_on_image(self, pipe, image, mask):
        """Return the complete image; SeamPainter never clears masked pixels."""
        del pipe, mask
        return image

    def amplify_conditioning(
        self,
        latent,
        inpaint_mask: Image.Image,
        quality_mask: Image.Image | None,
    ):
        seam_tensor = pil_mask_to_tensor(
            inpaint_mask,
            latent.shape[-2:],
            device=latent.device,
            dtype=latent.dtype,
            threshold=self.mask_threshold,
        )
        if quality_mask is None:
            quality_tensor = seam_tensor.new_zeros(seam_tensor.shape)
        else:
            quality_tensor = pil_mask_to_tensor(
                quality_mask,
                latent.shape[-2:],
                device=latent.device,
                dtype=latent.dtype,
                threshold=self.mask_threshold,
            )
        weight, _, _ = build_feature_weight(
            seam_tensor,
            quality_tensor,
            latent,
            seam_scale=self.seam_feature_scale,
            quality_scale=self.quality_feature_scale,
            threshold=self.mask_threshold,
        )
        return latent * weight

    def process(self, pipe, blockwise_controlnet_inputs, tiled, tile_size, tile_stride):
        if blockwise_controlnet_inputs is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        conditionings = []
        for controlnet_input in blockwise_controlnet_inputs:
            # Deliberately do not call the official image-masking operation.
            image = controlnet_input.image
            image = pipe.preprocess_image(image).to(
                device=pipe.device,
                dtype=pipe.torch_dtype,
            )
            latent = pipe.vae.encode(
                image,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
            if controlnet_input.inpaint_mask is not None:
                latent = self.amplify_conditioning(
                    latent,
                    controlnet_input.inpaint_mask,
                    getattr(controlnet_input, "seam_quality_mask", None),
                )
                latent = self.apply_controlnet_mask_on_latents(
                    pipe,
                    latent,
                    controlnet_input.inpaint_mask,
                )
            conditionings.append(latent)
        return {"blockwise_controlnet_conditioning": conditionings}


def install_seampainter_controlnet_unit(
    pipe,
    seam_feature_scale: float = DEFAULT_SEAM_FEATURE_SCALE,
    quality_feature_scale: float = DEFAULT_QUALITY_FEATURE_SCALE,
    mask_threshold: float = DEFAULT_MASK_THRESHOLD,
):
    """Replace DiffSynth's official blockwise preprocessing unit in-place."""
    replacement = SeamPainterBlockwiseControlNetUnit(
        seam_feature_scale=seam_feature_scale,
        quality_feature_scale=quality_feature_scale,
        mask_threshold=mask_threshold,
    )
    replaced = 0
    for index, unit in enumerate(pipe.units):
        if isinstance(unit, QwenImageUnit_BlockwiseControlNet):
            pipe.units[index] = replacement
            replaced += 1
    if replaced != 1:
        raise RuntimeError(
            "Expected exactly one QwenImageUnit_BlockwiseControlNet in the "
            f"DiffSynth pipeline, found {replaced}. Check diffsynth==1.1.9."
        )
    return pipe
