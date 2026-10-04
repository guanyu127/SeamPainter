"""Reusable training and inference module for FLUX.2 SeamPainter."""

from __future__ import annotations

import numpy as np
import torch
from einops import rearrange
from PIL import Image

from diffsynth.core import ModelConfig
from diffsynth.diffusion.training_module import DiffusionTrainingModule
from diffsynth.pipelines.flux2_image import Flux2ImagePipeline

from .constants import (
    DEFAULT_MASK_TOKEN_SCALE,
    DEFAULT_QUALITY_FEATURE_SCALE,
    DEFAULT_QUALITY_LOSS_WEIGHT,
    DEFAULT_SEAM_FEATURE_SCALE,
)
from .loss import SeamPainterFlux2Loss
from .masks import mask_to_tokens, normalize_mask_pair
from .model import add_seampainter_conditions, model_fn_flux2_seampainter


def _pil_mask_to_tokens(
    mask: Image.Image,
    height: int,
    width: int,
    *,
    device,
    dtype,
) -> torch.Tensor:
    array = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array)[None, None].to(device=device)
    return mask_to_tokens(tensor, height, width).to(dtype=dtype)


class SeamPainterFlux2Module(DiffusionTrainingModule):
    """FLUX.2 LoRA with full seam input and hierarchical conditioning/loss."""

    def __init__(
        self,
        model_paths=None,
        model_id_with_origin_paths=None,
        tokenizer_path=None,
        lora_target_modules="",
        lora_rank=32,
        lora_checkpoint=None,
        seam_feature_scale=DEFAULT_SEAM_FEATURE_SCALE,
        quality_feature_scale=DEFAULT_QUALITY_FEATURE_SCALE,
        quality_loss_weight=DEFAULT_QUALITY_LOSS_WEIGHT,
        mask_token_scale=DEFAULT_MASK_TOKEN_SCALE,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        device="cpu",
    ):
        super().__init__()
        configs = self.parse_model_configs(
            model_paths, model_id_with_origin_paths, device=device
        )
        tokenizer = self.parse_path_or_model_id(
            tokenizer_path,
            default_value=ModelConfig(
                model_id="black-forest-labs/FLUX.2-klein-4B",
                origin_file_pattern="tokenizer/",
            ),
        )
        self.pipe = Flux2ImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=configs,
            tokenizer_config=tokenizer,
        )
        self.switch_pipe_to_training_mode(
            self.pipe,
            trainable_models=None,
            lora_base_model="dit",
            lora_target_modules=lora_target_modules,
            lora_rank=lora_rank,
            lora_checkpoint=lora_checkpoint,
            task="sft",
        )
        if seam_feature_scale < 0 or quality_feature_scale < 0:
            raise ValueError("Feature scales must be non-negative")
        if quality_loss_weight < 0:
            raise ValueError("Quality loss weight must be non-negative")
        self.seam_feature_scale = float(seam_feature_scale)
        self.quality_feature_scale = float(quality_feature_scale)
        self.mask_token_scale = float(mask_token_scale)
        self.loss_fn = SeamPainterFlux2Loss(quality_loss_weight)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self.use_gradient_checkpointing_offload = bool(
            use_gradient_checkpointing_offload
        )
        self.latest_metrics = {}

    def set_trainable_mode(self, training: bool) -> None:
        self.pipe.eval()
        self.pipe.dit.train(training)

    def trainable_modules(self):
        return filter(lambda parameter: parameter.requires_grad, self.parameters())

    def get_pipeline_inputs(self, data: dict):
        target = data.get("image")
        if target is None:
            raise ValueError("Training requires the clean target image")
        shared = {
            # The clean GT follows the official input-image encoder and noising
            # path; the stitched condition is encoded separately below.
            "input_image": target,
            "denoising_strength": 1.0,
            # Deliberately pass the complete stitched image. The seam region is
            # not blacked out or otherwise cleared.
            "edit_image": data["edit_image"],
            "edit_image_auto_resize": False,
            "warp1": None,
            "warp2": None,
            "coarse_image": None,
            "valid_mask1": None,
            "valid_mask2": None,
            "height": target.height,
            "width": target.width,
            "seed": None,
            "rand_device": self.pipe.device,
            "initial_noise": None,
            "embedded_guidance": 1.0,
            "cfg_scale": 1.0,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            # Training uses the max-pooled masks added after the unit runner.
            "inpaint_mask": None,
            "inpaint_blur_size": None,
            "inpaint_blur_sigma": None,
        }
        positive = {
            "prompt": data.get("prompt", ""),
            "kv_cache": None,
            "extra_text_embedding": None,
        }
        negative = {
            "negative_prompt": "",
            "kv_cache": None,
            "extra_text_embedding": None,
        }
        return shared, positive, negative

    def _encode_reference(self, image: Image.Image) -> torch.Tensor:
        self.pipe.load_models_to_device(["vae"])
        with torch.no_grad():
            latent = self.pipe.vae.encode(self.pipe.preprocess_image(image))
        return rearrange(latent, "B C H W -> B (H W) C")

    def _add_conditions(
        self,
        shared: dict,
        seam_mask: Image.Image,
        quality_mask: Image.Image,
    ) -> None:
        latent_h, latent_w = shared["height"] // 16, shared["width"] // 16
        seam_mask, quality_mask = normalize_mask_pair(
            seam_mask, quality_mask, (shared["width"], shared["height"])
        )
        seam_tokens = _pil_mask_to_tokens(
            seam_mask,
            latent_h,
            latent_w,
            device=self.pipe.device,
            dtype=self.pipe.torch_dtype,
        )
        if not bool(torch.any(seam_tokens).item()):
            raise ValueError("SeamPainter received an empty seam mask")
        quality_tokens = _pil_mask_to_tokens(
            quality_mask,
            latent_h,
            latent_w,
            device=self.pipe.device,
            dtype=self.pipe.torch_dtype,
        ) * seam_tokens
        add_seampainter_conditions(
            shared,
            seam_tokens,
            quality_tokens,
            seam_feature_scale=self.seam_feature_scale,
            quality_feature_scale=self.quality_feature_scale,
            mask_token_scale=self.mask_token_scale,
        )

    def _prepare_loss_inputs(
        self,
        data: dict,
        deterministic_seed=None,
        deterministic_timestep_id=None,
    ):
        inputs = self.get_pipeline_inputs(data)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        shared, positive, negative = inputs
        shared["reference_latents"] = self._encode_reference(data["input_image"])
        self._add_conditions(
            shared, data["inpaint_mask"], data["seam_quality_mask"]
        )
        shared["deterministic_seed"] = deterministic_seed
        shared["deterministic_timestep_id"] = deterministic_timestep_id
        return self.transfer_data_to_device(
            (shared, positive, negative), self.pipe.device, self.pipe.torch_dtype
        )

    def compute_loss(
        self,
        data: dict,
        deterministic_seed=None,
        deterministic_timestep_id=None,
        return_metrics=True,
    ):
        shared, positive, _ = self._prepare_loss_inputs(
            data, deterministic_seed, deterministic_timestep_id
        )
        return self.loss_fn(
            self.pipe, return_metrics=return_metrics, **shared, **positive
        )

    def forward(self, data: dict):
        loss, metrics = self.compute_loss(data, return_metrics=True)
        self.latest_metrics = metrics
        return loss

    @torch.no_grad()
    def generate(
        self,
        data: dict,
        *,
        seed: int,
        num_inference_steps: int,
        embedded_guidance: float = 1.0,
        composite_output: bool = True,
    ) -> Image.Image:
        input_image = data["input_image"].convert("RGB")
        seam_mask, quality_mask = normalize_mask_pair(
            data["inpaint_mask"], data["seam_quality_mask"], input_image.size
        )
        if input_image.width % 16 or input_image.height % 16:
            raise ValueError("Inference width and height must be divisible by 16")
        self.pipe.scheduler.set_timesteps(
            num_inference_steps,
            denoising_strength=1.0,
            dynamic_shift_len=input_image.height // 16 * input_image.width // 16,
        )
        shared = {
            "cfg_scale": 1.0,
            "embedded_guidance": embedded_guidance,
            "input_image": input_image,
            "denoising_strength": 1.0,
            # Keep the seam content visible to the FLUX.2 edit-image encoder.
            "edit_image": input_image,
            "edit_image_auto_resize": False,
            "warp1": None,
            "warp2": None,
            "coarse_image": None,
            "valid_mask1": None,
            "valid_mask2": None,
            "height": input_image.height,
            "width": input_image.width,
            "seed": seed,
            "rand_device": "cpu",
            "initial_noise": None,
            "inpaint_mask": seam_mask,
            "inpaint_blur_size": None,
            "inpaint_blur_sigma": None,
        }
        positive = {
            "prompt": data.get("prompt", ""),
            "kv_cache": None,
            "extra_text_embedding": None,
        }
        negative = {
            "negative_prompt": "",
            "kv_cache": None,
            "extra_text_embedding": None,
        }
        for unit in self.pipe.units:
            shared, positive, negative = self.pipe.unit_runner(
                unit, self.pipe, shared, positive, negative
            )
        reference_latents = shared["input_latents"]
        if reference_latents is None:
            raise RuntimeError("The input-image encoder produced no reference latents")
        self._add_conditions(shared, seam_mask, quality_mask)
        seam_tokens = shared["seampainter_seam_tokens"]
        shared["inpaint_mask"] = seam_tokens
        shared["latents"] = (
            reference_latents * (1 - seam_tokens) + shared["noise"] * seam_tokens
        )

        self.pipe.load_models_to_device(self.pipe.in_iteration_models)
        for progress_id, timestep in enumerate(self.pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(
                dtype=self.pipe.torch_dtype, device=self.pipe.device
            )
            prediction = model_fn_flux2_seampainter(
                dit=self.pipe.dit,
                timestep=timestep,
                **shared,
                **positive,
            )
            proposal = self.pipe.step(
                self.pipe.scheduler,
                latents=shared["latents"],
                progress_id=progress_id,
                noise_pred=prediction,
                input_latents=reference_latents,
                inpaint_mask=seam_tokens,
            )
            shared["latents"] = (
                reference_latents * (1 - seam_tokens) + proposal * seam_tokens
            )

        self.pipe.load_models_to_device(["vae"])
        latents = rearrange(
            shared["latents"],
            "B (H W) C -> B C H W",
            H=shared["height"] // 16,
            W=shared["width"] // 16,
        )
        repaired = self.pipe.vae_output_to_image(self.pipe.vae.decode(latents))
        self.pipe.load_models_to_device([])
        if composite_output:
            repaired = Image.composite(repaired.convert("RGB"), input_image, seam_mask)
        return repaired


__all__ = ["SeamPainterFlux2Module"]
