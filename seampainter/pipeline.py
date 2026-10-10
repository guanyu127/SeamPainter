from __future__ import annotations

import torch

from diffsynth.pipelines.qwen_image import ModelConfig, QwenImagePipeline

from .constants import (
    DEFAULT_CONTROLNET_MODEL,
    DEFAULT_QUALITY_FEATURE_SCALE,
    DEFAULT_QWEN_MODEL,
    DEFAULT_SEAM_FEATURE_SCALE,
)
from .controlnet import install_seampainter_controlnet_unit


def model_configs(
    qwen_model: str = DEFAULT_QWEN_MODEL,
    controlnet_model: str = DEFAULT_CONTROLNET_MODEL,
) -> list[ModelConfig]:
    return [
        ModelConfig(
            model_id=qwen_model,
            origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors",
        ),
        ModelConfig(
            model_id=qwen_model,
            origin_file_pattern="text_encoder/model*.safetensors",
        ),
        ModelConfig(
            model_id=qwen_model,
            origin_file_pattern="vae/diffusion_pytorch_model.safetensors",
        ),
        ModelConfig(
            model_id=controlnet_model,
            origin_file_pattern="model.safetensors",
        ),
    ]


def create_pipeline(
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    qwen_model: str = DEFAULT_QWEN_MODEL,
    controlnet_model: str = DEFAULT_CONTROLNET_MODEL,
    lora_path: str | None = None,
    seam_feature_scale: float = DEFAULT_SEAM_FEATURE_SCALE,
    quality_feature_scale: float = DEFAULT_QUALITY_FEATURE_SCALE,
):
    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=dtype,
        device=device,
        model_configs=model_configs(qwen_model, controlnet_model),
        tokenizer_config=ModelConfig(
            model_id=qwen_model,
            origin_file_pattern="tokenizer/",
        ),
    )
    install_seampainter_controlnet_unit(
        pipe,
        seam_feature_scale=seam_feature_scale,
        quality_feature_scale=quality_feature_scale,
    )
    if lora_path is not None:
        pipe.load_lora(pipe.dit, lora_path)
    return pipe
