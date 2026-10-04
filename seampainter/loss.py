from __future__ import annotations

import torch

from .masks import tensor_mask_like


def hierarchical_loss_weight(
    inpaint_mask: torch.Tensor,
    quality_mask: torch.Tensor,
    reference: torch.Tensor,
    quality_weight: float = 1.1,
    threshold: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build zero/base/quality supervision weights."""
    seam = tensor_mask_like(inpaint_mask, reference, threshold)
    quality = tensor_mask_like(quality_mask, reference, threshold) * seam
    weight = seam + quality * (quality_weight - 1.0)
    return weight, seam, quality


def hierarchical_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    inpaint_mask: torch.Tensor,
    quality_mask: torch.Tensor,
    quality_weight: float = 1.1,
    threshold: float = 0.1,
) -> torch.Tensor:
    """MSE supervised only in the expanded seam, with extra quality weight."""
    weight, _, _ = hierarchical_loss_weight(
        inpaint_mask,
        quality_mask,
        prediction,
        quality_weight=quality_weight,
        threshold=threshold,
    )
    squared_error = (prediction.float() - target.float()).pow(2)
    denominator = weight.float().sum().clamp_min(1.0)
    return (squared_error * weight.float()).sum() / denominator


def diffusion_training_loss(
    pipe,
    inputs: dict,
    models: dict,
    *,
    quality_weight: float = 1.1,
    threshold: float = 0.1,
) -> torch.Tensor:
    """Normal GT noising followed by SeamPainter hierarchical supervision."""
    timestep_id = torch.randint(0, pipe.scheduler.num_train_timesteps, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(
        dtype=pipe.torch_dtype,
        device=pipe.device,
    )
    clean_latents = inputs["input_latents"]
    noise = torch.randn_like(clean_latents)
    inputs["latents"] = pipe.scheduler.add_noise(clean_latents, noise, timestep)
    target = pipe.scheduler.training_target(clean_latents, noise, timestep)
    prediction = pipe.model_fn(**models, **inputs, timestep=timestep)
    loss = hierarchical_mse_loss(
        prediction,
        target,
        inputs["inpaint_mask"],
        inputs["seam_quality_mask"],
        quality_weight=quality_weight,
        threshold=threshold,
    )
    return loss * pipe.scheduler.training_weight(timestep)
