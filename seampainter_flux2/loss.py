"""Hierarchical flow-matching supervision for FLUX.2 SeamPainter."""

from __future__ import annotations

import torch

from .masks import build_supervision_weight
from .model import model_fn_flux2_seampainter


class SeamPainterFlux2Loss(torch.nn.Module):
    def __init__(self, quality_loss_weight: float = 1.5):
        super().__init__()
        self.quality_loss_weight = float(quality_loss_weight)

    @staticmethod
    def _fixed_noise(reference: torch.Tensor, seed: int) -> torch.Tensor:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        noise = torch.randn(reference.shape, generator=generator, dtype=torch.float32)
        return noise.to(device=reference.device, dtype=reference.dtype)

    def forward(
        self,
        pipe,
        reference_latents: torch.Tensor,
        seampainter_seam_tokens: torch.Tensor,
        seampainter_quality_tokens: torch.Tensor,
        deterministic_seed: int | None = None,
        deterministic_timestep_id: int | None = None,
        return_metrics: bool = False,
        **inputs,
    ):
        target = inputs["input_latents"]
        if target is None:
            raise ValueError("Training requires the clean target image")
        seam = seampainter_seam_tokens.to(device=target.device, dtype=target.dtype)
        quality = seampainter_quality_tokens.to(device=target.device, dtype=target.dtype)
        if reference_latents.shape != target.shape or seam.shape[:2] != target.shape[:2]:
            raise ValueError(
                "Latent shapes disagree: "
                f"reference={tuple(reference_latents.shape)}, target={tuple(target.shape)}, "
                f"seam={tuple(seam.shape)}, quality={tuple(quality.shape)}"
            )

        timestep_count = len(pipe.scheduler.timesteps)
        timestep_id = (
            torch.randint(0, timestep_count, (1,)).item()
            if deterministic_timestep_id is None
            else int(deterministic_timestep_id) % timestep_count
        )
        timestep = pipe.scheduler.timesteps[timestep_id : timestep_id + 1].to(
            dtype=pipe.torch_dtype, device=pipe.device
        )
        noise = (
            torch.randn_like(target)
            if deterministic_seed is None
            else self._fixed_noise(target, deterministic_seed)
        )
        noisy_target = pipe.scheduler.add_noise(target, noise, timestep)
        inputs["latents"] = reference_latents * (1 - seam) + noisy_target * seam
        velocity_target = pipe.scheduler.training_target(target, noise, timestep)
        prediction = model_fn_flux2_seampainter(dit=pipe.dit, timestep=timestep, **inputs)
        squared = (prediction.float() - velocity_target.float()).pow(2)
        supervision = build_supervision_weight(
            seam.float(), quality.float(), self.quality_loss_weight
        )
        denominator = supervision.sum() * squared.shape[-1]
        if float(denominator.detach().cpu()) <= 0:
            raise ValueError("SeamPainter received an empty seam mask")
        loss = (squared * supervision).sum() / denominator
        loss = loss * pipe.scheduler.training_weight(timestep)

        if not return_metrics:
            return loss
        metrics = {
            "loss/total": loss.detach(),
            "loss/hierarchical_seam_flow": loss.detach(),
            "mask/seam_token_ratio": seam.float().mean().detach(),
            "mask/quality_token_ratio": quality.float().mean().detach(),
            "train/timestep_id": torch.tensor(float(timestep_id), device=target.device),
        }
        return loss, metrics


__all__ = ["SeamPainterFlux2Loss"]
