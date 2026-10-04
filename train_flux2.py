"""Train the FLUX.2 SeamPainter LoRA."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import accelerate
import numpy as np
import torch
from PIL import Image, ImageDraw
from safetensors.torch import load_file
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from diffsynth.core import OffloadTrainingManager
from diffsynth.diffusion import add_general_config, add_image_size_config
from diffsynth.diffusion.runner import (
    get_optimizer_class,
    initialize_deepspeed_gradient_checkpointing,
)

from seampainter_flux2.constants import (
    DEFAULT_MASK_TOKEN_SCALE,
    DEFAULT_QUALITY_FEATURE_SCALE,
    DEFAULT_QUALITY_LOSS_WEIGHT,
    DEFAULT_SEAM_FEATURE_SCALE,
)
from seampainter_flux2.dataset import SeamPainterFlux2Dataset
from seampainter_flux2.module import SeamPainterFlux2Module

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _single(batch):
    if len(batch) != 1:
        raise ValueError("FLUX.2 SeamPainter currently expects batch_size=1")
    return batch[0]


def _tensorboard_image(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def _overview(sample: dict, repaired: Image.Image) -> Image.Image:
    panels = [
        ("stitched input", sample["input_image"]),
        ("seam mask", sample["inpaint_mask"].convert("RGB")),
        ("seam-quality mask", sample["seam_quality_mask"].convert("RGB")),
        ("FLUX.2 SeamPainter", repaired),
        ("clean GT", sample["image"]),
    ]
    tile_w, tile_h, title_h, columns = 420, 300, 24, 2
    rows = (len(panels) + columns - 1) // columns
    canvas = Image.new("RGB", (tile_w * columns, (tile_h + title_h) * rows), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (title, image) in enumerate(panels):
        x = index % columns * tile_w
        y = index // columns * (tile_h + title_h)
        draw.text((x + 4, y + 4), title, fill="black")
        canvas.paste(
            image.convert("RGB").resize((tile_w, tile_h), Image.Resampling.BILINEAR),
            (x, y + title_h),
        )
    return canvas


@torch.no_grad()
def _fixed_loss(model, dataset, count, seed):
    values = []
    for index in range(min(count, len(dataset))):
        loss = model.compute_loss(
            dataset[index],
            deterministic_seed=seed + index,
            deterministic_timestep_id=(seed * 17 + index * 97) % 1000,
            return_metrics=False,
        )
        values.append(float(loss.detach().float().cpu()))
    return float(np.mean(values))


def _save(accelerator, model, output_dir: Path, step: int):
    accelerator.wait_for_everyone()
    state = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        state = accelerator.unwrap_model(model).export_trainable_state_dict(state)
        output_dir.mkdir(parents=True, exist_ok=True)
        accelerator.save(
            state,
            output_dir / f"step-{step:06d}.safetensors",
            safe_serialization=True,
        )


def _evaluate(accelerator, model, dataset, writer, step, args):
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        accelerator.wait_for_everyone()
        return
    module = accelerator.unwrap_model(model)
    module.set_trainable_mode(False)
    module.pipe.scheduler.set_timesteps(1000, training=True)
    writer.add_scalar(
        "eval/hierarchical_seam_flow",
        _fixed_loss(module, dataset, args.eval_loss_samples, args.eval_seed),
        step,
    )
    sample = dataset[args.validation_index]
    repaired = module.generate(
        sample,
        seed=args.validation_seed,
        num_inference_steps=args.validation_num_inference_steps,
        embedded_guidance=args.validation_embedded_guidance,
    )
    overview = _overview(sample, repaired)
    image_dir = Path(args.output_path) / "validation_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    overview.save(image_dir / f"step-{step:06d}.png")
    writer.add_image("inference/fixed_sample", _tensorboard_image(overview), step)
    writer.flush()
    module.pipe.scheduler.set_timesteps(1000, training=True)
    module.set_trainable_mode(True)
    accelerator.wait_for_everyone()


def build_parser():
    parser = add_image_size_config(
        add_general_config(argparse.ArgumentParser(description=__doc__))
    )
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument(
        "--seam_feature_scale", type=float, default=DEFAULT_SEAM_FEATURE_SCALE
    )
    parser.add_argument(
        "--quality_feature_scale", type=float, default=DEFAULT_QUALITY_FEATURE_SCALE
    )
    parser.add_argument(
        "--quality_loss_weight", type=float, default=DEFAULT_QUALITY_LOSS_WEIGHT
    )
    parser.add_argument(
        "--mask_token_scale", type=float, default=DEFAULT_MASK_TOKEN_SCALE
    )
    parser.add_argument("--random_horizontal_flip", action="store_true")
    parser.add_argument("--augment_seed", type=int, default=20260904)
    parser.add_argument("--validation_steps", type=int, default=200)
    parser.add_argument("--eval_loss_samples", type=int, default=8)
    parser.add_argument("--eval_seed", type=int, default=1234)
    parser.add_argument("--validation_index", type=int, default=0)
    parser.add_argument("--validation_seed", type=int, default=42)
    parser.add_argument("--validation_num_inference_steps", type=int, default=30)
    parser.add_argument("--validation_embedded_guidance", type=float, default=1.0)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0)
    return parser


def main():
    args = build_parser().parse_args()
    if args.task != "sft":
        raise ValueError("FLUX.2 SeamPainter supports only --task sft")
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[
            accelerate.DistributedDataParallelKwargs(
                find_unused_parameters=args.find_unused_parameters
            )
        ],
    )
    metadata = args.dataset_metadata_path or "metadata.csv"
    dataset_args = {
        "max_pixels": args.max_pixels,
        "height": args.height,
        "width": args.width,
    }
    train_dataset = SeamPainterFlux2Dataset(
        args.dataset_base_path,
        metadata,
        repeat=args.dataset_repeat,
        random_horizontal_flip=args.random_horizontal_flip,
        augment_seed=args.augment_seed,
        **dataset_args,
    )
    if args.max_train_samples is not None:
        if args.max_train_samples <= 0:
            raise ValueError("--max_train_samples must be positive")
        train_dataset = torch.utils.data.Subset(
            train_dataset, range(min(args.max_train_samples, len(train_dataset)))
        )
    eval_dataset = SeamPainterFlux2Dataset(
        args.dataset_base_path, metadata, repeat=1, **dataset_args
    )
    if not 0 <= args.validation_index < len(eval_dataset):
        raise ValueError(
            f"--validation_index must be in [0,{len(eval_dataset) - 1}]"
        )

    model = SeamPainterFlux2Module(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        seam_feature_scale=args.seam_feature_scale,
        quality_feature_scale=args.quality_feature_scale,
        quality_loss_weight=args.quality_loss_weight,
        mask_token_scale=args.mask_token_scale,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        device="cpu" if args.enable_model_cpu_offload else accelerator.device,
    )
    if args.resume_from_checkpoint:
        result = model.load_state_dict(
            load_file(args.resume_from_checkpoint, device="cpu"), strict=False
        )
        missing = sorted(
            name for name in model.trainable_param_names() if name in result.missing_keys
        )
        if result.unexpected_keys or missing:
            raise RuntimeError(
                f"Invalid checkpoint: unexpected={result.unexpected_keys[:10]}, "
                f"missing={missing[:10]}"
            )

    optimizer = get_optimizer_class(args.customized_optimizer)(
        model.trainable_modules(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=_single,
        num_workers=args.dataset_num_workers,
    )
    if args.enable_model_cpu_offload:
        optimizer, loader, scheduler = accelerator.prepare(optimizer, loader, scheduler)
        model.pipe.device = accelerator.device
        offload = OffloadTrainingManager(
            model,
            accelerator.device,
            args.enable_optimizer_cpu_offload,
            args.cpu_offload_split_threshold,
        )
    else:
        model.to(accelerator.device)
        model, optimizer, loader, scheduler = accelerator.prepare(
            model, optimizer, loader, scheduler
        )
        offload = None

    initialize_deepspeed_gradient_checkpointing(accelerator)
    module = accelerator.unwrap_model(model)
    module.pipe.scheduler.set_timesteps(1000, training=True)
    module.set_trainable_mode(True)
    output_dir = Path(args.output_path)
    writer = SummaryWriter(output_dir / "tensorboard_log") if accelerator.is_main_process else None
    global_step = 0
    should_stop = False
    for epoch in range(args.num_epochs):
        for data in tqdm(
            loader,
            desc=f"Epoch {epoch + 1}/{args.num_epochs}",
            disable=not accelerator.is_local_main_process,
        ):
            with accelerator.accumulate(model):
                loss = model(data)
                accelerator.backward(loss)
                if accelerator.sync_gradients and args.gradient_clip_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
                if offload is not None:
                    offload.after_backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            if not accelerator.sync_gradients:
                continue
            global_step += 1
            if accelerator.is_main_process:
                writer.add_scalar("train/loss_step", float(loss.detach().cpu()), global_step)
                for key, value in accelerator.unwrap_model(model).latest_metrics.items():
                    writer.add_scalar(key, float(value.detach().cpu()), global_step)
            if args.validation_steps > 0 and global_step % args.validation_steps == 0:
                _evaluate(accelerator, model, eval_dataset, writer, global_step, args)
            if args.save_steps and global_step % args.save_steps == 0:
                _save(accelerator, model, output_dir, global_step)
            if args.max_train_steps is not None and global_step >= args.max_train_steps:
                should_stop = True
                break
        if should_stop:
            break
    _save(accelerator, model, output_dir, global_step)
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
