from __future__ import annotations

import os

import torch

from diffsynth.pipelines.qwen_image import ModelConfig, QwenImagePipeline
from diffsynth.trainers.unified_dataset import UnifiedDataset
from diffsynth.trainers.utils import (
    DiffusionTrainingModule,
    ModelLogger,
    launch_data_process_task,
    launch_training_task,
    qwen_image_parser,
)

from seampainter.controlnet import (
    SeamPainterControlNetInput,
    install_seampainter_controlnet_unit,
)
from seampainter.loss import diffusion_training_loss
from seampainter.masks import pil_mask_to_tensor

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class SeamPainterTrainingModule(DiffusionTrainingModule):
    """Qwen-Image LoRA training with SeamPainter conditioning and supervision."""

    def __init__(
        self,
        model_paths=None,
        model_id_with_origin_paths=None,
        tokenizer_path=None,
        trainable_models=None,
        lora_base_model=None,
        lora_target_modules="",
        lora_rank=32,
        lora_checkpoint=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        enable_fp8_training=False,
        task="sft",
        seam_feature_scale=1.003,
        quality_feature_scale=1.006,
        quality_loss_weight=1.1,
        mask_threshold=0.1,
    ):
        super().__init__()
        model_configs = self.parse_model_configs(
            model_paths,
            model_id_with_origin_paths,
            enable_fp8_training=enable_fp8_training,
        )
        tokenizer_config = (
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/")
            if tokenizer_path is None
            else ModelConfig(tokenizer_path)
        )
        self.pipe = QwenImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cpu",
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            processor_config=None,
        )
        install_seampainter_controlnet_unit(
            self.pipe,
            seam_feature_scale=seam_feature_scale,
            quality_feature_scale=quality_feature_scale,
            mask_threshold=mask_threshold,
        )
        self.switch_pipe_to_training_mode(
            self.pipe,
            trainable_models,
            lora_base_model,
            lora_target_modules,
            lora_rank,
            lora_checkpoint=lora_checkpoint,
            enable_fp8_training=enable_fp8_training,
        )
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.task = task
        self.quality_loss_weight = float(quality_loss_weight)
        self.mask_threshold = float(mask_threshold)

    def forward_preprocess(self, data):
        gt_image = data["image"]
        condition_image = data["blockwise_controlnet_image"]
        seam_mask = data["blockwise_controlnet_inpaint_mask"]
        quality_mask = data["seam_quality_mask"]

        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {"negative_prompt": ""}
        inputs_shared = {
            # GT follows the unmodified DiffSynth input-image encoder and normal
            # diffusion noising path. No second GT encoder is used.
            "input_image": gt_image,
            "inpaint_mask": seam_mask,
            "height": gt_image.size[1],
            "width": gt_image.size[0],
            "cfg_scale": 1,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "edit_image_auto_resize": False,
            "blockwise_controlnet_inputs": [
                SeamPainterControlNetInput(
                    image=condition_image,
                    inpaint_mask=seam_mask,
                    seam_quality_mask=quality_mask,
                )
            ],
        }

        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(
                unit,
                self.pipe,
                inputs_shared,
                inputs_posi,
                inputs_nega,
            )

        latent = inputs_shared["input_latents"]
        inputs_shared["seam_quality_mask"] = pil_mask_to_tensor(
            quality_mask,
            latent.shape[-2:],
            device=latent.device,
            dtype=latent.dtype,
            threshold=self.mask_threshold,
        )
        return {**inputs_shared, **inputs_posi}

    def forward(self, data, inputs=None, return_inputs=False):
        if inputs is None:
            inputs = self.forward_preprocess(data)
        else:
            inputs = self.transfer_data_to_device(
                inputs,
                self.pipe.device,
                self.pipe.torch_dtype,
            )
        if return_inputs:
            return inputs
        if self.task == "sft":
            models = {
                name: getattr(self.pipe, name)
                for name in self.pipe.in_iteration_models
            }
            return diffusion_training_loss(
                self.pipe,
                inputs,
                models,
                quality_weight=self.quality_loss_weight,
                threshold=self.mask_threshold,
            )
        if self.task == "data_process":
            return inputs
        raise NotImplementedError(f"Unsupported task: {self.task}")


def parse_args():
    parser = qwen_image_parser()
    parser.set_defaults(
        data_file_keys=(
            "image,blockwise_controlnet_image,"
            "blockwise_controlnet_inpaint_mask,seam_quality_mask"
        )
    )
    parser.add_argument("--seam_feature_scale", type=float, default=1.05)
    parser.add_argument("--quality_feature_scale", type=float, default=1.10)
    parser.add_argument("--quality_loss_weight", type=float, default=2.0)
    parser.add_argument("--mask_threshold", type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_image_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
        ),
    )
    model = SeamPainterTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        enable_fp8_training=args.enable_fp8_training,
        task=args.task,
        seam_feature_scale=args.seam_feature_scale,
        quality_feature_scale=args.quality_feature_scale,
        quality_loss_weight=args.quality_loss_weight,
        mask_threshold=args.mask_threshold,
    )
    logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    launcher = {
        "sft": launch_training_task,
        "data_process": launch_data_process_task,
    }
    launcher[args.task](dataset, model, logger, args=args)


if __name__ == "__main__":
    main()
