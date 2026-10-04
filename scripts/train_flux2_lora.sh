#!/usr/bin/env bash
set -euo pipefail

# Edit these two paths before training.
DATASET_ROOT="/path/to/seampainter_dataset"
OUTPUT_ROOT="checkpoints/seampainter-flux2-lora"
LORA_TARGETS="to_q,to_k,to_v,to_out.0,add_q_proj,add_k_proj,add_v_proj,to_add_out,linear_in,linear_out,to_qkv_mlp_proj,single_transformer_blocks.0.attn.to_out,single_transformer_blocks.1.attn.to_out,single_transformer_blocks.2.attn.to_out,single_transformer_blocks.3.attn.to_out,single_transformer_blocks.4.attn.to_out,single_transformer_blocks.5.attn.to_out,single_transformer_blocks.6.attn.to_out,single_transformer_blocks.7.attn.to_out,single_transformer_blocks.8.attn.to_out,single_transformer_blocks.9.attn.to_out,single_transformer_blocks.10.attn.to_out,single_transformer_blocks.11.attn.to_out,single_transformer_blocks.12.attn.to_out,single_transformer_blocks.13.attn.to_out,single_transformer_blocks.14.attn.to_out,single_transformer_blocks.15.attn.to_out,single_transformer_blocks.16.attn.to_out,single_transformer_blocks.17.attn.to_out,single_transformer_blocks.18.attn.to_out,single_transformer_blocks.19.attn.to_out"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

accelerate launch train_flux2.py \
  --dataset_base_path "${DATASET_ROOT}" \
  --dataset_metadata_path "${DATASET_ROOT}/metadata.csv" \
  --max_pixels 1048576 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "black-forest-labs/FLUX.2-klein-4B:text_encoder/*.safetensors,black-forest-labs/FLUX.2-klein-base-4B:transformer/*.safetensors,black-forest-labs/FLUX.2-klein-4B:vae/diffusion_pytorch_model.safetensors" \
  --tokenizer_path "black-forest-labs/FLUX.2-klein-4B:tokenizer/" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --save_steps 500 \
  --output_path "${OUTPUT_ROOT}" \
  --lora_base_model dit \
  --lora_target_modules "${LORA_TARGETS}" \
  --lora_rank 32 \
  --seam_feature_scale 1.05 \
  --quality_feature_scale 1.10 \
  --quality_loss_weight 1.5 \
  --mask_token_scale 1.0 \
  --use_gradient_checkpointing \
  --random_horizontal_flip \
  --find_unused_parameters
