#!/usr/bin/env bash
set -euo pipefail

LORA_PATH="/path/to/seampainter.safetensors"

python inference.py \
  --input-dir examples \
  --output-dir outputs \
  --lora-path "${LORA_PATH}" \
  --num-inference-steps 20 \
  --seed 123
