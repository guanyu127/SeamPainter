#!/usr/bin/env bash
set -euo pipefail

# Set this to the FLUX.2 SeamPainter LoRA checkpoint.
LORA_PATH="/path/to/seampainter-flux2.safetensors"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

python inference_flux2.py \
  --input-dir examples \
  --output-dir outputs_flux2 \
  --checkpoint "${LORA_PATH}" \
  --seam-feature-scale 1.05 \
  --quality-feature-scale 1.10 \
  --quality-loss-weight 1.5 \
  --num-inference-steps 30 \
  --seed 123
