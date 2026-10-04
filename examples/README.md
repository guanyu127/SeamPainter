# Examples

Every sample directory is a ready SeamPainter input and contains:

```text
stitched_result.(jpg|png)
seam_mask.(jpg|png)
seam_quality_mask.(jpg|png)
```

The examples intentionally do not execute seam cutting, seam expansion, or seam-quality estimation. Those optional preparation steps live under `tools/prepare_inputs/`.

Run:

```bash
python inference.py \
  --input examples/sample_001 \
  --output outputs/sample_001.png \
  --lora-path /path/to/seampainter.safetensors
```
