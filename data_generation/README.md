# Data generation

This directory contains two independent workflows.

## 1. Seam-template generation

Input layout:

```text
warped_pairs/
└── 00001/
    ├── warped_image1.jpg
    ├── warped_image2.jpg
    ├── mask1.jpg
    └── mask2.jpg
```

Run either:

```bash
python -m data_generation.seam_cutting.dynamic_programming \
  --input-root warped_pairs \
  --output-root seam_templates
```

or:

```bash
python -m data_generation.seam_cutting.graph_cut \
  --input-root warped_pairs \
  --output-root seam_templates
```

Output layout:

```text
seam_templates/
├── stitched_result/
├── stitched_mask/
├── seam_mask_yuan/
├── seam_mask/
└── split_mask/<sample_id>/{left,right}.png
```

## 2. Synthetic training dataset

Run all stages:

```bash
python -m data_generation.synthetic.generate_dataset \
  --natural-dir natural_images \
  --template-root seam_templates \
  --output-root example_image_dataset
```

Intermediate directories are retained for inspection:

```text
color_jitter/
structure_misalignment/
seam_quality_mask/
```

Final training inputs are `natural_img/`, `input_img/`, `inpaint_mask/`, `seam_quality_mask/`, and `metadata.csv`.

## Real data

`traditional/prepare_inputs.py` exposes the same real-data preparation command as `tools/prepare_inputs/`. It creates the ready three-file inference input but is never invoked by the bundled examples.

The quality estimator reproduces the research implementation: it computes
masked local SSIM along the raw seam with a 3×3 Gaussian kernel, selects seam
points below the configured threshold, expands them with the same convolution
rule as the seam mask, and intersects the result with the expanded seam mask.
