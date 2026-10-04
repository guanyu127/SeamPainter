# SeamPainter: Learning to Paint around Cutting Seam for Image Stitching
<p align="center">
  <a href="assets/first.pdf">
    <img src="assets/first.png" alt="SeamPainter framework" width="100%">
  </a>
</p>

SeamPainter takes a seam-cutting result, an expanded seam mask, and an expanded
seam-quality mask as input, then generates the refined stitching result.

## Installation

Python 3.10 or 3.11 is recommended. Install a CUDA-compatible PyTorch build
first, then install the remaining dependencies:

```bash
git clone <repository-url>
cd SeamPainter
pip install -r requirements.txt
pip install -e .
```

The code is tested against `diffsynth==1.1.9`. The Qwen-Image base model and
Blockwise ControlNet model are downloaded by DiffSynth on first use.

## Model

The SeamPainter model checkpoint is available at [Link](MODEL_LINK).

## Dataset

The SeamPainter training dataset is available at [Link](DATASET_LINK).

## Inference

Each example contains:

```text
sample_xxx/
├── stitched_result.jpg
├── seam_mask.jpg
└── seam_quality_mask.jpg
```

Set `LORA_PATH` in `scripts/run_examples.sh`, then run:

```bash
bash scripts/run_examples.sh
```

Results are saved in `outputs/`.

## Training

Set the dataset and output paths at the top of `scripts/train_lora.sh`, then run:

```bash
bash scripts/train_lora.sh
```

## Data preparation

### Training data

Synthetic training-data generation is provided in
[`data_generation/synthetic/`](data_generation/synthetic/). Seam-cutting code
for producing training templates is provided in
[`data_generation/seam_cutting/`](data_generation/seam_cutting/). See
[`data_generation/README.md`](data_generation/README.md) for details.

### Inference inputs

Code for expanding the seam mask and estimating the seam-quality mask is
provided in [`tools/prepare_inputs/`](tools/prepare_inputs/). The samples in
`examples/` are already prepared and can be used directly for inference.

## Acknowledgements

This project is built with
[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio), Qwen-Image,
and Qwen-Image Blockwise ControlNet Inpaint.
