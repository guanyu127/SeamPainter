# SeamPainter: Learning to Paint around Cutting Seam for Image Stitching
<p align="center">
  <a href="assets/first.pdf">
    <img src="assets/first.png" alt="SeamPainter" width="100%">
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

## Model checkpoint

Download the SeamPainter LoRA checkpoint from the link in
[MODEL_ZOO.md](MODEL_ZOO.md). The checkpoint should be a DiffSynth-compatible
`.safetensors` file.

## Inference

Each input sample is a directory containing three prepared images:

```text
sample_001/
├── stitched_result.jpg
├── seam_mask.jpg
└── seam_quality_mask.jpg
```

- `stitched_result`: the output produced by a seam-cutting method.
- `seam_mask`: the expanded seam repair region.
- `seam_quality_mask`: the expanded low-quality seam region.

The bundled example already contains these three inputs and does not run seam
cutting or mask preparation.

Run one sample:

```bash
python inference.py \
  --input examples/sample_001 \
  --output outputs/sample_001.png \
  --lora-path /path/to/seampainter.safetensors
```

Run every sample directory under `examples/`:

```bash
python inference.py \
  --input-dir examples \
  --output-dir outputs \
  --lora-path /path/to/seampainter.safetensors
```

Use `--save-debug-grid` if an additional visualization of the input, masks,
generated image, and final result is required.

## Preparing real stitching inputs

Code for expanding a cutting seam and estimating the seam-quality mask is in
[`tools/prepare_inputs/`](tools/prepare_inputs/). Given two warped images,
their masks, the raw cutting seam, and the stitched image, run:

```bash
python -m tools.prepare_inputs.prepare_seampainter_input \
  --warp1 path/to/warped_image1.jpg \
  --warp2 path/to/warped_image2.jpg \
  --mask1 path/to/mask1.jpg \
  --mask2 path/to/mask2.jpg \
  --raw-seam path/to/raw_seam.png \
  --stitched-result path/to/stitched_result.jpg \
  --output-dir prepared/sample_001
```

The output directory can be passed directly to `inference.py`.

## Synthetic training-data generation

Dataset-generation code is in [`data_generation/`](data_generation/). Detailed
directory formats and individual commands are documented in
[`data_generation/README.md`](data_generation/README.md).

Generate seam-cutting templates with either dynamic programming or GraphCut:

```bash
python -m data_generation.seam_cutting.dynamic_programming \
  --input-root data/warped_pairs \
  --output-root data/seam_templates
```

```bash
python -m data_generation.seam_cutting.graph_cut \
  --input-root data/warped_pairs \
  --output-root data/seam_templates
```

Generate the complete synthetic training dataset:

```bash
python -m data_generation.synthetic.generate_dataset \
  --natural-dir data/natural_images \
  --template-root data/seam_templates \
  --output-root data/example_image_dataset \
  --seed 42
```

## Training

The generated dataset contains `natural_img/`, `input_img/`, `inpaint_mask/`,
`seam_quality_mask/`, and `metadata.csv`. Start LoRA training with:

```bash
bash scripts/train_lora.sh \
  data/example_image_dataset \
  checkpoints/seampainter-lora
```

All training arguments can also be inspected with:

```bash
python train.py --help
```

## Acknowledgements

This project is built with
[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio), Qwen-Image,
and Qwen-Image Blockwise ControlNet Inpaint.

## License and citation

The code is released under the Apache License 2.0. See
[`CITATION.cff`](CITATION.cff) for citation information.
