DEFAULT_PROMPT = (
    "As a high-precision image inpainting model, your core task is to repair "
    "the masked area (white mask region) in the input image: 1. Take the "
    "complete original texture of the input image itself as the exclusive "
    "texture reference for the masked area. 2. Correct all defects in the "
    "masked seam region: eliminate color jitter, fix structural misalignment, "
    "and ensure the restored content in the mask area is perfectly consistent "
    "with the background of the input image in color, lighting, and "
    "perspective. The transition between the restored area and the original "
    "background must be natural and seamless, with no visible stitching "
    "traces. 3. Strictly preserve all content in the non-masked area of the "
    "input image — ensure zero modification to the background outside the "
    "inpaint_mask region, keeping the original details and texture completely "
    "intact."
)

DEFAULT_QWEN_MODEL = "Qwen/Qwen-Image"
DEFAULT_CONTROLNET_MODEL = "DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Inpaint"

DEFAULT_SEAM_FEATURE_SCALE = 1.05
DEFAULT_QUALITY_FEATURE_SCALE = 1.10
DEFAULT_QUALITY_LOSS_WEIGHT = 1.5
