"""Shared defaults for the FLUX.2 SeamPainter implementation."""

DEFAULT_PROMPT = (
    "A high-quality stitched panoramic image with a natural, seamless transition "
    "inside the masked seam region. Preserve the original content, geometry, "
    "illumination, color, and texture outside the seam."
)

DEFAULT_SEAM_FEATURE_SCALE = 1.05
DEFAULT_QUALITY_FEATURE_SCALE = 1.10
DEFAULT_QUALITY_LOSS_WEIGHT = 2.0
DEFAULT_MASK_TOKEN_SCALE = 1.0

DEFAULT_MODEL_CONFIG = (
    "black-forest-labs/FLUX.2-klein-4B:text_encoder/*.safetensors,"
    "black-forest-labs/FLUX.2-klein-base-4B:transformer/*.safetensors,"
    "black-forest-labs/FLUX.2-klein-4B:vae/diffusion_pytorch_model.safetensors"
)
DEFAULT_TOKENIZER_CONFIG = "black-forest-labs/FLUX.2-klein-4B:tokenizer/"

DEFAULT_LORA_TARGET_MODULES = ",".join(
    [
        "to_q",
        "to_k",
        "to_v",
        "to_out.0",
        "add_q_proj",
        "add_k_proj",
        "add_v_proj",
        "to_add_out",
        "linear_in",
        "linear_out",
        "to_qkv_mlp_proj",
    ]
    + [f"single_transformer_blocks.{index}.attn.to_out" for index in range(20)]
)
