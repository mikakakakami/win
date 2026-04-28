"""Backward-compat shim. All functionality moved to wmark.compress."""
from .compress import (  # noqa: F401
    SUPPORTED_METHODS,
    compress_gptq4 as quantize_gptq_4bit,
    compress_wanda as prune_wanda_50,
    get_calibration_data as get_calibration_samples,
    load_model as load_compressed,
)
