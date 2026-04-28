"""Multi-bit LLM watermarking under model compression.

Core modules:
    wmark.mpac      MPAC encoder/decoder (Yoo et al. NAACL 2024)
    wmark.compress  Model compression (GPTQ, AWQ, WANDA, INT8, ...)
    wmark.metrics   bit_accuracy, exact_match
    wmark.data      Prompt loading (C4, OpenGen, ...)
    wmark.utils     Hash, RNG, radix conversion
"""

__version__ = "0.1.0"
