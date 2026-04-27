"""Prompt loading. Pilot-A uses C4 validation; later experiments add OpenGen + Essays."""
from __future__ import annotations

from datasets import load_dataset


def load_c4_prompts(
    n: int = 500,
    prompt_tokens: int = 30,
    min_text_chars: int = 200,
    seed: int = 0,
) -> list[str]:
    """Sample n prompts from C4 validation. Each prompt = first prompt_tokens whitespace tokens.

    We use whitespace tokenization here purely to slice prompt length;
    actual model tokenization happens at generation time.
    """
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    prompts: list[str] = []
    for i, row in enumerate(ds):
        if len(prompts) >= n:
            break
        text = row["text"]
        if len(text) < min_text_chars:
            continue
        words = text.split()
        if len(words) < prompt_tokens + 20:
            continue
        prompts.append(" ".join(words[:prompt_tokens]))
    if len(prompts) < n:
        raise RuntimeError(f"only collected {len(prompts)} prompts from C4 stream")
    return prompts
