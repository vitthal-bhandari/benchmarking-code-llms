"""
Additional LanguageModel entries for LiveCodeBench lm_styles.py.

Run apply_patch.py (or copy-paste the ADDITIONS list into
LiveCodeBench/lcb_runner/lm_styles.py) to register these models.

LMStyle notes
-------------
- Qwen3.6         → QwQ      : Qwen3 thinking-capable model; uses Qwen chat template
- North Mini Code → LLaMa3   : Cohere model; approximate with LLaMa3 chat format
- Devstral Small 2→ LLaMa3   : Mistral v7 format; close enough for initial runs
- Poolside XS.2   → LLaMa3   : Format TBD; LLaMa3 as safe default
- Gemma4          → LLaMa3   : Gemma4 format TBD; LLaMa3 as safe default

All styles can be tuned once a first eval run confirms prompt formatting.
"""

from datetime import datetime
from lcb_runner.lm_styles import LanguageModel, LMStyle

ADDITIONS: list[LanguageModel] = [
    # ── Qwen3.6 35B-A3B (Alibaba) ──────────────────────────────────────────
    # Official FP8 checkpoint fits on L40S (48 GB): ~35 GB weights at FP8.
    # QwQ style enables <think> reasoning tokens, best for coding benchmarks.
    LanguageModel(
        model_name="Qwen/Qwen3.6-35B-A3B-FP8",
        model_repr="Qwen3.6-35B-A3B-FP8",
        model_style=LMStyle.QwQ,
        release_date=datetime(2025, 5, 1),
        link="https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8",
    ),
    # ── North Mini Code 30B-A3B (Cohere) ───────────────────────────────────
    # Official W4A16 quantized checkpoint: ~18-20 GB on L40S.
    LanguageModel(
        model_name="CohereLabs/North-Mini-Code-1.0-w4a16",
        model_repr="North-Mini-Code-w4a16",
        model_style=LMStyle.LLaMa3,
        release_date=datetime(2025, 5, 1),
        link="https://huggingface.co/CohereLabs/North-Mini-Code-1.0-w4a16",
    ),
    # ── Devstral Small 2 24B Dense (Mistral) ───────────────────────────────
    # Full BF16 is ~48 GB (too tight); load at 8-bit via vLLM quantization flag
    # or use bitsandbytes. Alternatively use the GGUF Q8 (25 GB).
    LanguageModel(
        model_name="mistralai/Devstral-Small-2-24B-Instruct-2512",
        model_repr="Devstral-Small-2-24B",
        model_style=LMStyle.LLaMa3,
        release_date=datetime(2025, 5, 1),
        link="https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512",
    ),
    # ── Poolside Laguna XS.2 33B-A3B (Poolside) ────────────────────────────
    # Official NVFP4 checkpoint. NOTE: vLLM mainline NVFP4 support may require
    # a recent vLLM version (>=0.8). Fall back to base checkpoint if unsupported.
    LanguageModel(
        model_name="poolside/Laguna-XS.2-NVFP4",
        model_repr="Laguna-XS.2-NVFP4",
        model_style=LMStyle.LLaMa3,
        release_date=datetime(2025, 5, 1),
        link="https://huggingface.co/poolside/Laguna-XS.2-NVFP4",
    ),
    # ── Gemma4 26B-A4B (Google) ─────────────────────────────────────────────
    # HF model ID TBD — update once confirmed.
    # Placeholder uses LLaMa3 style; may need a dedicated Gemma4 style later.
    LanguageModel(
        model_name="google/gemma-4-27b-it",   # TODO: verify exact HF ID
        model_repr="Gemma4-27B-IT",
        model_style=LMStyle.LLaMa3,
        release_date=datetime(2025, 5, 1),
        link="https://huggingface.co/google/gemma-4-27b-it",
    ),
]
