"""Omni-Persona v2.2 — minimal reproducible evaluation pipeline.

One cycle:  data -> infer -> judge -> score

    python -m omni_persona.infer  ...   # model predictions
    python -m omni_persona.judge  ...   # LLM-as-a-judge verdicts
    python -m omni_persona.score  ...   # guarded main-table row

Public symbols are re-exported lazily so that ``python -m omni_persona.<mod>``
does not double-import the submodule.
"""

import importlib
from typing import Any

__version__ = "2.2.0"

_EXPORTS = {
    "SYSTEM_PROMPT": "data",
    "build_multimodal_messages": "data",
    "load_benchmark": "data",
    "stratified_subset": "data",
    "InferenceClient": "infer",
    "JUDGE_SYSTEM": "judge",
    "JUDGE_USER_TEMPLATE": "judge",
    "JudgeClient": "judge",
    "is_abstain": "judge",
    "GROUP": "score",
    "compute_guarded": "score",
}

__all__ = [*_EXPORTS, "__version__"]


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        return getattr(importlib.import_module(f".{_EXPORTS[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
