"""Omni-Persona RLVR training skeleton (Qwen2.5-Omni)."""

from .reward import reward_localize, reward_text_qa, reward_verify
from .dataset import OmniPersonaRLVRDataset, build_task_weights

__all__ = [
    "reward_localize",
    "reward_verify",
    "reward_text_qa",
    "OmniPersonaRLVRDataset",
    "build_task_weights",
]
