# config.py
from __future__ import annotations
from dataclasses import dataclass, asdict, replace
from contextlib import contextmanager
from typing import Any, Dict, Optional
import os
import random
import numpy as np
import torch


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():  # Apple Silicon
        return "mps"
    return "cpu"


@dataclass(frozen=True)
class Config:
    # ---- physics / model ----
    omega: float = 1.0

    # ---- compute policy ----
    device: str = _default_device()  # store as string for easy serialization
    dtype: str = "float64"  # one of: float32, float64, bfloat16, float16, etc.
    seed: Optional[int] = 0  # None = don't touch RNGs

    # ---- paths (extend as needed) ----
    data_dir: Optional[str] = None
    results_dir: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    # runtime helpers
    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.device)

    @property
    def torch_dtype(self) -> torch.dtype:
        mapping = {
            "float64": torch.float64,
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if self.dtype not in mapping:
            raise ValueError(f"Unsupported dtype '{self.dtype}'.")
        return mapping[self.dtype]


# ---- global current config (immutable) ----
_CURRENT = Config()


def get() -> Config:
    """Return the current Config (immutable)."""
    return _CURRENT


def update(**overrides) -> Config:
    """
    Install a new Config with fields changed.
    Example: config.update(omega=0.7, device="cuda", dtype="float64", seed=123)
    """
    global _CURRENT
    _CURRENT = replace(_CURRENT, **overrides)
    _apply_seed_policy(_CURRENT)
    return _CURRENT


@contextmanager
def override(**overrides):
    """
    Temporarily override config values inside a 'with' block.
    """
    global _CURRENT
    prev = _CURRENT
    try:
        _CURRENT = replace(_CURRENT, **overrides)
        _apply_seed_policy(_CURRENT)
        yield _CURRENT
    finally:
        _CURRENT = prev
        _apply_seed_policy(_CURRENT)


def _apply_seed_policy(cfg: Config) -> None:
    """Set RNG seeds if cfg.seed is not None."""
    if cfg.seed is None:
        return
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    os.environ["PYTHONHASHSEED"] = str(cfg.seed)
