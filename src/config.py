from __future__ import annotations

import os
import random
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

import numpy as np
import torch


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


_ACTIVATIONS = {
    "gelu": torch.nn.GELU,
    "relu": torch.nn.ReLU,
    "tanh": torch.nn.Tanh,
    "silu": torch.nn.SiLU,
    "mish": getattr(torch.nn, "Mish", torch.nn.SiLU),
}


@dataclass(frozen=True)
class Config:
    # -----------------------------
    # physics / model
    # -----------------------------
    omega: float = 0.1

    # Basis selection & parameters
    basis: Literal["cart", "fd"] = "cart"  # "cart" (Cartesian HO) or "fd" (Fock–Darwin)
    emax: int = 2  # FD only: shell cutoff (2n+|m| <= emax)
    nx: int = 1  # Cartesian only: number of 1D x-basis functions
    ny: int = 1  # Cartesian only: number of 1D y-basis functions
    fd_make_real: bool = True  # FD Slater uses real cos/sin combos
    fd_idx: list | None = None

    # Coulomb / convolution controls
    kappa: float = 1.0  # dielectric screening (1/kappa) prefactor
    pad_factor: int = 2  # FFT padding for Coulomb convolution

    # -----------------------------
    # compute policy
    # -----------------------------
    device: str = _default_device()
    dtype: str = "float64"  # "float32" | "float64" | "float16" | "bfloat16"
    seed: int | None = 0

    # -----------------------------
    # training / architecture
    # -----------------------------
    hidden_dim: int = 64
    n_layers: int = 3
    act_fn_name: str = "gelu"
    learning_rate: float = 1e-4
    N_collocation: int = 2000
    n_epochs: int = 3000
    n_epochs_norm: int = 200

    # -----------------------------
    # system constants (kept as-is)
    # -----------------------------
    E: float = 0.44079
    V: float = 1.0
    d: int = 2
    n_particles: int = 2
    dimensions: int = 2

    # -----------------------------
    # grids / sampling
    # -----------------------------
    L: float = 8.0
    L_E: float = 9.0
    n_grid: int = 30
    batch_size: int = int(1e3)
    n_samples: int = int(1e5)

    # -----------------------------
    # paths
    # -----------------------------
    data_dir: str | None = None
    results_dir: str | None = None

    # -----------------------------
    # HF solver (optional overrides used by hartree_fock_closed_shell)
    # -----------------------------
    hf_max_iter: int = 100
    hf_tol: float = 1e-8
    hf_damping: float = 0.0
    hf_verbose: bool = True

    # --- helpers ---
    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

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

    @property
    def act_fn(self) -> torch.nn.Module:
        key = self.act_fn_name.lower()
        if key not in _ACTIVATIONS:
            raise ValueError(
                f"Unsupported activation '{self.act_fn_name}'. Valid: {sorted(_ACTIVATIONS)}"
            )
        return _ACTIVATIONS[key]()


_CURRENT = Config()


def get() -> Config:
    return _CURRENT


def update(**overrides) -> Config:
    global _CURRENT
    _CURRENT = replace(_CURRENT, **overrides)
    _apply_seed_policy(_CURRENT)
    return _CURRENT


@contextmanager
def override(**overrides):
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
    if cfg.seed is None:
        return
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    os.environ["PYTHONHASHSEED"] = str(cfg.seed)
