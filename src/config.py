from __future__ import annotations

import os
import random
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
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

# ---------------------------------------------------------------------
# Reference DMC energies E(N, omega)  [fill/extend as needed]
# omega keys should be floats like 0.1, 0.28, 0.5, 1.0
# ---------------------------------------------------------------------
DMC_ENERGIES: dict[int, dict[float, float]] = {
    2: {0.1: 0.44079, 0.28: 1.02164, 0.5: 1.65977, 1.0: 3.00000},
    6: {0.1: 3.55385, 0.28: 7.60019, 0.5: 11.78484, 1.0: 20.15932},
    12: {0.1: 12.26984, 0.28: 25.63577, 0.5: 39.15960, 1.0: 65.70010},
    20: {0.1: 29.97790, 0.28: 61.92680, 0.5: 93.87520, 1.0: 155.88220},
}
_SUPPORTED_OMEGAS = sorted({w for table in DMC_ENERGIES.values() for w in table.keys()})


def _snap_omega(omega: float) -> float:
    # snap to nearest supported omega
    return min(_SUPPORTED_OMEGAS, key=lambda w: abs(w - float(omega)))


def _lookup_dmc_energy(n_particles: int, omega: float) -> float:
    n = int(n_particles)
    w = _snap_omega(float(omega))
    if n not in DMC_ENERGIES or w not in DMC_ENERGIES[n]:
        raise KeyError(
            f"No DMC energy for N={n}, omega≈{omega} (snapped to {w}). "
            f"Known omegæ: {_SUPPORTED_OMEGAS}; known N: {sorted(DMC_ENERGIES)}"
        )
    return float(DMC_ENERGIES[n][w])


@dataclass(frozen=True)
class Config:
    # -----------------------------
    # physics / model
    # -----------------------------
    omega: float = 0.1

    # Basis selection & parameters
    basis: Literal["cart", "fd"] = "cart"
    emax: int = 2
    nx: int = 1
    ny: int = 1
    fd_make_real: bool = True
    fd_idx: list | None = None

    # Coulomb / convolution controls
    kappa: float = 1.0
    pad_factor: int = 2
    cart_scale: np.ndarray | None = field(default=None, repr=False, compare=False)

    # -----------------------------
    # compute policy
    # -----------------------------
    device: str = _default_device()
    dtype: str = "float64"
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
    std: float = 1.8

    # -----------------------------
    # system constants
    # -----------------------------
    # E can be "auto" to pull DMC(E) from table based on (n_particles, omega)
    E: float | Literal["auto"] = "auto"
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
    # HF solver
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


def _apply_seed_policy(cfg: Config) -> None:
    if cfg.seed is None:
        return
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    os.environ["PYTHONHASHSEED"] = str(cfg.seed)


def _maybe_set_auto_energy(cfg: Config) -> Config:
    # if E is "auto", set it from DMC table using current (n_particles, omega)
    if isinstance(cfg.E, str) and cfg.E == "auto":
        E_val = _lookup_dmc_energy(cfg.n_particles, cfg.omega)
        return replace(cfg, E=E_val)
    return cfg


def get() -> Config:
    return _CURRENT


def update(**overrides) -> Config:
    global _CURRENT
    _CURRENT = replace(_CURRENT, **overrides)
    _CURRENT = _maybe_set_auto_energy(_CURRENT)
    _apply_seed_policy(_CURRENT)
    return _CURRENT


@contextmanager
def override(**overrides):
    global _CURRENT
    prev = _CURRENT
    try:
        tmp = replace(_CURRENT, **overrides)
        tmp = _maybe_set_auto_energy(tmp)
        _apply_seed_policy(tmp)
        object.__setattr__(globals(), "_CURRENT", tmp)  # keep global in sync inside context
        yield tmp
    finally:
        object.__setattr__(globals(), "_CURRENT", prev)
        _apply_seed_policy(prev)
