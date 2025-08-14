# utils.py
from __future__ import annotations
import functools
import inspect
from typing import Any, Callable, Dict
import torch
import config as _cfg

# ---------- public helpers ----------


def get_params_dict() -> Dict[str, Any]:
    """
    Return a plain dict view of the current config,
    plus ready-to-use torch objects (torch_device, torch_dtype).
    """
    c = _cfg.get()
    d = c.as_dict()
    d["torch_device"] = c.torch_device
    d["torch_dtype"] = c.torch_dtype
    return d


def to_policy(t: torch.Tensor, params: Dict[str, Any] | None = None) -> torch.Tensor:
    """
    Move tensor to (device, dtype) specified by params or current config.
    """
    if params is None:
        params = get_params_dict()
    return t.to(device=params.get("torch_device"), dtype=params.get("torch_dtype"))


def make_tensor(x, params: Dict[str, Any] | None = None) -> torch.Tensor:
    """
    Create a tensor on the configured (device, dtype).
    """
    if params is None:
        params = get_params_dict()
    return torch.as_tensor(
        x, device=params.get("torch_device"), dtype=params.get("torch_dtype")
    )


# ---------- the wrapper (decorator) you asked for ----------


def inject_params(fn: Callable) -> Callable:
    """
    Decorator that auto-injects a 'params' kwarg with the *current* config
    (as a dict) if the caller didn't pass one.

    - Keeps functions PURE (no hidden global reads).
    - Still allows explicit per-call overrides (pass params=...).
    - Plays nice with testing & logging.

    Usage:
        @inject_params
        def f(x, *, params=None):
            omega = params["omega"]
            ...
    """
    sig = inspect.signature(fn)
    # Ensure 'params' exists in the signature (keyword-only is recommended)
    if "params" not in sig.parameters:
        raise TypeError(
            f"@inject_params: function '{fn.__name__}' must accept a 'params' argument."
        )

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if "params" not in kwargs or kwargs["params"] is None:
            kwargs["params"] = get_params_dict()
        return fn(*args, **kwargs)

    return wrapper


# ---------- optional: tiny validator you can sprinkle in hot paths ----------


def require_even_closed_shell(n_particles: int) -> None:
    if n_particles % 2 != 0:
        raise ValueError(
            f"Closed-shell requires even number of particles, got {n_particles}."
        )
