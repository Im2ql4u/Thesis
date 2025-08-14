from __future__ import annotations
import functools
import inspect
from typing import Any, Callable, Dict
import torch
import config as _cfg


def get_params_dict() -> Dict[str, Any]:
    c = _cfg.get()
    d = c.as_dict()
    d["torch_device"] = c.torch_device
    d["torch_dtype"] = c.torch_dtype
    d["act_fn"] = c.act_fn
    return d


def to_policy(t: torch.Tensor, params: Dict[str, Any] | None = None) -> torch.Tensor:
    if params is None:
        params = get_params_dict()
    return t.to(device=params["torch_device"], dtype=params["torch_dtype"])


def make_tensor(x, params: Dict[str, Any] | None = None) -> torch.Tensor:
    if params is None:
        params = get_params_dict()
    return torch.as_tensor(
        x, device=params["torch_device"], dtype=params["torch_dtype"]
    )


def inject_params(fn: Callable) -> Callable:
    sig = inspect.signature(fn)
    if "params" not in sig.parameters:
        raise TypeError(
            f"@inject_params: '{fn.__name__}' must accept a 'params' argument."
        )

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if "params" not in kwargs or kwargs["params"] is None:
            kwargs["params"] = get_params_dict()
        return fn(*args, **kwargs)

    return wrapper


def require_even_closed_shell(n_particles: int) -> None:
    if n_particles % 2 != 0:
        raise ValueError(
            f"Closed-shell requires even number of particles, got {n_particles}."
        )
