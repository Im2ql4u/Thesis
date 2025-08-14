# src/utils.py
from __future__ import annotations
import functools
import inspect
from typing import Any, Callable, Dict
import torch
import config as _cfg  # flat layout: top-level module (not relative)


def get_params_dict() -> Dict[str, Any]:
    """
    Dict view of the current config, plus ready-to-use torch objects.
    """
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
    """
    Auto-inject a 'params' kwarg built directly from the current config if the
    caller didn't pass one. No dependency on other helpers to avoid NameError.
    """
    sig = inspect.signature(fn)
    if "params" not in sig.parameters:
        raise TypeError(
            f"@inject_params: '{fn.__name__}' must accept a 'params' argument."
        )

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if "params" not in kwargs or kwargs["params"] is None:
            c = _cfg.get()
            p = c.as_dict()
            # add ready-to-use runtime objects
            p["torch_device"] = c.torch_device
            p["torch_dtype"] = c.torch_dtype
            p["act_fn"] = c.act_fn
            kwargs["params"] = p
        return fn(*args, **kwargs)

    return wrapper


# Optional: notebook helper
def get_promoted_params(
    names: list[str] | None = None, include_runtime: bool = False
) -> Dict[str, Any]:
    d = _cfg.get().as_dict()
    if include_runtime:
        r = get_params_dict()
        for k in ("torch_device", "torch_dtype", "act_fn"):
            d[k] = r[k]
    if names:
        d = {k: d[k] for k in names}
    return d
