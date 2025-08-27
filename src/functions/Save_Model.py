import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

from utils import inject_params


# ---------- path helpers ----------
def omega_tag(omega: float) -> str:
    """
    Format like the user's convention:
      1.0 -> w_10, 0.5 -> w_05, 0.1 -> w_01
    """
    return f"w_{int(round(omega * 10)):02d}"


def model_dir(root: str, n_particles: int, omega: float) -> Path:
    """
    models/<Np>p/w_XX/
    """
    return Path(root) / f"{n_particles}p" / omega_tag(omega)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ---------- module helpers ----------
def unwrap_module(mod):
    """
    If you wrapped a module (e.g., DetachWrapper(mod) with .mod),
    return the underlying .mod by default; otherwise return mod.
    """
    return getattr(mod, "mod", mod)


def has_state_dict(obj) -> bool:
    return hasattr(obj, "state_dict") and callable(obj.state_dict)


def is_nn_module(obj) -> bool:
    return isinstance(obj, nn.Module)


# ---------- save / load ----------
@inject_params
def save_model(
    model,
    *,
    root: str = "../results/models",
    name: str,  # e.g. "mapper", "backflow", "f_net"
    unwrap: bool = True,
    extra_meta: dict | None = None,
    params=None,
) -> Path:
    """
    Save a model under models/<Np>p/w_XX/<name>.pt with a meta.json.
    Prefers state_dict for nn.Modules. Falls back to whole-object torch.save if needed.
    Returns the path to the saved .pt file.
    """
    n_particles = params["n_particles"]
    omega = params["omega"]
    # Resolve directory and filenames
    out_dir = model_dir(root, n_particles, omega)
    ensure_dir(out_dir)
    pt_path = out_dir / f"{name}.pt"
    meta_path = out_dir / f"{name}.meta.json"

    # Resolve which object to save (unwrap wrapper if requested)
    obj = unwrap_module(model) if unwrap else model

    # Prepare the payload
    payload = {}
    meta = {
        "saved_at": datetime.utcnow().isoformat() + "Z",
        "name": name,
        "n_particles": int(n_particles),
        "omega": float(omega),
        "omega_tag": omega_tag(omega),
        "class": obj.__class__.__name__,
        "file_type": None,  # "state_dict" or "object"
    }
    if extra_meta:
        meta.update(extra_meta)

    # Prefer state_dict when possible
    if is_nn_module(obj) and has_state_dict(obj):
        payload = {
            "type": "state_dict",
            "class": obj.__class__.__name__,
            "state_dict": obj.state_dict(),
        }
        meta["file_type"] = "state_dict"
        torch.save(payload, pt_path)
    else:
        payload = {"type": "object", "object": obj}
        meta["file_type"] = "object"
        torch.save(payload, pt_path)

    # Write meta
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return pt_path


@inject_params
def load_model_into(
    model: nn.Module,
    *,
    root: str = "../results/models",
    name: str,
    strict: bool = True,
    params=None,
) -> nn.Module:
    """
    Load weights into an already constructed Module.
    Expects a .pt saved by save_model with 'state_dict' type.
    """
    n_particles = params["n_particles"]
    omega = params["omega"]
    pt_path = model_dir(root, n_particles, omega) / f"{name}.pt"
    payload = torch.load(pt_path, map_location="cpu")
    if payload.get("type") != "state_dict":
        raise ValueError(
            f"{pt_path} does not contain a state_dict (found type={payload.get('type')})."
        )
    state = payload["state_dict"]
    model.load_state_dict(state, strict=strict)
    return model


@inject_params
def load_object(*, root: str = "../results/models", name: str, params=None):
    """
    Load a full object saved as type='object' (fallback path).
    Useful for non-Module mappers. Returns the Python object.
    """
    n_particles = params["n_particles"]
    omega = params["omega"]
    pt_path = model_dir(root, n_particles, omega) / f"{name}.pt"
    payload = torch.load(pt_path, map_location="cpu")
    if payload.get("type") != "object":
        raise ValueError(
            f"{pt_path} does not contain a full object (found type={payload.get('type')})."
        )
    return payload["object"]
