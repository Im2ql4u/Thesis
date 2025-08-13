# src/thesislib/config.py
from __future__ import annotations
from dataclasses import dataclass, asdict, replace
from pathlib import Path
import yaml  # you already installed pyyaml
from typing import Any, Dict


@dataclass(frozen=True)
class Config:
    # ---- Put your standard defaults here ----
    project_root: Path = Path(".").resolve()
    data_dir: Path = Path("data/processed")
    results_dir: Path = Path("results")
    seed: int = 42
    # ML-ish examples
    train_split: float = 0.8
    model_name: str = "baseline"
    learning_rate: float = 1e-3
    epochs: int = 100

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Convert Paths to str for readability
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
        return d


# A module-level singleton: all modules import this same object.
_cfg = Config()


def cfg() -> Config:
    """Read-only accessor for the current configuration."""
    return _cfg


def set_cfg(**overrides: Any) -> Config:
    """
    Non-destructively create a new config with overrides and install it globally.
    Example: set_cfg(seed=123, model_name="resnet")
    """
    global _cfg
    _cfg = replace(_cfg, **overrides)
    return _cfg


def load_yaml(path: Path | str) -> Config:
    """Load overrides from YAML and install them globally."""
    global _cfg
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    _cfg = replace(_cfg, **data)
    return _cfg
