# src/thesislib/utils.py
from __future__ import annotations
from pathlib import Path
from thesislib.config import cfg


def ensure_dirs():
    c = cfg()
    for p in [c.data_dir, c.results_dir]:
        Path(p).mkdir(parents=True, exist_ok=True)


def save_text(name: str, text: str) -> Path:
    c = cfg()
    out = Path(c.results_dir) / f"{name}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return out


def split_ratio() -> float:
    # functions can use whatever is in cfg
    return cfg().train_split
