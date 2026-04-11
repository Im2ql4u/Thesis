---
applyTo: "**/*.py"
---

# Python — ML code standards

- Type hints on every function signature and return type.
- NaN/Inf check immediately after any loss or gradient computation. Raise on detection: `if torch.isnan(loss): raise RuntimeError(f"NaN at step {step}")`.
- No hardcoded hyperparameters in source files. All tunables live in config YAML, loaded at runtime.
- Reproducibility: set `torch.manual_seed`, `numpy.random.seed`, `random.seed` from config. Set `torch.backends.cudnn.deterministic = True` when reproducibility is required.
- Results go to `results/YYYY-MM-DD_<descriptor>/`. Never overwrite previous runs.
- Logging: use Python `logging` module, not print statements. Log to both console and file.
- Every new module needs a known-answer test in `tests/`.
- Imports: standard library → third-party → local, separated by blank lines. No wildcard imports.
- Device handling: never hardcode `cuda:0`. Use a device variable from config or auto-detect.
