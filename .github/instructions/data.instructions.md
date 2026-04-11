---
applyTo: "data/**"
---

# Data — integrity and handling rules

- Raw data is read-only. Never modify files in `data/raw/`.
- Processed data goes to `data/processed/`. Track the script and config that produced it.
- Always validate after every preprocessing step: check shape, dtype, value range, missing values.
- Check for data leakage: ensure train/val/test splits have no overlapping samples.
- Document data provenance: source, download date, version, preprocessing steps.
- Large data files go in `.gitignore`. Document how to obtain them in README or `data/README.md`.
- Class distributions: log and verify balance across splits. Report imbalance as a risk.
