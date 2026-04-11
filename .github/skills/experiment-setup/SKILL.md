---
name: experiment-setup
description: 'Scaffold a new ML experiment. Use when starting a new experiment, creating a new training run, setting up a baseline, or beginning model development. Handles directory structure, config creation, environment verification, and baseline reference.'
---

# Experiment Setup

## When to Use
- Starting a new experiment or training run
- Creating a new baseline comparison
- Scaffolding a new model variant
- Setting up directory structure for a new research direction

## Procedure

### 1. Define the experiment
- State the hypothesis being tested in one sentence
- Identify the independent variable (what changes) and dependent variable (what we measure)
- Name the experiment: `YYYY-MM-DD_<short-descriptor>`

### 2. Create directory structure
```
results/<experiment-name>/
├── config.yaml          # Copied from source config
├── logs/                # Training logs
├── checkpoints/         # Model checkpoints
├── figures/             # Visualizations
└── README.md            # Experiment description and results summary
```

### 3. Create or copy config
- Start from an existing config or template in `config/`
- Change ONLY the parameters relevant to this experiment
- Document what changed and why in the experiment README
- Verify config loads without errors: `python -c "import yaml; yaml.safe_load(open('results/<name>/config.yaml'))"`

### 4. Verify environment
- Check GPU availability: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"`
- Check library versions: `pip list | grep -E 'torch|numpy|pandas'`
- Verify data exists and is accessible: `ls -la data/` and check expected files
- Run existing tests: `pytest -v tests/`

### 5. Establish baseline reference
- Identify the comparison point (previous best, published result, or simple baseline)
- Record baseline metrics in the experiment README
- If no baseline exists, document this explicitly — first experiments ARE the baseline

### 6. Commit scaffold
```bash
git add results/<experiment-name>/config.yaml results/<experiment-name>/README.md
git commit -m "experiment(<name>): scaffold and config"
```
Do NOT commit empty directories — add `.gitkeep` files if needed.

## Acceptance Criteria
- [ ] Experiment directory exists with config and README
- [ ] Config loads without errors
- [ ] Environment verified (GPU, libraries, data)
- [ ] Baseline reference documented
- [ ] Scaffold committed
