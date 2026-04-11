---
name: reproducibility-check
description: 'Verify an experiment can be reproduced from its commit and config. Use before sharing results, before publishing, after suspicious results, or when resuming old work. Covers environment, config, code, and output verification.'
---

# Reproducibility Check

## When to Use
- Before sharing or publishing results
- When results seem too good or too bad
- When resuming work after a break
- Before basing new experiments on previous results

## Procedure

### 1. Verify artifacts exist
- [ ] Commit hash recorded in results
- [ ] Config file saved in results directory
- [ ] Training logs saved
- [ ] Best checkpoint saved (if applicable)
- [ ] Results directory contains all expected outputs

### 2. Environment match
```bash
# Record current environment
pip freeze > environment_current.txt

# Compare with recorded environment (if exists)
diff environment_recorded.txt environment_current.txt
```
- Check Python version, CUDA version, key library versions (torch, numpy, etc.)
- Flag any version mismatches

### 3. Code match
```bash
# Check if current code matches results commit
git status              # Should be clean
git log --oneline -1    # Should match recorded commit hash
```
- If code has changed since the results were produced, checkout the exact commit
- Verify no uncommitted changes were used in the original run

### 4. Config match
- Compare config in results directory with current config
- Document ALL differences
- If configs differ, the reproduction must use the ORIGINAL config

### 5. Subset reproduction
- Run training on ~5% of data or 2–5 epochs with the EXACT config and commit
- Compare loss trajectory with original logs at the same point
- Acceptable variance: loss within 5% at same step (accounting for hardware differences)
- If variance exceeds 5%, investigate before claiming reproducibility

### 6. Report
```
## Reproducibility Check — [YYYY-MM-DD]
Experiment: <name>
Original commit: <hash>
Original date: <date>

### Artifact status
- Config: present / missing
- Logs: present / missing
- Checkpoint: present / missing

### Environment match
- Exact / Differences: <list>

### Code match
- Exact / Differences: <list>

### Subset reproduction
- Loss at step N: original <value> vs reproduced <value> (Δ <pct>%)
- Status: REPRODUCIBLE / DISCREPANCY / UNABLE TO VERIFY

### Verdict
<reproducible / not reproducible — reason>
```

## Acceptance Criteria
- [ ] All artifacts located and checked
- [ ] Environment documented and compared
- [ ] Code state verified
- [ ] Subset reproduction attempted (or documented why infeasible)
- [ ] Reproducibility verdict produced
