---
name: data-audit
description: 'Audit dataset integrity before training. Use when validating a new dataset, checking for data leakage, verifying preprocessing, diagnosing data-related training failures, or before any first training run. Covers integrity, distributions, splits, and leakage checks.'
---

# Data Audit

## When to Use
- Before first training run on a dataset
- After preprocessing changes
- When training results are suspicious (unexpectedly high or low performance)
- When adding new data sources
- After data pipeline modifications

## Procedure

### 1. Inventory
- List all data files: `find data/ -type f | head -50`
- Check sizes: `du -sh data/*/`
- Verify expected files are present; flag unexpected files

### 2. Integrity checks
Load each data source and check for:
- Missing values: `df.isnull().sum()`
- Duplicate rows: `df.duplicated().sum()`
- Corrupted entries (NaN, Inf, empty strings)
- Expected shape and dtype

For image data: verify files open and have expected dimensions.
For text data: check encoding, empty documents, tokenization bounds.

### 3. Distribution analysis
For each feature and the target variable:
- Compute basic statistics (mean, std, min, max, quartiles)
- Check for extreme outliers (>5 sigma)
- Check class balance for classification tasks
- Visualize distributions if matplotlib available

Report any unexpected patterns (bimodal distributions, heavy skew, zero-inflated).

### 4. Split validation
- Verify train/val/test splits exist and have expected sizes
- Check NO overlap between splits: `len(set(train_ids) & set(val_ids))` must be 0
- Check stratification: class proportions should be similar across splits
- If time-series: verify temporal ordering (no future data in training set)

### 5. Leakage detection
- Check for target information in features (correlation > 0.95 with target)
- Check for identifier columns that shouldn't be features
- Verify no preprocessing was fitted on test data
- If data augmentation: verify augmented samples don't cross split boundaries

### 6. Report
```
## Data Audit — [YYYY-MM-DD]
Dataset: <name>
Total samples: <n> (train: <n> / val: <n> / test: <n>)
Features: <n> | Target: <name> (<type>)

### Integrity
- Missing values: <none / details>
- Duplicates: <none / details>
- Corruptions: <none / details>

### Distribution concerns
- <finding or "none detected">

### Split validation
- Overlap: <none / detected>
- Stratification: <balanced / imbalanced — details>

### Leakage risk
- <none detected / findings>

### Recommendation
- <proceed / investigate <issue> before training>
```

## Acceptance Criteria
- [ ] All data files loaded and inspected
- [ ] No integrity issues, or issues documented with severity
- [ ] No split overlap
- [ ] No leakage detected, or leakage risk documented
- [ ] Audit report produced
