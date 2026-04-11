# Plan: Gradient-Quality Chain Improvements

Date: 2026-04-11
Status: active
Depends-on: plans/2026-04-10_stabilization-generalization.md (Phase 1 complete)

## Motivation

The Phase 1 reliability campaign (30 runs, N=6, 5 omegas, 3 seeds × 2 recipes)
revealed a causal chain governing collocation accuracy:

```
proposal mismatch → khat inflation → gradient noise → optimization stochasticity → run variance
```

This plan addresses each link of the chain, in order of expected impact per
engineering effort. Each phase produces independently valuable results.

## Current state (after Phase 1)

| ω     | Best err% | khat (robust avg) | Bottleneck |
|-------|-----------|-------------------|------------|
| 1.0   | 0.013%    | 0.50              | Evaluation noise (VMC-best ≈ 0.001%) |
| 0.5   | 0.002%    | 0.54              | Evaluation noise |
| 0.1   | 0.263%    | 0.70              | Borderline IS + optimization gets stuck |
| 0.01  | 0.237%    | 1.39              | IS broken, but robust recipe compensates |
| 0.001 | 0.250%    | 1.86              | IS broken, adaptive proposal partially helps |

Key insight: at ω=1.0, baseline seed 314 achieved VMC-best of +0.001% during
training but final IS evaluation reported +0.060%. The model IS this good.
We just can't reliably measure or select it.

## Phase 2: Evaluation fix (high value, low effort)

**Hypothesis:** Increasing evaluation sample count eliminates the 0.05-0.12%
VMC-best → final-eval gap at high ω, and improves checkpoint selection at all ω.

**Changes:**
- `--vmc-n 50000` (was 20000) — 2.5× more VMC probe samples
- `--n-eval 50000` (was 20000) — 2.5× more final evaluation samples  
- `--vmc-every 25` (was 50) — probe twice as often for finer checkpoint selection

**Grid:** N=6, ω ∈ {1.0, 0.5}, robust recipe only, 3 seeds
**Expected:** 6 runs, ~90 min each. Should close the evaluation gap and produce
consistent sub-0.02% results at ω=1.0 and sub-0.05% at ω=0.5.

**Success criterion:** Worst-seed error < 0.03% at ω=1.0. If met, this confirms
that the model expressiveness is sufficient and the remaining problem is
measurement.

**Decision gate:** If successful, apply same evaluation settings to all
subsequent phases. If unsuccessful, the problem is deeper than evaluation noise.

---

## Phase 3: Cascade warm-start (high value, medium effort)

**Hypothesis:** Initializing from a converged higher-ω checkpoint keeps the
proposal-target mismatch small throughout training, preventing khat inflation
at the source.

**Evidence from history:** The historical best at ω=0.1 (+0.091%) used
`--resume` from a converged ω=0.1 checkpoint. It started at 0.24% error and
optimized from there. Our Phase 1 runs start from a ω=1.0 pretrained init
and must bridge a large distribution shift.

**Cascade sequence:**
```
ω=1.0 (best seed from Phase 2)
  → ω=0.5 (resume, 800 epochs)
    → ω=0.1 (resume, 1200 epochs)
      → ω=0.01 (resume, 1500 epochs)
        → ω=0.001 (resume, 2000 epochs)
```

Each stage inherits the full checkpoint (Jastrow + backflow parameters).
The key idea is that adjacent ω values have similar |Ψ|² support, so the
proposal remains well-matched throughout.

**Grid:** N=6, cascade chain, robust recipe, 3 seeds (each seed runs the
full 5-step cascade independently)
**Expected:** 15 runs total (5 omegas × 3 seeds), roughly 6h wall time per
cascade chain on one GPU.

**Success criterion:** ω=0.1 error < 0.1%, ω=0.01 < 0.2%, ω=0.001 < 0.2%
with CV < 0.5 across seeds.

**Decision gate:** If cascade dramatically improves low-ω, adopt it as the
standard protocol. If it doesn't help, the problem is in the proposal shape
(needs non-Gaussian proposals).

---

## Phase 4: Proposal improvement (medium value, medium effort)

**Hypothesis:** The Gaussian mixture proposal is intrinsically mismatched to
|Ψ|² at low ω because the correlated electron cloud has shell structure that
isotropic Gaussians cannot represent. Increasing GMM components and/or using
ω-scaled widths can partially close this gap.

**Changes (additive to Phase 3):**
- `--gmm-components 16` (was 8) — more mixture components
- `--gmm-refit-every 15` (was 30) — refit more frequently as |Ψ|² evolves
- `--sigma-fs 0.6,1.0,1.5,2.5` — wider range of proposal scales

**Grid:** N=6, ω ∈ {0.1, 0.01, 0.001}, cascade-initialized, 3 seeds
**Expected:** 9 runs. Should lower khat by providing better tail coverage.

**Success criterion:** Average khat < 1.0 at ω=0.01 (currently 1.39).

---

## Phase 5: Extended training and multi-stage polishing (medium value, low effort)

**Hypothesis:** Given good initialization (cascade) and good gradients (proposal),
longer training with learning-rate annealing can squeeze out the remaining gap.

Historical evidence: the best ω=1.0 result (+0.009%) used a 3-stage chain
totaling 2150 epochs with decreasing LR. Our Phase 1 runs use 1200 epochs.

**Changes:**
- Stage 1: 1200 epochs at lr=5e-4 (current)
- Stage 2: 800 epochs at lr=2e-4 (resume from Stage 1 best)
- Stage 3: 500 epochs at lr=1e-4 (resume from Stage 2 best)

**Grid:** Apply to the best cascade result at each ω.
**Expected:** 5 runs (one per ω), ~3h each.

**Success criterion:** Improve on Phase 3 results by >30% at each ω.

---

## Phase 6: Scale to N=12 (high value, high effort)

**Hypothesis:** The gradient-quality chain applies identically at N=12, just
shifted to worse khat at every ω due to the higher dimensionality (24D vs 12D).
The cascade + robust recipe should unlock regimes that were previously
inaccessible.

**Grid:** N=12, ω ∈ {1.0, 0.5, 0.1}, cascade, robust recipe, 3 seeds
**Expected:** 9 runs. Serves as the generalization test.

**Success criterion:** Match or improve on Table 5 results (currently: +0.018%
at ω=1.0, +0.028% at ω=0.5, +0.122% at ω=0.1).

---

## Priority ordering and resource plan

| Phase | Runs | GPU-hours (est.) | Value | Effort | Priority |
|-------|------|------------------|-------|--------|----------|
| 2     | 6    | 9                | high  | low    | **now**  |
| 3     | 15   | 30               | high  | medium | next     |
| 4     | 9    | 18               | medium| medium | after 3  |
| 5     | 5    | 15               | medium| low    | after 4  |
| 6     | 9    | 36               | high  | high   | after 5  |

Total: ~44 runs, ~108 GPU-hours.

## Current State
- Phase 1: COMPLETE (30 runs, all rc=0, results analyzed)
- Phase 2: NOT STARTED
- Phase 3: NOT STARTED
- Phase 4: NOT STARTED
- Phase 5: NOT STARTED
- Phase 6: NOT STARTED
