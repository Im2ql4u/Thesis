# Plan: Optimizer Tournament — Breaking the Low-ω Plateau

Date: 2026-04-05
Status: draft

## Project objective
Produce a defensible, multi-seed consistency matrix of weak-form collocation results across the (N, ω) grid, with honest characterization of achievable accuracy per regime.

## Objective
Push N=6 ω=0.1 from +0.076% toward ≤0.02%, and N=6 ω=0.001 from +0.119% toward ≤0.05%, by systematically testing natural-gradient optimizers that were never tried post-importance-sampling-bugfix. Then consolidate with multi-seed runs and scale to N=12.

Success condition: At least one optimizer/loss combination that demonstrably and reproducibly outperforms the current best at ω=0.1 and/or ω=0.001, validated by 100k heavy-VMC.

## Context

### What triggered this plan

The Consistency Campaign (Phases 0–4) established hard numbers with 100k heavy-VMC but hit a wall:

| Checkpoint | N | ω | err (%) | Recipe |
|---|---:|---:|---:|---|
| p4m1_fdmatched | 6 | 0.1 | **+0.076** | FD-colloc, Huber, proximal, 4000ep, cosine LR, Adam |
| p4m2_nogate | 6 | 0.001 | **+0.119** | REINFORCE, no-gate, oversample=32, 4000ep, cosine LR, Adam |
| p3b baseline | 6 | 0.1 | +0.127 | FD-colloc, 1500ep, Adam |
| p3a baseline | 6 | 0.001 | +0.213 | no-gate, oversample=32, 1133ep, Adam |

Phase 4 improved over Phase 3 by ~40-50% — good but insufficient. The target is ≤0.01%, and the best result is 7.6× away.

### The central hypothesis

**The ω≤0.1 plateau is an optimization barrier, not an architecture or sampling barrier.**

Evidence:
1. **SR reaches +0.002% at ω=1.0 in essentially one run** (CG-SR campaign, 2026-03-17). Adam needs multi-stage chains to get there.
2. **The decision to ban SR at low ω (DECISIONS.md 2026-03-26) was based on pre-bugfix evidence.** The importance sampling bug (found 2026-03-24) exponentially corrupted importance weights at low ω, making Fisher estimation garbage → SR appeared unstable → SR was banned. Post-bugfix, SR at low ω was NEVER tested.
3. **Adam's core weakness maps exactly onto the low-ω difficulty.** At low ω, Var(E_L) is large. Adam's second moment conflates Var(E_L) with Fisher curvature, suppressing step size for ALL parameters. Natural gradient separates these.
4. **The architecture reaches +0.002% at ω=1.0** with 49K params. Same architecture at ω=0.1 should have similar capacity — the ground-state structure at ω=0.1 is arguably simpler (more diffuse, less cusp structure relative to scale).

### What has NOT been tried

| Technique | ω=1.0 | ω=0.1 | ω=0.001 | Notes |
|---|---|---|---|---|
| CG-SR | ✓ (+0.002%) | **NEVER post-fix** | **NEVER post-fix** | Banned prematurely |
| Woodbury-SR | ✓ | **NEVER post-fix** | **NEVER post-fix** | Same |
| Diagonal Fisher | ✓ | **NEVER** | **NEVER** | Implemented, never systematically tested at low ω |
| FD-colloc + any SR | **NEVER** | **NEVER** | **NEVER** | Two best ingredients never combined |
| Gradient accumulation | NEVER | NEVER | NEVER | Standard variance reduction, not implemented |

This is the single largest unexplored territory in the codebase. Every result at ω≤0.1 was obtained with Adam only.

## Approach

Run a systematic factorial tournament across optimizers and loss functions at the two critical regimes (ω=0.1 and ω=0.001). All runs warm-start from the current best Phase 4 checkpoints to avoid wasting compute on the Adam-easy early phase. Include one from-scratch SR run at ω=0.1 to test whether SR can solve the whole problem in one shot.

8 GPUs (RTX 2080 Ti, 11GB each) are available. All phases use tmux for persistence.

**Regime-specific strategy:**
- **ω=0.1:** FD-colloc is the proven loss; test {DiagFisher, CG-SR, Woodbury-SR} against Adam baseline
- **ω=0.001:** REINFORCE + no-gate is the proven loss; test {DiagFisher, CG-SR} against Adam baseline
- **ω=0.01:** Bridge gap — apply ω=0.1 winner as transfer to ω=0.01

**Key parameter choices for natural gradient at low ω:**
- Higher damping start (0.01 vs usual 0.005) due to noisier Fisher estimate
- Higher Fisher EMA (0.99 vs 0.95) for more smoothing
- Fisher subsample 2048 (vs default 1024) for better Fisher estimate
- LR warmup (30-50 epochs) to prevent initial SR instability
- Moderate CG iterations (50 i/o 100) to balance cost vs quality

**Timing estimates per optimizer (N=6, n-coll=8192):**
| Optimizer | ~sec/epoch | 4000ep wall | 6000ep wall |
|---|---:|---:|---:|
| Adam | ~10s | ~11h | ~17h |
| DiagFisher+SGD | ~11s | ~12h | ~18h |
| Woodbury-SR (sub=1024) | ~20s | ~22h | ~33h |
| CG-SR (50 iters, sub=1024) | ~25s | ~28h | — |

## Foundation checks (must pass before new code)

- [x] Data pipeline known-input check (Phase 0 — verified)
- [x] Split/leakage validity check (N/A — no train/test split)
- [x] Baseline existence (p4m1 +0.076% at ω=0.1, p4m2 +0.119% at ω=0.001)
- [x] Importance sampling bug fixed (2026-03-24)
- [x] Rolling-best checkpoint metadata fixed (Phase 3.5)
- [ ] Natural gradient + FD-colloc compose correctly (Step 1.1)
- [ ] Optimizer state handling on resume verified (Step 1.2)

## Scope

**In scope:**
- Systematic optimizer comparison at N=6 ω=0.1 and ω=0.001
- One from-scratch CG-SR run at ω=0.1
- Bridge to ω=0.01 with winning recipe
- Multi-seed consolidation of winning recipe
- Scale to N=12 with winning recipe
- Small code change: gradient accumulation (if needed)

**Out of scope:**
- Architecture changes (backflow hidden size, new network types, KFAC)
- New sampling schemes (MCMC, normalizing flows, learned proposals)
- N=20 (postpone until N=12 is consolidated)
- New loss functions not already in the codebase
- ω≥0.5 (already near target; verify-only in Phase 5)

---

## Phase 1 — Smoke Tests: Verify SR Works at Low ω Post-Bugfix (2-3h)

**Goal:** Confirm that natural gradient methods run stably at ω=0.1 and ω=0.001 post-bugfix. Identify any integration issues (FD-colloc + natural grad, resume vs fresh optimizer state).

**Estimated scope:** 6 short runs, no code changes needed. ~30 min per run.

### Step 1.1 — Run 6 smoke tests (100 epochs each)

**What:** Launch 6 parallel 100-epoch runs, one per GPU, all resuming from Phase 4 checkpoints. Monitor for NaN, energy degradation, or instability.

**Test matrix:**

| GPU | Tag | ω | Optimizer | Loss | Resume from |
|---:|---|---:|---|---|---|
| 0 | smoke_df_fd_w01 | 0.1 | DiagFisher | FD-colloc | p4m1_fdmatched_n6w01.pt |
| 1 | smoke_cg_fd_w01 | 0.1 | CG-SR(50i) | FD-colloc | p4m1_fdmatched_n6w01.pt |
| 2 | smoke_wb_fd_w01 | 0.1 | Woodbury-SR | FD-colloc | p4m1_fdmatched_n6w01.pt |
| 3 | smoke_df_re_w01 | 0.1 | DiagFisher | REINFORCE | p4m1_fdmatched_n6w01.pt |
| 4 | smoke_df_re_w001 | 0.001 | DiagFisher | REINFORCE | p4m2_nogate_n6w001.pt |
| 5 | smoke_cg_re_w001 | 0.001 | CG-SR(50i) | REINFORCE | p4m2_nogate_n6w001.pt |

**CLI for GPU 0 (DiagFisher + FD-colloc at ω=0.1):**
```bash
CUDA_MANUAL_DEVICE=0 python3 src/run_weak_form.py \
  --tag smoke_df_fd_w01 --mode bf --resume results/arch_colloc/p4m1_fdmatched_n6w01.pt \
  --n-elec 6 --omega 0.1 --seed 42 \
  --loss-type fd-colloc --fd-h 0.01 --fd-huber-delta 0.5 --prox-mu 0.1 \
  --n-coll 8192 --oversample 8 --micro-batch 1024 \
  --epochs 100 --lr 2e-4 --lr-jas 2e-5 --lr-min-frac 0.02 \
  --natural-grad --sr-mode diagonal --fisher-damping 0.01 --fisher-ema 0.99 \
  --fisher-probes 4 --fisher-subsample 2048 --nat-momentum 0.9 \
  --patience 0 --vmc-every 50 --vmc-n 10000 --save-best-window 20
```

**CLI for GPU 1 (CG-SR + FD-colloc at ω=0.1):**
```bash
CUDA_MANUAL_DEVICE=1 python3 src/run_weak_form.py \
  --tag smoke_cg_fd_w01 --mode bf --resume results/arch_colloc/p4m1_fdmatched_n6w01.pt \
  --n-elec 6 --omega 0.1 --seed 42 \
  --loss-type fd-colloc --fd-h 0.01 --fd-huber-delta 0.5 --prox-mu 0.1 \
  --n-coll 4096 --oversample 8 --micro-batch 1024 \
  --epochs 100 --lr 2e-4 --lr-jas 2e-5 --lr-min-frac 0.02 \
  --natural-grad --sr-mode cg --sr-cg-iters 50 --fisher-damping 0.01 \
  --fisher-subsample 1024 --sr-max-param-change 0.1 --sr-trust-region 1.0 \
  --patience 0 --vmc-every 50 --vmc-n 10000 --save-best-window 20
```

**CLI for GPU 2 (Woodbury-SR + FD-colloc at ω=0.1):**
```bash
CUDA_MANUAL_DEVICE=2 python3 src/run_weak_form.py \
  --tag smoke_wb_fd_w01 --mode bf --resume results/arch_colloc/p4m1_fdmatched_n6w01.pt \
  --n-elec 6 --omega 0.1 --seed 42 \
  --loss-type fd-colloc --fd-h 0.01 --fd-huber-delta 0.5 --prox-mu 0.1 \
  --n-coll 4096 --oversample 8 --micro-batch 1024 \
  --epochs 100 --lr 2e-4 --lr-jas 2e-5 --lr-min-frac 0.02 \
  --natural-grad --sr-mode woodbury --fisher-damping 0.01 \
  --fisher-subsample 1024 --sr-max-param-change 0.1 --sr-trust-region 1.0 \
  --patience 0 --vmc-every 50 --vmc-n 10000 --save-best-window 20
```

**CLI for GPU 3 (DiagFisher + REINFORCE at ω=0.1):**
```bash
CUDA_MANUAL_DEVICE=3 python3 src/run_weak_form.py \
  --tag smoke_df_re_w01 --mode bf --resume results/arch_colloc/p4m1_fdmatched_n6w01.pt \
  --n-elec 6 --omega 0.1 --seed 42 \
  --loss-type reinforce --direct-weight 0 --clip-el 5.0 \
  --n-coll 8192 --oversample 8 --micro-batch 1024 \
  --epochs 100 --lr 2e-4 --lr-jas 2e-5 --lr-min-frac 0.02 \
  --natural-grad --sr-mode diagonal --fisher-damping 0.01 --fisher-ema 0.99 \
  --fisher-probes 4 --fisher-subsample 2048 --nat-momentum 0.9 \
  --patience 0 --vmc-every 50 --vmc-n 10000 --save-best-window 20
```

**CLI for GPU 4 (DiagFisher + REINFORCE at ω=0.001):**
```bash
CUDA_MANUAL_DEVICE=4 python3 src/run_weak_form.py \
  --tag smoke_df_re_w001 --mode bf --resume results/arch_colloc/p4m2_nogate_n6w001.pt \
  --n-elec 6 --omega 0.001 --seed 42 \
  --loss-type reinforce --direct-weight 0 --clip-el 5.0 \
  --n-coll 4096 --oversample 32 --micro-batch 1024 \
  --epochs 100 --lr 1e-4 --lr-jas 1e-5 --lr-min-frac 0.02 \
  --natural-grad --sr-mode diagonal --fisher-damping 0.01 --fisher-ema 0.99 \
  --fisher-probes 4 --fisher-subsample 2048 --nat-momentum 0.9 \
  --patience 0 --vmc-every 50 --vmc-n 10000 --save-best-window 20
```

**CLI for GPU 5 (CG-SR + REINFORCE at ω=0.001):**
```bash
CUDA_MANUAL_DEVICE=5 python3 src/run_weak_form.py \
  --tag smoke_cg_re_w001 --mode bf --resume results/arch_colloc/p4m2_nogate_n6w001.pt \
  --n-elec 6 --omega 0.001 --seed 42 \
  --loss-type reinforce --direct-weight 0 --clip-el 5.0 \
  --n-coll 4096 --oversample 32 --micro-batch 1024 \
  --epochs 100 --lr 1e-4 --lr-jas 1e-5 --lr-min-frac 0.02 \
  --natural-grad --sr-mode cg --sr-cg-iters 50 --fisher-damping 0.01 \
  --fisher-subsample 1024 --sr-max-param-change 0.1 --sr-trust-region 1.0 \
  --patience 0 --vmc-every 50 --vmc-n 10000 --save-best-window 20
```

**Files:** `scripts/launch_tournament_phase1_smoke.sh`
**Acceptance check:** All 6 JSONL logs exist with 100 entries. No NaN in final energy. Final energy ≤ starting energy (no degradation). `python3 -c "import json; [print(json.loads(l)['E']) for l in open('outputs/tournament/smoke_df_fd_w01_epochs.jsonl')][-1]"` → energy ≤ 3.557 (starting energy of p4m1 checkpoint).
**Risk:** `--resume` may try to restore Adam optimizer state when switching to SGD+momentum. If so, need to use `--init-jas`/`--init-bf` instead. Test this explicitly.

### Step 1.2 — Analyze smoke results and calibrate

**What:** For each smoke test, extract: final E, E trajectory trend, ESS stability, seconds per epoch. Compare to Phase 4 baselines to ensure no degradation. Determine which combinations are stable enough for long runs. Calibrate wall-time estimates for Phase 2.

**Files:** Smoke test JSONL logs
**Acceptance check:** A summary table with pass/fail per combination, measured sec/epoch, and recommended epoch count for Phase 2.
**Risk:** If ALL natural-gradient variants degrade energy relative to the starting checkpoint, the hypothesis is wrong and we need to revisit.

**Phase 1 failure protocol:**
- If CG-SR fails but DiagFisher works: proceed with DiagFisher only (cheaper anyway)
- If all SR modes fail at ω=0.1: check Fisher eigenspectrum, try higher damping (0.05-0.1), reduce LR to 5e-5
- If all SR modes work at ω=0.1 but fail at ω=0.001: proceed with ω=0.1 only, keep Adam for ω=0.001
- If everything fails: the optimization hypothesis is wrong → revisit architecture/sampling (out of scope here)

---

## Phase 2 — Main Optimizer Tournament at N=6 (20-28h)

**Goal:** Find the best optimizer/loss combination at ω=0.1 and ω=0.001 by running 8 parallel experiments and comparing heavy-VMC results.

**Depends on:** Phase 1 smoke tests pass (≥3 of 6 combinations stable).

**Estimated scope:** 8 long runs, 1 per GPU, outputs in `outputs/tournament/phase2/`. Launcher script + eval script.

### Step 2.1 — Launch 8-way tournament

**What:** All 8 GPUs run simultaneously. Runs are designed to finish within ~24h.

**ω=0.1 runs (GPUs 0-4), all resume from `p4m1_fdmatched_n6w01.pt`:**

| GPU | Tag | Optimizer | Loss | Epochs | n-coll | Key params |
|---:|---|---|---|---:|---:|---|
| 0 | t2_df_fd_w01 | DiagFisher | FD-colloc | 6000 | 8192 | damping=0.01, ema=0.99, sub=2048, lr=2e-4 |
| 1 | t2_cg_fd_w01 | CG-SR(50i) | FD-colloc | 2000 | 4096 | damping=0.01, sub=1024, lr=2e-4 |
| 2 | t2_wb_fd_w01 | Woodbury-SR | FD-colloc | 2000 | 4096 | damping=0.01, sub=1024, lr=2e-4 |
| 3 | t2_df_re_w01 | DiagFisher | REINFORCE | 6000 | 8192 | damping=0.01, ema=0.99, sub=2048, lr=2e-4 |
| 4 | t2_adam_fd_w01 | Adam | FD-colloc | 8000 | 8192 | lr=5e-5, baseline comparison |

**ω=0.001 runs (GPUs 5-6), all resume from `p4m2_nogate_n6w001.pt`:**

| GPU | Tag | Optimizer | Loss | Epochs | n-coll | Key params |
|---:|---|---|---|---:|---:|---|
| 5 | t2_df_re_w001 | DiagFisher | REINFORCE | 6000 | 4096 | oversample=32, damping=0.01, ema=0.99, lr=1e-4 |
| 6 | t2_cg_re_w001 | CG-SR(50i) | REINFORCE | 2000 | 4096 | oversample=32, damping=0.01, lr=1e-4 |

**ω=0.1 from scratch (GPU 7) — the decisive test:**

| GPU | Tag | Optimizer | Loss | Epochs | n-coll | Key params |
|---:|---|---|---|---:|---:|---|
| 7 | t2_cg_scratch_w01 | CG-SR(50i) | FD-colloc | 3000 | 4096 | FROM SCRATCH (bf_ctnn_vcycle.pt), damping anneal 0.01→0.001, lr=5e-4 |

The from-scratch CG-SR run is the most informative: if SR converges from the basic BF checkpoint to ≤0.03% at ω=0.1 in one run (like it does at ω=1.0), it conclusively proves the optimization hypothesis.

**All runs include:** `--lr-warmup-epochs 30 --lr-warmup-init-frac 0.1 --lr-min-frac 0.01 --vmc-every 50 --vmc-n 15000 --save-best-window 30 --patience 0` (no early stopping — let them run full duration).

**Files:** `scripts/launch_tournament_phase2.sh`, outputs in `outputs/tournament/phase2/`
**Acceptance check:** All 8 summary JSONs exist with `epochs_logged` equal to target. No NaN in final energy.
**Risk:** CG-SR and Woodbury-SR runs are expensive. If they are slower than estimated (>30s/epoch), they may not finish in 24h. Mitigation: reduce epochs in real-time if needed.

### Step 2.2 — Heavy-VMC evaluation of tournament results

**What:** Run `eval_checkpoint_matrix.py` with 100k samples on all 16 checkpoints (8 final + 8 best). Include Phase 4 baselines for direct comparison.

**Files:** `scripts/eval_tournament_phase2.sh`
**Acceptance check:** A comparison table like:

| Tag | ω | Optimizer | Loss | err (%) | ± SE (%) | vs p4 baseline |
|---|---:|---|---|---:|---:|---|
| t2_df_fd_w01 | 0.1 | DiagFisher | FD | ? | ? | Δ vs +0.076% |
| ... | ... | ... | ... | ... | ... | ... |

At least one combination shows statistically significant improvement over the Phase 4 baseline (reduction > 2×SE).

**Risk:** If no combination improves, see Phase 2 failure protocol below.

### Step 2.3 — Declare tournament winner(s)

**What:** Rank all runs by heavy-VMC error. For the top 2, also check the training curve shape (monotone vs noisy) and per-epoch cost. Select:
- **ω=0.1 winner:** The optimizer/loss with lowest 100k heavy-VMC error
- **ω=0.001 winner:** Same criterion
- If CG-SR wins but is 3× slower than DiagFisher and only 10% better → prefer DiagFisher for Phase 3+ (practical tradeoff)

**Files:** Analysis in `outputs/tournament/phase2/tournament_results.md`
**Acceptance check:** Winner declared with evidence, including training curve plots if energy trajectory matters.
**Risk:** None — this is an analysis step.

### Phase 2 failure protocol

If no Phase 2 run improves meaningfully over Phase 4 baselines:
1. Check whether energy trajectories are **flat** (capacity limit reached) or **noisy** (optimization unstable)
2. If flat: the BF+Jastrow at 49K params has reached its capacity at ω=0.1. This is a thesis finding. Report the achievable bound. **Do not continue to Phase 3 — go directly to Phase 5 with current best for the consistency matrix.**
3. If noisy (SR step size too large/unstable): retry with:
   - Lower LR (5e-5 instead of 2e-4)
   - Higher damping (0.05 instead of 0.01)
   - Longer warmup (100 epochs instead of 30)
   Track as Phase 2B (one more attempt).
4. If the from-scratch CG-SR run (GPU 7) works spectacularly but warm-start runs don't: the optimization basin matters — try CG-SR from scratch for more regimes.

---

## Phase 3 — Extension, Bridge, and Gradient Accumulation (24-36h)

**Goal:** Push tournament winners further. Bridge to ω=0.01. Optionally implement and test gradient accumulation.

**Depends on:** Phase 2 complete with at least one winner identified.

### Step 3.1 — Extend top performers (4000 more epochs)

**What:** Take the top 2 checkpoints from Phase 2 at each omega and continue training with reduced LR (cosine from current LR to 0.005× base with no warmup).

| GPU | Tag | ω | Src checkpoint | Epochs | Key change |
|---:|---|---:|---|---:|---|
| 0 | t3_ext_w01_a | 0.1 | Phase 2 winner #1 | 4000 | LR = 0.5× Phase 2 base |
| 1 | t3_ext_w01_b | 0.1 | Phase 2 winner #2 | 4000 | LR = 0.5× Phase 2 base |
| 2 | t3_ext_w001_a | 0.001 | Phase 2 winner #1 | 4000 | LR = 0.5× Phase 2 base |
| 3 | t3_ext_w001_b | 0.001 | Phase 2 winner #2 | 4000 | LR = 0.5× Phase 2 base |

**Files:** `scripts/launch_tournament_phase3_extend.sh`
**Acceptance check:** All 4 runs complete. Check if extended training improves heavy-VMC further.
**Risk:** Diminishing returns. If Phase 2 winner is already flat, extensions won't help.

### Step 3.2 — Bridge to ω=0.01

**What:** Transfer the ω=0.1 winner checkpoint to ω=0.01 using the winning optimizer. This bridges the gap between ω=0.1 (FD-colloc preferred) and ω=0.001 (REINFORCE preferred).

| GPU | Tag | ω | From | Epochs | Recipe |
|---:|---|---:|---|---:|---|
| 4 | t3_bridge_w01_to_w001_fd | 0.01 | ω=0.1 winner | 3000 | FD-colloc + winning optimizer |
| 5 | t3_bridge_w01_to_w001_re | 0.01 | ω=0.1 winner | 3000 | REINFORCE + winning optimizer |
| 6 | t3_bridge_w001_to_w01_up | 0.01 | ω=0.001 winner | 3000 | REINFORCE + winning optimizer |

Test bridging ω=0.01 from both directions (ω=0.1 down, ω=0.001 up) to see which transfer works better.

**Files:** `scripts/launch_tournament_phase3_bridge.sh`
**Acceptance check:** Three ω=0.01 checkpoints with heavy-VMC evaluation. Best should beat the existing +0.193%.
**Risk:** ω=0.01 may need its own recipe tuning (FD step size, oversample level).

### Step 3.3 — (Optional) Implement gradient accumulation

**What:** If Phase 2 analysis shows that gradient variance is the limiting factor (high variance in training energy, ESS fluctuations), implement K-fold gradient accumulation: resample K independent batches per optimizer step, accumulate gradients, then step once.

**Code change in `src/run_weak_form.py`:** Add `--grad-accumulate K` CLI argument. Wrap the existing micro-batch loop in an outer loop of K iterations, each with fresh samples from `importance_resample`. Divide loss by K.

```python
# Current code (simplified):
X = importance_resample(...)
for xb in microbatches(X):
    loss = compute_loss(xb)
    (loss / n_microbatch).backward()
optimizer.step()

# With gradient accumulation:
for k in range(grad_accumulate):
    X = importance_resample(...)  # fresh samples each time
    for xb in microbatches(X):
        loss = compute_loss(xb)
        (loss / (n_microbatch * grad_accumulate)).backward()
optimizer.step()
```

**Acceptance check:** `python3 src/run_weak_form.py --tag test_accum --mode bf --n-elec 6 --omega 1.0 --epochs 5 --grad-accumulate 4 --n-coll 1024` runs without error and produces expected number of epoch entries.
**Risk:** Low — this is a standard technique. The only subtlety is ensuring the importance resampling happens inside the accumulation loop.

**GPU 7 (if available):** Quick test of gradient accumulation with winning optimizer at ω=0.1.

---

## Phase 4 — Multi-Seed Consolidation at N=6 (24-36h)

**Goal:** Run the winning recipe across the full ω grid with 3 seeds to establish consistency.

**Depends on:** Phase 2 or Phase 3 produced a recipe that demonstrably beats Phase 4 baselines.

### Step 4.1 — Define consolidated recipe per regime

**What:** Based on Phase 2-3 results, select one recipe per omega band:

| ω band | Optimizer | Loss | Epochs | n-coll | Special |
|---:|---|---|---:|---:|---|
| 1.0 | Phase 2 winner (or historical best recipe) | REINFORCE | 2000 | 4096 | Easy regime; verify-only |
| 0.5 | Same as ω=1.0 | REINFORCE | 2000 | 4096 | Easy regime; verify-only |
| 0.1 | Phase 2 winner | Phase 2 winner loss | Phase 2 epoch count | Phase 2 n-coll | Main target |
| 0.01 | Phase 3 bridge winner | Phase 3 winner | Phase 3 epoch count | TBD | Bridged regime |
| 0.001 | Phase 2 winner | REINFORCE, no-gate | Phase 2 epoch count | 4096, oversample=32 | Hard regime |

**Files:** Recipe specification in `scripts/tournament_recipes.json`
**Acceptance check:** One clearly specified CLI command per (ω, seed) combination.
**Risk:** If recipes differ too much across ω, the "consistency" story is weaker.

### Step 4.2 — Launch 15-run multi-seed campaign

**What:** 15 runs: 5 ω × 3 seeds {42, 11, 77}. Schedule in 2-3 waves across 8 GPUs.

Wave 1 (8 GPUs): {ω=0.1, ω=0.001, ω=0.01} × 3 seeds = 9 runs → fits in 8 GPUs (queue last one)
Wave 2 (6 GPUs): {ω=0.5, ω=1.0} × 3 seeds = 6 runs

Easy regimes (ω≥0.5) use shorter runs (2000 epochs) and finish faster.

**Files:** `scripts/launch_tournament_phase4_multiseed.sh`
**Acceptance check:** All 15 summary JSONs exist.
**Risk:** 15 runs × ~12h average = ~22h wall with 8 GPUs in 2 waves.

### Step 4.3 — Heavy-VMC evaluation and N=6 evidence bundle

**What:** 100k heavy-VMC on all 30 checkpoints (15 final + 15 best). Compute per-ω statistics.

**Acceptance check:** Summary table:

| ω | Seed mean err (%) | Seed std (%) | Best single (%) | All 3 below X%? |
|---:|---:|---:|---:|---|
| 1.0 | ? | ? | ? | ? |
| 0.5 | ? | ? | ? | ? |
| 0.1 | ? | ? | ? | ? |
| 0.01 | ? | ? | ? | ? |
| 0.001 | ? | ? | ? | ? |

Archive as `outputs/tournament/phase4/n6_evidence_bundle.json`.
**Risk:** High seed-to-seed variance at hard omegas would indicate optimization instability even with the new recipe. This itself is a meaningful finding.

---

## Phase 5 — Scale to N=12 (24-36h)

**Goal:** Transfer the winning N=6 recipe to N=12.

**Depends on:** Phase 4 complete. At least ω ∈ {0.1, 0.5, 1.0} have consistent multi-seed results at N=6.

### Step 5.1 — N=12 recipe adaptation

**What:** Adapt winning recipe for N=12:
- Warm-start from existing N=12 checkpoints (`long_n12w1.pt` at +0.018%, `long_n12w05.pt` at +0.028%)
- Reduce n-coll if OOM at N=12 (4096 → 2048 with BF)
- Use same optimizer as N=6 winner
- Test memory fit: `python3 src/run_weak_form.py --mode bf --n-elec 12 --omega 1.0 --n-coll 4096 --epochs 1 --natural-grad --sr-mode <winner>`

**Files:** `scripts/launch_tournament_phase5_n12.sh`
**Acceptance check:** Memory fit confirmed (<10GB). One recipe spec per (N=12, ω).
**Risk:** N=12 backflow with n-coll=4096 + SR storage may OOM. Fallback: n-coll=2048.

### Step 5.2 — N=12 campaign (ω ∈ {0.1, 0.5, 1.0} × 2 seeds)

**What:** 6 runs. Compare with historical bests.
**Files:** Same launcher
**Acceptance check:** All 6 summary JSONs. Heavy-VMC compared to historical best (+0.018% at ω=1.0, +0.028% at ω=0.5, +0.122% at ω=0.1).
**Risk:** N=12 runs are 2-3× slower. Budget ~16-20h per run.

---

## Phase 6 — Final Consolidation (4-6h)

**Goal:** Compile final matrix, commit, tag.

### Step 6.1 — Compile full evidence matrix

**What:** Merge N=6 and N=12 results into the final table:

| N | ω | Mean err (%) | Std (%) | Best (%) | Seeds | Status |
|---:|---:|---:|---:|---:|---:|---|

### Step 6.2 — Session close and archival

**What:** Git tag `result/2026-04-XX-optimizer-tournament-final`. Update SESSION_LOG.md, JOURNAL.md, DECISIONS.md.

---

## Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| SR unstable at ω≤0.1 post-bugfix | Low (hypothesis says this was the bug's fault) | High (plan pivot needed) | Phase 1 smoke tests. Higher damping + warmup as backup. |
| CG-SR too slow per epoch | Medium | Medium (reduces epoch budget) | Use DiagFisher as primary (nearly free per epoch). CG-SR as premium only. |
| `--resume` restores Adam states into SGD optimizer | Medium | Low (fixable) | Test in Phase 1. Use `--init-jas`/`--init-bf` workaround if needed. |
| All optimizers hit the same plateau as Adam | Low (would prove architecture limit) | High but scientifically valuable | This IS the question the tournament answers. A confirmed capacity limit is still a thesis finding. |
| GPU contention | Low (all 8 free now) | Medium (delays) | All runs in tmux. Can reduce to 7 GPUs if GPU 0 needed. |
| FD-colloc + natural gradient don't compose in code | Low | High (blocks 4 of 8 runs) | Test explicitly in Phase 1 smoke. |

## Success criteria

**Minimum viable outcome (1 week):**
- Tournament completed: 8+ runs at N=6 with systematic optimizer comparison
- At least one optimizer shows statistically significant improvement over Adam baseline at ω=0.1
- Published comparison table with honest assessment

**Target outcome (2 weeks):**
- N=6 ω=0.1 pushed below +0.03% (3× improvement over +0.076%)
- N=6 ω=0.001 pushed below +0.06% (2× improvement over +0.119%)
- Multi-seed consistency at N=6 for ω≥0.1 (3 seeds, std < mean error)
- N=12 results with winning recipe

**Stretch outcome:**
- Any (N=6, ω) point with ≤0.01% at 3/3 seeds
- CG-SR from scratch converges to ≤0.01% at ω=0.1 in one run (conclusive optimization proof)

## Current State

**Active phase:** Phase 2 (preparing)
**Active step:** Step 2.1
**Last evidence:** Phase 1 smoke tests complete (2026-04-05). Full results below.

### Phase 1 Results (100 epochs, seed=42, 30k final VMC)

| Tag | Optimizer | Loss | ω | VMC err% | Δ vs base | s/ep | Gate |
|---|---|---|---|---|---|---|---|
| smoke_cg_fd_w01 | CG-SR | FD-colloc | 0.1 | +0.084% | +0.008pp | 15.8 | **PASS** |
| smoke_wb_fd_w01 | Woodbury | FD-colloc | 0.1 | +0.097% | +0.021pp | 16.1 | FAIL |
| smoke_df_fd_w01 | DiagFisher | FD-colloc | 0.1 | +0.114% | +0.038pp | 9.9 | FAIL |
| smoke_df_re_w01 | DiagFisher | REINFORCE | 0.1 | +0.079% | +0.003pp | 4.3 | **PASS** |
| smoke_df_re_w001 | DiagFisher | REINFORCE | 0.001 | +0.123% | +0.004pp | 3.4 | **PASS** |
| smoke_cg_re_w001 | CG-SR | REINFORCE | 0.001 | +0.159% | +0.040pp | 15.1 | FAIL |

Baselines: ω=0.1 p4m1 +0.076% (100k VMC), ω=0.001 p4m2 +0.119% (100k VMC)

**Key finding:** CG-SR + FD-colloc at ω=0.1 reached training-time E as low as 3.5528 (err=-0.030%) at epoch 60 — well below Adam baseline. But it oscillates and the VMC evaluation catches it at +0.084%. This means the optimizer CAN find better landscape regions but needs convergence control (lower LR, warmup, tighter trust region).

**Phase 2 modifications from plan:**
- Drop Woodbury ω=0.1 (redundant with CG-SR, slightly worse)
- Drop DiagFisher + FD ω=0.1 (diagonal approx insufficient for FD loss)
- Drop CG-SR + REINFORCE ω=0.001 (ESS=2-6, Fisher estimate too noisy for full SR)
- Add CG-SR + FD ω=0.1 with lower LR and warmup (address oscillation)
- Keep: CG-SR+FD ω=0.1, DiagFisher+REINF ω=0.1, DiagFisher+REINF ω=0.001, Adam baseline, from-scratch CG-SR

**Foundation checks updated:**
- [x] Natural gradient + FD-colloc compose correctly (confirmed: preconditioner runs after backward())
- [x] Optimizer state handling on resume verified (confirmed: optimizer state never restored from checkpoint)
- [x] patience=0 correctly disables early stopping (confirmed)

**Current risk:** CG-SR oscillation at ω=0.1. Need lower LR + warmup for Phase 2 to see convergence.
**Next action:** Create Phase 2 launcher, launch 7-8 runs on GPUs
**Blockers:** None. All 8 GPUs are free.
