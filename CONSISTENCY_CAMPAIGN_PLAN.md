# Consistency Campaign Plan

> **Goal:** Consistently reach <=0.01% of DMC across high N and both high AND low omega.
> **Resources:** 8× RTX 2080 Ti (11 GB each), multi-day budget.
> **Date:** 2026-03-28

---

## Current State of the Art

Best achieved errors from curated inventory + COLLOCATION_BEST_RESULTS:

| N | ω | Best err (%) | Best method | Status |
|--:|----:|---:|---|---|
| 2 | 1.0 | 0.004 | BF+Adam | ✅ consistent |
| 2 | 0.5 | 0.004 | BF+Fisher | ✅ consistent |
| 2 | 0.1 | 0.004 | BF+Adam | ✅ consistent |
| 2 | 0.01 | 0.032 | BF+Adam | ⚠️ close but not <0.01 |
| 6 | 1.0 | 0.002 | SR probe | ✅ achieved once |
| 6 | 0.5 | 0.002 | SR | ✅ achieved once |
| 6 | 0.1 | 0.091 | Adam transfer | ❌ far from target |
| 6 | 0.01 | 0.193 | Adam probe | ❌ far from target |
| 6 | 0.001 | 0.334 | Adam probe | ❌ far from target |
| 12 | 1.0 | 0.009 | Adam continue | ✅ achieved once |
| 12 | 0.5 | 0.024 | Fisher cascade | ⚠️ close |
| 12 | 0.1 | 0.122 | Adam transfer | ❌ far |
| 20 | 1.0 | 32.9 | smoke only | ❌ not seriously attempted |
| 20 | 0.1 | 5.5 | Jastrow-only | ❌ limited arch |

**Key observation:** <=0.01% has only been achieved at omega >= 0.5 and N <= 12. The entire low-omega regime (omega <= 0.1) is an order of magnitude away from target. N=20 has not been seriously pushed.

---

## Root Cause Hypotheses (Ordered by Diagnostic Layer)

### H1 — Foundation: DMC reference table is incomplete for hard regimes
**Layer 1 (Data).** Confirmed bug.

`_snap_omega` + `_lookup_dmc_energy` silently fail for:
- N=12, omega=0.001 → KeyError (falls to NaN)
- N=12, omega=0.005 → snaps to 0.001 → KeyError (falls to NaN)
- N=20, omega=0.001 → KeyError
- N=20, omega=0.01 → KeyError
- N=20, omega=0.05 → snaps to 0.01 → KeyError

With NaN reference, the trainer cannot:
- Select checkpoints by percent error
- Apply error-based rollback
- Report meaningful progress

Every prior low-omega high-N campaign was blind to its own quality metric.

### H2 — Foundation: snap_omega silently maps unsupported omega to wrong reference
**Layer 1 (Data).** Partial risk.

If someone requests omega=0.005 and the snap maps to 0.001 (which exists for N=6 but not N=12), the error comparison would use a completely wrong DMC energy for N=6. For N=12 it fails outright, but a future table addition could silently produce the wrong comparison.

### H3 — Implementation: adaptive sigma only activates when default is unchanged
**Layer 2 (Implementation).** Code-confirmed risk.

`adapt_sigma_fs()` in `run_weak_form.py:194` checks `sigma_fs_default != (0.8, 1.3, 2.0)` and returns early if user overrode sigma. This is correct in isolation, but:
- Campaign scripts that explicitly pass `--sigma-fs=0.8,1.3,2.0` (same as default) still bypass adaptation because the tuple comparison is by value, so that's fine.
- But the comment at line 1308 says "no auto-widening needed" which contradicts the runtime adaptation at line 776. Operator confusion risk.

### H4 — Training: ESS gating creates skip loops that prevent all progress in hard regimes
**Layer 4 (Training setup).** Empirically confirmed.

When `min_ess > 0` and the proposal has poor overlap with |Ψ|², every epoch is skipped and reverted. The model never takes an accepted step. This is a deadlock:
- Model needs updates to improve → updates require good ESS → good ESS requires model to already be good in this regime.

The adaptive-oversample mechanism tries to break this by widening the candidate pool, but at N=12/20 low omega, even maximum oversample may not produce sufficient ESS.

### H5 — Training: REINFORCE reward signal degrades at low omega
**Layer 4 (Training setup).** Hypothesized, not yet confirmed.

At low omega:
- Energy scale shrinks (E_DMC ∝ ω for ω→0 in the non-interacting limit)
- E_L variance relative to E_DMC may stay large or grow
- REINFORCE reward = E_L - baseline; if variance >> |E_DMC|, the signal-to-noise ratio collapses
- Adam's adaptive moments partially help, but the fundamental SNR problem remains

This would explain why:
- omega=0.01 can be made to work with patience (energy scale still ~0.07 for N=2, ~0.69 for N=6)
- omega=0.001 fails (energy scale ~0.007 for N=2, ~0.14 for N=6 — gradient noise swamps signal)

### H6 — Training: SR amplifies importance-weight noise at low omega
**Layer 4 (Training setup).** Empirically supported by user observation.

SR/natural gradient works by inverting the Fisher information matrix to precondition gradients. At low omega:
- Importance weights are more skewed (proposal-target mismatch wider)
- Fisher estimate inherits this skew since it's computed on resampled collocation points
- Badly conditioned Fisher → large update directions → instability

This is consistent with the observation that "SR works well for high omega but not low omega." The importance sampling fix (March 24) improved things but didn't eliminate this because:
- Even with correct mixture density, the proposal-target overlap is fundamentally worse at low omega
- Fisher estimation is fundamentally noisier with fewer effective samples

**Decision: SR for omega >= 0.1 only.** Adam for omega < 0.1. This is not a workaround — it's the correct tool selection given the noise profile.

### H7 — Training: loss landscape changes qualitatively at low omega
**Layer 3/4 boundary.** Hypothesized.

For omega → 0:
- Harmonic trap becomes flat → wavefunction spreads widely
- Coulomb interaction (kappa/r) dominates over confinement
- Kinetic energy (½|∇logΨ|²) becomes the primary nontrivial term
- The loss surface for a Jastrow/BF network changes from "fit a localized bump" to "fit extended correlations over a large volume"

This may require either:
- More collocation points (larger spatial coverage needed)
- Different sampling strategy (current sigma_fs may still be too narrow even after adaptation)
- Different architecture emphasis (Jastrow needs to model longer-range correlation)

### H8 — Sampling: oversample multiplier is GPU-memory-limited at high N
**Layer 2/4 boundary.** Practical constraint.

At N=20, d=2, with oversample=8 and n_coll=4096: candidate pool = 32,768 points of shape (20,2). Each needs forward pass through the network. Memory pressure limits effective oversample, which limits ESS, which limits training progress.

### H9 — Checkpoint continuations introduce drift
**Layer 4.** Hypothesized.

The best results at high omega come from specific checkpoint chains (e.g., the N=6 bf_hardfocus chain). These are not reproducible from scratch — the continuation path matters. For low omega, the "right" continuation path is not known, and blind transfer from high omega may put parameters in a poor basin.

---

## Campaign Structure

### Phase 0: Foundation Fixes [~2 hours, no GPU needed]

All code changes before any training run. These are prerequisites.

#### 0A — Fix DMC reference table and safeguard snap_omega

**What:** Add missing DMC entries where known. For entries where DMC is genuinely unknown, add explicit guards that error loudly rather than snapping to a wrong omega.

Concrete changes:
1. In `src/config.py`, update `DMC_ENERGIES`:
   - N=12: add `0.001` if a reference exists (literature or extrapolation). If not, leave absent but ensure `_lookup_dmc_energy` raises clearly.
   - N=20: add `0.01` if reference exists. Same policy.
2. In `_snap_omega`: add a tolerance check. If `|omega - snapped_omega| / omega > 0.5`, raise instead of silently snapping. This prevents 0.005 snapping to 0.001.
3. In `_lookup_dmc_energy`: when `allow_missing=True` returns NaN, print a WARNING line that shows in logs.

**Verify:** Run the same test as above; all target regimes should either have a reference or fail loudly with an actionable message.

#### 0B — Fix contradictory code comment

**What:** Remove the misleading comment at `run_weak_form.py:1308` ("no auto-widening needed") since adaptive widening IS active at line 776.

**Verify:** Grep for the old comment, confirm it's gone.

#### 0C — Create universal diagnostic evaluation script

**What:** Create `scripts/eval_checkpoint_matrix.py` that:
- Takes a checkpoint path and (N, omega) pair
- Loads model from checkpoint
- Runs heavy VMC evaluation (30k samples, burn_in=800, thin=5, sampler_steps=120)
- Reports: E_mean, E_std, err_pct vs DMC, ESS from importance resampling, min-pair statistics
- Returns structured JSON

This separates "evaluation" from "training" — we never again rely on in-training probes for final claims.

**Verify:** Run on one known-good checkpoint (e.g., `adam_control_v1.pt` at N=6 omega=1.0) and confirm <0.01%.

#### 0D — Create instrumented training wrapper

**What:** Create `scripts/instrumented_run.py` that wraps `train_weak_form` with epoch-level logging of:
- ESS (raw and effective)
- Accepted vs rejected epochs (with reason)
- Gradient norm per parameter group
- E_L mean, std, and outlier fraction (|resid| > 3*MAD)
- Learning rate at each epoch
- Wall time per epoch

Output: JSON lines file, one object per epoch. This is the diagnostic instrument for all subsequent phases.

**Verify:** Short 20-epoch smoke test at N=6 omega=1.0, inspect JSON output.

---

### Phase 1: Reproduce Known-Good Regimes [Day 1, 8 GPUs, ~12 hours]

**Purpose:** Confirm the codebase can still consistently produce <=0.01% results at regimes where we've done it before. If this fails, something is broken and we stop.

#### GPU allocation (8 GPUs, 3 seeds each where possible):

| GPU | Job | N | ω | Seed | Resume from | Optimizer | Expected |
|---|---|---|---|---|---|---|---|
| 0 | repro_n6w1_s42 | 6 | 1.0 | 42 | bf_ctnn_vcycle.pt | Adam | <0.01% |
| 1 | repro_n6w1_s11 | 6 | 1.0 | 11 | bf_ctnn_vcycle.pt | Adam | <0.01% |
| 2 | repro_n6w1_s77 | 6 | 1.0 | 77 | bf_ctnn_vcycle.pt | Adam | <0.01% |
| 3 | repro_n12w1_s42 | 12 | 1.0 | 42 | w1_n12w1_xfer.pt | Adam | <0.01% |
| 4 | repro_n12w1_s11 | 12 | 1.0 | 11 | w1_n12w1_xfer.pt | Adam | <0.01% |
| 5 | repro_n6w05_s42 | 6 | 0.5 | 42 | w1_n6w05_hisamp.pt | Adam | <0.01% |
| 6 | repro_n6w05_s11 | 6 | 0.5 | 11 | w1_n6w05_hisamp.pt | Adam | <0.01% |
| 7 | repro_n12w05_s42 | 12 | 0.5 | 42 | n12w05_cascade.pt | Adam | <0.02% |

**Settings:** 800 epochs, n_coll=4096, lr=5e-4, oversample=8, vmc_every=50.

**Pass criteria:** At least 2/3 seeds per (N, omega) reach <0.01% on heavy VMC eval.
**Fail criteria:** If >=2 seeds per regime fail, investigate before proceeding.

**After Phase 1:** Run `eval_checkpoint_matrix.py` on all 8 outputs. Record results in `outputs/consistency_campaign/phase1_results.json`.

---

### Phase 2: Characterize Failure Modes at Low Omega [Day 1-2, 8 GPUs, ~24 hours]

**Purpose:** Understand WHY low omega fails by running instrumented diagnostic experiments that test specific hypotheses. Each GPU runs one targeted test.

#### 2A — ESS/overlap characterization (GPUs 0-1)

**Theory tested:** H4, H8 — ESS is too low at low omega for training to proceed.

**Method:** Run `instrumented_run.py` with `min_ess=0` (no gating, all epochs accepted) and log ESS. Compare:

| GPU | Job | N | ω | oversample | sigma_fs | Notes |
|---|---|---|---|---|---|---|
| 0 | diag_ess_n6w01 | 6 | 0.1 | 16 | adaptive | No ESS gate |
| 1 | diag_ess_n6w001 | 6 | 0.001 | 16 | adaptive | No ESS gate |

**What we learn:**
- What ESS values look like epoch-by-epoch at low omega WITHOUT gating
- Whether training progresses at all when ESS is low
- Whether the model self-corrects (ESS improves as model improves)

**Key metric:** Does energy decrease monotonically even with low ESS? If yes, gating was preventing progress. If no, low ESS genuinely corrupts gradients.

#### 2B — REINFORCE signal-to-noise at low omega (GPUs 2-3)

**Theory tested:** H5 — gradient SNR degrades faster than energy scale shrinks.

**Method:** Run training and log per-epoch:
- E_L mean and std
- |E_L std| / |E_L mean| (coefficient of variation)
- Gradient norm per param group
- Actual energy improvement per epoch

| GPU | Job | N | ω | lr | Notes |
|---|---|---|---|---|---|
| 2 | diag_snr_n6w01 | 6 | 0.1 | 5e-4 | Standard LR |
| 3 | diag_snr_n6w001 | 6 | 0.001 | 1e-4 | Lower LR for small energy scale |

**What we learn:**
- Whether CV(E_L) explodes at low omega
- Whether gradient norm is dominated by noise vs. signal
- What the effective gradient SNR is

**Key metric:** Plot CV(E_L) vs. epoch. If CV >> 10 at low omega vs. CV ~ 1-3 at high omega, REINFORCE reward noise is the dominant blocker.

#### 2C — FD-collocation vs. REINFORCE at low omega (GPUs 4-5)

**Theory tested:** H5 (alternative) — FD-collocation loss may be more robust because it doesn't use E_L as a reward signal.

**Method:** Run identical setups with different loss types:

| GPU | Job | N | ω | loss_type | lr | Notes |
|---|---|---|---|---|---|---|
| 4 | diag_fdcolloc_n6w01 | 6 | 0.1 | fd-colloc | 2e-4 | FD-Laplacian collocation |
| 5 | diag_reinf_n6w01 | 6 | 0.1 | reinforce | 5e-4 | Standard REINFORCE |

Both resume from the same checkpoint (`w1_n6w05_hisamp.pt` transferred to omega=0.1).

**What we learn:**
- Whether FD-colloc is more stable at low omega
- Whether the loss landscape itself is the problem vs. the REINFORCE estimator

**Key metric:** Compare final error after 500 epochs. If FD-colloc is significantly better, the REINFORCE reward noise hypothesis is confirmed.

#### 2D — From-scratch vs. transfer at low omega (GPUs 6-7)

**Theory tested:** H9 — high-omega checkpoints may put network in a bad basin for low omega.

**Method:**

| GPU | Job | N | ω | Resume | lr | epochs | Notes |
|---|---|---|---|---|---|---|---|
| 6 | diag_scratch_n6w01 | 6 | 0.1 | none | 5e-4 | 2000 | From random init |
| 7 | diag_xfer_n6w01 | 6 | 0.1 | bf_ctnn_vcycle.pt | 5e-4 | 800 | Transfer from ω=1.0 |

**What we learn:**
- Whether transfer helps or hurts
- Whether from-scratch can eventually match or beat transfer
- What the asymptotic error is without transfer bias

**Key metric:** Compare best-achieved error. If from-scratch matches transfer within 2x epochs, transfer initialization is not critical and may be harmful.

---

### Phase 3: Targeted Fix Experiments [Day 2-4, 8 GPUs, ~48 hours]

**Purpose:** Based on Phase 2 diagnostics, run targeted fixes. All experiments below are independent and can run in parallel. Execute in priority order based on Phase 2 findings.

#### 3A — No-gate + aggressive oversample at low omega

**Theory:** H4 fix — remove the ESS gate entirely, compensate with massive oversampling.

| GPU | Job | N | ω | min_ess | oversample | ess_floor | epochs | lr |
|---|---|---|---|---|---|---|---|---|
| 0 | fix_nogate_n6w01_s42 | 6 | 0.1 | 0 | 32 | 0 | 1500 | 3e-4 |
| 1 | fix_nogate_n6w001_s42 | 6 | 0.001 | 0 | 32 | 0 | 2000 | 1e-4 |

**Settings:** Adam, adaptive sigma, no rollback gating, `clip_el=3.0`, `reward_qtrim=0.01`.

**Pass criterion:** Error decreases monotonically for >=100 consecutive epochs. Final error < 0.1%.

#### 3B — FD-collocation + proximal at low omega

**Theory:** H5 fix — bypass REINFORCE entirely. Use FD-Laplacian collocation with proximal penalty to prevent mode collapse.

| GPU | Job | N | ω | loss_type | fd_h | prox_mu | huber_delta | lr | epochs |
|---|---|---|---|---|---|---|---|---|---|
| 2 | fix_fdcol_n6w01 | 6 | 0.1 | fd-colloc | 0.005 | 0.1 | 0.5 | 2e-4 | 1500 |
| 3 | fix_fdcol_n6w001 | 6 | 0.001 | fd-colloc | 0.002 | 0.1 | 0.2 | 1e-4 | 2000 |

**Rationale for fd_h scaling:** At low omega, wavefunction features are broader → can use smaller h for better accuracy. Huber delta is proportional to energy scale.

**Pass criterion:** Stable training (no divergence) and error < 0.1%.

#### 3C — Ultra-patient Adam with learning rate warmup

**Theory:** H5/H7 fix — at low omega, the optimization landscape is harder. Use much smaller LR and much more epochs.

| GPU | Job | N | ω | lr | epochs | lr_min_frac | n_coll | oversample |
|---|---|---|---|---|---|---|---|---|
| 4 | fix_patient_n6w01 | 6 | 0.1 | 1e-4 | 3000 | 0.1 | 8192 | 16 |
| 5 | fix_patient_n6w001 | 6 | 0.001 | 5e-5 | 5000 | 0.1 | 8192 | 24 |

**Extra:** Double collocation points (8192 vs 4096) to get more spatial coverage in the broader wavefunction.

**Pass criterion:** Error < 0.05% at omega=0.1, error < 0.2% at omega=0.001 (improvement over current best).

#### 3D — Energy-scale-normalized REINFORCE

**Theory:** H5 fix (novel) — the REINFORCE reward E_L has different absolute scales at different omega. Normalizing by E_ref (or by running std) should equalize gradient signal.

This requires a small code modification: in `rayleigh_hybrid_loss`, divide (E_L - baseline) by a running estimate of std(E_L) or by |E_ref| before using it as the REINFORCE reward.

| GPU | Job | N | ω | Notes |
|---|---|---|---|---|
| 6 | fix_normrew_n6w01 | 6 | 0.1 | Normalized REINFORCE reward |
| 7 | fix_normrew_n6w001 | 6 | 0.001 | Normalized REINFORCE reward |

**Code change required:** Add `--reward-normalize` flag to `train_weak_form` that divides REINFORCE advantage by `max(std(E_L), epsilon)` before computing policy gradient.

**Pass criterion:** Gradient norm stability (std(grad_norm) / mean(grad_norm) < 2 over a 50-epoch window).

---

### Phase 4: Scale Fixes to High N [Day 3-5, 8 GPUs, ~48 hours]

**Purpose:** Take whatever worked in Phase 3 for N=6 and push it to N=12 and N=20.

#### 4A — N=12 at omega=0.1 (first serious push)

Apply the winning low-omega recipe from Phase 3 to N=12.

| GPU | Job | N | ω | Seed | Resume | Notes |
|---|---|---|---|---|---|---|
| 0 | scale_n12w01_s42 | 12 | 0.1 | 42 | n12w05_cascade.pt | Transfer from ω=0.5 |
| 1 | scale_n12w01_s11 | 12 | 0.1 | 11 | n12w05_cascade.pt | Transfer from ω=0.5 |
| 2 | scale_n12w01_scratch | 12 | 0.1 | 42 | none | From scratch, 3000 ep |

**Target:** Error < 0.05% (improvement from current 0.122%).

#### 4B — N=12 at omega=0.5 + 1.0 (consistency push)

Push existing near-misses to consistent <0.01%.

| GPU | Job | N | ω | Seed | Resume | Notes |
|---|---|---|---|---|---|---|
| 3 | scale_n12w05_s42 | 12 | 0.5 | 42 | n12w05_cascade.pt | Polish |
| 4 | scale_n12w05_s11 | 12 | 0.5 | 11 | n12w05_cascade.pt | Polish |
| 5 | scale_n12w1_s42 | 12 | 1.0 | 42 | v7_n12w1_continue.pt | Polish |

**Target:** All 3 seeds <0.01%.

#### 4C — N=20 at omega=1.0 (first serious push with BF)

N=20 has only been attempted with Jastrow-only and tiny smoke runs. Now try properly.

| GPU | Job | N | ω | Seed | mode | bf_hidden | n_coll | micro_batch | epochs |
|---|---|---|---|---|---|---|---|---|---|
| 6 | scale_n20w1_bf | 20 | 1.0 | 42 | bf | 64 | 2048 | 128 | 2000 |
| 7 | scale_n20w1_jas | 20 | 1.0 | 42 | jastrow | - | 4096 | 256 | 2000 |

**Notes:** BF at N=20 needs reduced architecture (bf_hidden=64, bf_layers=2) to fit 11 GB. Compare BF vs. Jastrow-only to establish whether BF helps at this N.

**Target:** Error < 1% (massive improvement from current 32%).

---

### Phase 5: Low-Omega at High N [Day 4-6, 8 GPUs]

**Purpose:** Only attempt this after Phases 3-4 establish the right recipe.

#### 5A — N=12 at omega=0.01 (with reference verification)

**Prerequisite:** DMC reference for N=12 omega=0.01 must be verified. Currently listed as 2.0 in config table — this looks like a placeholder (suspiciously round). Verify against literature or compute independently before trusting.

If verified:
| GPU | Job | N | ω | Method | Notes |
|---|---|---|---|---|---|
| 0 | deep_n12w001_a | 12 | 0.01 | Phase 3 winner | From best N=12 ω=0.1 checkpoint |
| 1 | deep_n12w001_b | 12 | 0.01 | Phase 3 winner | Different seed |

#### 5B — N=6 at omega=0.001 (precision push)

Current best is 0.334%. Push toward <0.05%.

| GPU | Job | N | ω | Method |
|---|---|---|---|---|
| 2 | deep_n6w0001_a | 6 | 0.001 | Phase 3 winner |
| 3 | deep_n6w0001_b | 6 | 0.001 | Phase 3 runner-up |
| 4 | deep_n6w0001_c | 6 | 0.001 | FD-colloc if not already winner |

#### 5C — N=20 at omega=0.1 (if N=20 omega=1.0 worked)

| GPU | Job | N | ω | Method |
|---|---|---|---|---|
| 5 | deep_n20w01_a | 20 | 0.1 | Transfer from best N=20 ω=1.0 |
| 6 | deep_n20w01_b | 20 | 0.1 | From scratch |

---

### Phase 6: Final Consistency Matrix [Day 6-7, 8 GPUs]

**Purpose:** Full multi-seed evaluation of the target claim.

For every (N, omega) pair in the target matrix, run 3 seeds using the established recipe and evaluate all with `eval_checkpoint_matrix.py`.

**Target matrix:**

| N | ω values to cover |
|---|---|
| 2 | 0.01, 0.1, 0.5, 1.0 |
| 6 | 0.01, 0.1, 0.5, 1.0 |
| 12 | 0.1, 0.5, 1.0 |
| 20 | 0.1, 1.0 |

omega=0.001 included only if Phase 5 shows feasibility.

**Consistency criterion:** For each (N, ω):
- All 3 seeds reach < 0.01% on heavy VMC eval (30k samples)
- Mean ± std of error across seeds reported

If some regimes hit <0.01% on 2/3 seeds but not 3/3, report separately as "achievable but not consistent."

---

## Specific Code Changes Required

### Change 1: snap_omega tolerance guard (Phase 0A)

In `src/config.py`, modify `_snap_omega`:
```python
def _snap_omega(omega: float) -> float:
    omega = float(omega)
    best = min(_SUPPORTED_OMEGAS, key=lambda w: abs(w - omega))
    # Guard: if the snap is more than 50% away, this omega is not in the table
    if abs(best - omega) / max(abs(omega), 1e-12) > 0.5:
        raise ValueError(
            f"omega={omega} is not close to any supported value "
            f"(nearest: {best}, >50% relative distance). "
            f"Supported: {_SUPPORTED_OMEGAS}"
        )
    return best
```

### Change 2: Remove contradictory comment (Phase 0B)

In `src/run_weak_form.py:1306-1308`, remove:
```python
    # Note: sigma_fs values are in oscillator-length units.
    # sample_gauss already scales by 1/sqrt(omega), so defaults (0.8,1.3,2.0)
    # are appropriate for all omega — no auto-widening needed.
```

Replace with:
```python
    # Note: sigma_fs values are in oscillator-length units.
    # sample_gauss already scales by 1/sqrt(omega).
    # Adaptive widening is applied at runtime in train_weak_form() via adapt_sigma_fs().
```

### Change 3: Reward normalization flag (Phase 3D)

In `src/functions/Neural_Networks.py`, modify `rayleigh_hybrid_loss` to accept `reward_normalize=False`. When True, divide advantage by `max(std(E_L), 1e-6)`.

In `src/run_weak_form.py`, wire through `--reward-normalize` CLI flag.

### Change 4: N=12 omega=0.01 reference verification

The value `2.0` in the config table for N=12 omega=0.01 looks like a placeholder. Before Phase 5A:
- Literature search for N=12 2D quantum dot at omega=0.01
- If not found, run a long VMC estimation from a well-converged N=12 checkpoint
- If uncertain, mark as unverified and do not use for error-based selection

---

## Decision Points (Pause and Assess)

### After Phase 1:
- If reproduction fails: STOP. Debug the regression before anything else.
- If reproduction succeeds: proceed to Phase 2.

### After Phase 2:
- If ESS analysis (2A) shows training progresses without gating → H4 confirmed, run 3A first in Phase 3.
- If SNR analysis (2B) shows CV(E_L) > 10 at low omega → H5 confirmed, run 3B/3D first in Phase 3.
- If FD-colloc (2C) dramatically outperforms REINFORCE → pivot to FD-colloc for all low-omega work.
- If from-scratch (2D) matches transfer → stop using transfer for low omega.

### After Phase 3:
- Rank all 4 approaches by best achieved error at omega=0.1.
- Use the winner as the default recipe for Phases 4-6.
- If no approach gets N=6 omega=0.1 below 0.05%, the problem may be architectural (H7) — consider more drastic changes.

### After Phase 4:
- If N=12 omega=0.5 and 1.0 are consistent at <0.01% → foundation is solid at high omega for high N.
- If N=20 remains >1% → current architecture may be capacity-limited at N=20. Consider Jastrow warm-start → BF fine-tune approach.

---

## Operational Notes

### tmux session naming
```
tmux new-session -s consistency_p1   # Phase 1
tmux new-session -s consistency_p2   # Phase 2
# etc.
```

### GPU assignment
All jobs use `CUDA_MANUAL_DEVICE=N` to pin GPU. Example:
```bash
CUDA_MANUAL_DEVICE=0 python src/run_weak_form.py --mode bf --n-elec 6 --omega 1.0 \
    --epochs 800 --n-coll 4096 --lr 5e-4 --seed 42 \
    --tag repro_n6w1_s42 --resume results/arch_colloc/bf_ctnn_vcycle.pt \
    > outputs/consistency_campaign/phase1/repro_n6w1_s42.log 2>&1 &
```

### Checkpoint naming convention
```
consistency_p<phase>_<tag>.pt
```

### Output directory
```
outputs/consistency_campaign/
  phase0/    # foundation fix smoke tests
  phase1/    # reproduction results
  phase2/    # diagnostic experiments
  phase3/    # fix experiments
  phase4/    # scaling experiments
  phase5/    # low-omega deep push
  phase6/    # final consistency matrix
```

### Evaluation protocol
Every final claim is backed by `eval_checkpoint_matrix.py` output, not by in-training probes.

### SR policy
- omega >= 0.1: SR (CG mode) is allowed and may be preferred.
- omega < 0.1: Adam only. No SR unless a specific fix addresses the noise amplification.
- This is not a workaround. It is the correct optimizer selection for the noise regime.

---

## Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| N=12 omega=0.01 DMC reference is wrong (2.0 looks like placeholder) | Results at this point are meaningless | Verify before Phase 5A |
| Phase 3 produces no improvement at omega=0.001 | Cannot claim consistency below omega=0.01 | Honestly report boundary of method |
| N=20 BF OOMs at 11 GB | Cannot use BF at N=20 | Jastrow-only fallback, or bf_hidden=48 |
| Reproduction phase fails (regression in known-good regimes) | Everything past Phase 0 is invalid | Debug aggressively before moving on |
| FD-colloc h parameter is wrong for low omega | Biased loss → wrong convergence | Test h sensitivity with h∈{0.001, 0.005, 0.01} |

---

## Timeline Summary

| Day | Phase | GPUs | Description |
|---|---|---|---|
| 0 (AM) | 0 | 0 | Foundation code fixes |
| 0 (PM) | 1 | 8 | Reproduce known-good regimes |
| 1 | 2 | 8 | Failure-mode diagnostics |
| 2-3 | 3 | 8 | Targeted fixes (4 independent approaches) |
| 3-4 | 4 | 8 | Scale to high N |
| 4-5 | 5 | 8 | Low-omega at high N (conditional) |
| 6 | 6 | 8 | Final consistency matrix |

**Total compute time:** ~6-7 days, 8 GPUs.
**Total GPU-hours:** ~1,100-1,300 GPU-hours.
