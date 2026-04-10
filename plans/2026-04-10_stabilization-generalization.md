# Plan: Stabilization & Generalization Campaign

Date: 2026-04-10
Status: draft
Depends-on: plans/2026-04-08_sampling-improvement-study.md (Phases 1–4 complete)

## Project objective

Produce **reliable** near-DMC weak-form collocation results across the canonical
(N, ω) grid — N ∈ {2, 6, 12, 20}, ω ∈ {1.0, 0.5, 0.1, 0.01, 0.001} — with
honest uncertainty quantification and clear understanding of what limits accuracy
in each regime.

"Reliable" means: across 3 seeds, the coefficient of variation (CV) of the
relative error is <2.0, and the worst-seed error is within 3× of the best-seed
error. Currently only N=6 ω=1.0 meets this bar (CV=1.17).

## Context

### What we know (from cross-regime analysis, 2026-04-10)

1. **Reliability is the problem, not expressiveness.** Best-ever runs reach
   DMC-level accuracy for most regimes, but mean performance across runs is
   5–650× worse than the best single run.

2. **Two difficulty axes govern error:**
   - **Dimension:** 2N coordinates (N=2 → 4D, N=20 → 40D)
   - **Correlation fraction:** (E_DMC − E_NI) / E_DMC increases as ω → 0.
     At N=6 ω=0.001, 92.9% of the energy is correlation; at ω=1.0 only 50.4%.

3. **Sampling quality (ESS, khat) is the dominant bottleneck.**
   Best results correlate with more samples (hisamp protocol ≥4× oversampling).
   N=20 ω≤0.1 is completely blocked by ESS collapse (ESS=1–3).

4. **Adam >> MinSR** at N=6 ω=0.5 in controlled 2×2 ablation
   (+0.19% vs +1.18%). MinSR is hurt by noisy Fisher when ESS is moderate.

5. **Adaptive GMM proposal gives marginal help** with Adam only
   (+0.19% adaptive vs +0.26% fixed, both with Adam).

6. **Energy reference table was wrong** for 6 entries (N=2 ω=0.001, N=12 ω=0.01
   were grossly wrong; 4 entries were missing). Fixed in this session.

### What has NOT been tested systematically

| Lever | Flag(s) | Hypothesis |
|-------|---------|------------|
| Higher sample budget | `--n-coll 8192 --oversample 16` | More samples → lower gradient variance → reliable convergence |
| ω-scaled proposal | `--sigma-fs` scaled by 1/√ω | Match proposal to wavefunction support at low ω |
| REINFORCE stabilization | `--reward-normalize --clip-el 5` | Prevent tail-dominated gradients |
| Hard-sample replay | `--replay-frac 0.1 --replay-top-frac 0.25` | Force tail coverage |
| Instability rollback | `--rollback-jump-sigma 3` | Catch divergence early, restore |
| Weight tempering | `--resample-weight-temp 0.8` | Flatten IS weight spikes |
| Transfer cascade | `--resume` from adjacent ω | Warm-start from solved regime |
| Multiple seeds | `--seed {42,137,314}` | Measure reliability, not luck |

### Best current results per regime (for comparison)

| N | ω | Best err% | Source | Reliability |
|---|---|-----------|--------|-------------|
| 2 | 1.0 | −0.0003% | thesis PINN+BF | reliable |
| 2 | 0.5 | −0.0006% | thesis PINN+BF | reliable |
| 2 | 0.1 | +0.0000% | thesis PINN+BF | reliable |
| 2 | 0.01 | −0.0014% | thesis PINN+BF | reliable |
| 2 | 0.001 | — (no DMC) | thesis PINN+CTNN | untested |
| 6 | 1.0 | +0.002% | regime_policy_campaign | reliable (CV=1.17) |
| 6 | 0.5 | −0.002% | sr_gen_v2 (hisamp) | unreliable (CV=3.51) |
| 6 | 0.1 | +0.091% | 2026-03-18 transfer | unreliable |
| 6 | 0.01 | +0.193% | regime_policy_campaign | sparse data |
| 6 | 0.001 | +0.334% | regime_policy_campaign | sparse data |
| 12 | 1.0 | +0.018% | long_campaign | moderate |
| 12 | 0.5 | +0.028% | long_campaign | moderate |
| 12 | 0.1 | +0.122% | 2026-03-18 transfer | sparse |
| 12 | 0.01 | — | untested (placeholder E_DMC=2.0) | blocked |
| 12 | 0.001 | — | untested (missing E_DMC) | blocked |
| 20 | 1.0 | — | N=20 only CTNN results exist | blocked |
| 20 | 0.5 | — | N=20 only CTNN results exist | blocked |
| 20 | 0.1 | +5.533% | 2026-03-18 transfer | unreliable |
| 20 | 0.01 | — | untested | blocked |
| 20 | 0.001 | — | untested | blocked |

## Hypotheses

### H1: Sample budget is the primary reliability lever

**Claim:** Increasing total candidate budget (n_coll × oversample) by 4× will
reduce cross-seed CV by >50% and reduce mean error by >30% at N=6 across all ω.

**Rationale:** The best-ever N=6 ω=0.5 result (err=−0.002%) used the "hisamp"
protocol (4× candidates). Standard runs at the same regime average +0.69%.
More candidates → higher ESS → lower gradient variance → more stable training.

**Test protocol:**
- Regime: N=6, all 5 ω values
- Config A (baseline): `--n-coll 4096 --oversample 8` (current default)
- Config B (hisamp): `--n-coll 4096 --oversample 16`
- Seeds: {42, 137, 314}
- Epochs: 1200
- Metric: CV of final err%, mean err%, best err%
- GPU cost: 30 runs × ~45 min = ~22.5 GPU-hours on 8 GPUs ≈ 3h wall-clock

**Success criterion:** CV(B) < 0.5 × CV(A) for at least 3 of 5 ω values.

### H2: ω-scaled proposal eliminates low-ω sampling collapse

**Claim:** Setting sigma_fs proportional to 1/√ω (wider for lower ω) will
produce healthy ESS (>50) at ω=0.01 and ω=0.001 where current fixed proposal
gives ESS < 10.

**Rationale:** The oscillator length a_ho = 1/√ω sets the natural scale of the
wavefunction support. At ω=0.001, a_ho ≈ 31.6 but sigma_fs defaults are {0.8, 1.3, 2.0}
in oscillator units — this means the proposal is already wider in absolute terms.
But the correlation hole and density tails at low ω extend much further than
the HO scale. The sigma_fs should be scaled MORE aggressively.

Proposed scaling: sigma_fs_i(ω) = base_i × max(1, (0.5/ω)^0.25)
- At ω=1.0: {0.8, 1.3, 2.0} (unchanged)
- At ω=0.5: {0.8, 1.3, 2.0} (unchanged)
- At ω=0.1: {1.07, 1.73, 2.66}
- At ω=0.01: {1.34, 2.18, 3.34}
- At ω=0.001: {1.69, 2.75, 4.22}

**Test protocol:**
- Regime: N=6, ω ∈ {0.1, 0.01, 0.001}
- Config A (fixed): default sigma_fs `"0.8,1.3,2.0"`
- Config B (ω-scaled): computed sigma_fs per ω
- Seeds: {42, 137}
- Epochs: 1200
- Track: ESS trajectory, khat trajectory, final err%
- GPU cost: 12 runs × ~45 min ≈ 2h wall-clock

**Success criterion:** Mean ESS(B) > 4× Mean ESS(A) at ω ∈ {0.01, 0.001}.

### H3: REINFORCE stabilization (reward normalization + E_L clipping) reduces training variance

**Claim:** Normalizing the REINFORCE advantage and clipping extreme E_L values
will reduce epoch-to-epoch energy oscillations by >50% and prevent divergent
training trajectories.

**Rationale:** At low ω, E_L has heavy tails due to Coulomb singularity samples.
A single extreme E_L value can dominate the REINFORCE gradient, causing a
large parameter update that makes the next batch worse. Clipping in MAD units
(--clip-el 5) removes the worst tail outliers while keeping 99% of the signal.
Reward normalization (--reward-normalize) scales the advantage by 1/std(E_L),
giving gradients a consistent scale across regimes.

**Test protocol:**
- Regime: N=6, ω ∈ {0.5, 0.1, 0.01}
- Config A (vanilla): no clipping, no normalization
- Config B (stabilized): `--clip-el 5 --reward-normalize`
- Seeds: {42, 137}
- Epochs: 1200
- Track: rolling std of E across 50-epoch windows, rollback count, final err%
- GPU cost: 12 runs ≈ 2h wall-clock

**Success criterion:** Epoch-energy oscillation amplitude(B) < 0.5 × amplitude(A) at ω=0.01.

### H4: Instability rollback prevents catastrophic mid-training divergence

**Claim:** Enabling rollback with `--rollback-jump-sigma 3` catches energy
spikes within 1 epoch and restores from the last good state, preventing
catastrophic divergence modes that currently waste entire runs.

**Rationale:** Visual inspection of training logs shows that many runs diverge
suddenly after 200–500 epochs of healthy training. Once diverged, they never
recover. A rollback mechanism that detects large energy jumps (>3σ) and
restores the prior state could save these runs.

**Test protocol:**
- Run as part of H1/H3 comparison — every B-config run includes
  `--rollback-jump-sigma 3 --rollback-decay 0.95`
- Track: number of rollbacks triggered, epoch of first rollback, recovery success
- GPU cost: 0 (bundled with H1/H3)

**Success criterion:** At least 50% of rollbacks lead to recovery (training resumes toward lower error after rollback).

### H5: Transfer cascade (warm-start from solved ω) accelerates convergence at harder ω

**Claim:** Starting from a checkpoint trained at ω_easy and gradually reducing
ω (cascade: 1.0 → 0.5 → 0.1 → 0.01 → 0.001) will converge faster and to
better final accuracy than training from scratch at each ω.

**Rationale:** Adjacent ω share most wavefunction structure. The Jastrow
correlation factor and backflow field at ω=0.5 are a good initial guess for ω=0.1.
Some best historical results (N=6 ω=0.1: +0.091%) came from transfer runs.

**Test protocol:**
- N=6: Train 800 ep at ω=1.0, then 600 ep each at 0.5, 0.1, 0.01, 0.001
  (using `--resume` from prior stage checkpoint)
- N=12: Same cascade
- Seeds: {42, 137}
- Compare against from-scratch runs at each ω (from H1-B)
- GPU cost: 2N × 5ω × 2 seeds ≈ 20 runs ≈ 5h wall-clock

**Success criterion:** Transfer final err% < from-scratch err% at ω ≤ 0.1 for ≥2 of 3 ω values.

### H6: Combined "robust recipe" generalizes across the full grid

**Claim:** The best combination of proven interventions (from H1–H5) will produce
reliable results across the entire canonical 20-cell grid, or will clearly
identify which cells are fundamentally blocked by the collocation approach.

**Test protocol:**
- Apply the winning interventions from H1–H5 to ALL 20 grid cells
- 3 seeds each = 60 runs
- This is the thesis-quality final campaign
- GPU cost: 60 runs ≈ 15h wall-clock

**Success criterion:** ≥14 of 20 cells have CV < 2.0 and best err% < 0.5%.

---

## Execution plan

### Phase 0: Foundation (this session — DONE)

- [x] Fix DMC energy table (6 corrections in src/config.py)
- [x] Verify tests pass (17/17)
- [ ] Commit energy fixes

### Phase 1: N=6 baseline reliability map + H1 + H3 + H4 (1 session)

**Goal:** Establish the reliability floor and test the three cheapest
interventions simultaneously.

Run matrix (N=6, all on 8 GPUs):

| Cell | ω | Config | Flags | Seeds | GPU |
|------|---|--------|-------|-------|-----|
| A1 | 1.0 | baseline | default | 42,137,314 | 0 |
| A2 | 0.5 | baseline | default | 42,137,314 | 1 |
| A3 | 0.1 | baseline | default | 42,137,314 | 2 |
| A4 | 0.01 | baseline | default | 42,137,314 | 3 |
| A5 | 0.001 | baseline | default | 42,137,314 | 4 |
| B1 | 1.0 | robust | hisamp+stab+rollback | 42,137,314 | 0 |
| B2 | 0.5 | robust | hisamp+stab+rollback | 42,137,314 | 1 |
| B3 | 0.1 | robust | hisamp+stab+rollback | 42,137,314 | 2 |
| B4 | 0.01 | robust | hisamp+stab+rollback | 42,137,314 | 3 |
| B5 | 0.001 | robust | hisamp+stab+rollback | 42,137,314 | 4 |

30 runs total, sequenced as 2 waves of 15 (5 GPUs × 3 seeds, then repeat).
Estimated wall-clock: ~4h.

**Baseline flags:**
```bash
--mode bf --n-elec 6 --omega $OMEGA \
--epochs 1200 --lr 5e-4 --lr-jas 5e-5 \
--n-coll 4096 --oversample 8 --micro-batch 512 \
--loss-type reinforce --direct-weight 0.1 --grad-clip 1.0 \
--vmc-every 50 --vmc-n 10000 --n-eval 30000 \
--seed $SEED --tag baseline_n6w${OMEGA_TAG}_s${SEED}
```

**Robust flags (H1+H3+H4 combined):**
```bash
--mode bf --n-elec 6 --omega $OMEGA \
--epochs 1200 --lr 5e-4 --lr-jas 5e-5 \
--n-coll 4096 --oversample 16 \          # H1: 2× more candidates
--micro-batch 512 \
--loss-type reinforce --direct-weight 0.1 --grad-clip 1.0 \
--clip-el 5.0 --reward-normalize \        # H3: REINFORCE stabilization
--rollback-jump-sigma 3 --rollback-decay 0.95 \  # H4: rollback
--adaptive-proposal --gmm-components 8 --gmm-refit-every 30 \  # adaptive proposal
--vmc-every 50 --vmc-n 10000 --n-eval 30000 \
--seed $SEED --tag robust_n6w${OMEGA_TAG}_s${SEED}
```

**Analysis after Phase 1:**
- Per ω: compare mean err%, CV, best err% between baseline and robust
- Per ω: examine ESS and khat trajectories
- Per ω: count rollbacks triggered and recovery rate
- **Decision gate:** Which interventions helped? Drop those that didn't. Lock the recipe.

### Phase 2: H2 ω-scaled proposal (1 session, parallel with Phase 1 analysis)

**Goal:** Test whether ω-scaled sigma_fs further improves ESS at low ω.

Only runs at ω ∈ {0.1, 0.01, 0.001} where proposal mismatch matters:

| Cell | ω | sigma_fs | Seeds |
|------|---|----------|-------|
| C3 | 0.1 | `"1.07,1.73,2.66"` | 42,137 |
| C4 | 0.01 | `"1.34,2.18,3.34"` | 42,137 |
| C5 | 0.001 | `"1.69,2.75,4.22"` | 42,137 |

Compare against B3/B4/B5 from Phase 1 (which use default sigma_fs + adaptive GMM).
6 runs ≈ 1.5h wall-clock.

**Decision gate:** If ω-scaled adds no benefit over adaptive GMM, drop it.
If it helps, fold into the robust recipe.

### Phase 3: H5 transfer cascade (1 session)

**Goal:** Test ω-cascade warm-starting for N=6 and N=12.

**N=6 cascade:**
```
ω=1.0 (800ep, from scratch) → checkpoint
  → ω=0.5 (600ep, --resume) → checkpoint
    → ω=0.1 (600ep, --resume) → checkpoint
      → ω=0.01 (600ep, --resume) → checkpoint
        → ω=0.001 (600ep, --resume)
```

**N=12 cascade:** (same structure)

Using the locked robust recipe from Phase 1.
2 seeds each → 2 cascades × 5 stages × 2 seeds = 20 sequential runs.
Sequential within each cascade but cascades run in parallel on separate GPUs.
Wall-clock: ~8h (the cascade is inherently sequential).

**Decision gate:** Compare cascade vs from-scratch (Phase 1) at each ω.
If cascade doesn't help at ω ≤ 0.1, abandon it for the final campaign.

### Phase 4: H6 full-grid campaign (1–2 sessions)

**Goal:** Apply the locked recipe to all 20 cells of the canonical grid.

This phase only starts AFTER Phases 1–3 have been analyzed and the recipe is locked.

**Grid:** N ∈ {2, 6, 12, 20} × ω ∈ {1.0, 0.5, 0.1, 0.01, 0.001}
**Seeds:** {42, 137, 314}
**Total runs:** 60

**Batching strategy (8 GPUs):**
- Wave 1: N=2 (5ω × 3 seeds = 15 runs, 2 per GPU, fast since N=2)
- Wave 2: N=6 (15 runs, may partially reuse Phase 1 results)
- Wave 3: N=12 (15 runs, ~2h each)
- Wave 4: N=20 (15 runs, ~3h each — may need reduced epochs for ω ≤ 0.01)

**Expected wall-clock:** ~24h total across waves.

**Analysis deliverables:**
- 20-cell heatmap of mean err% and CV
- Clear classification: solved / reachable / blocked
- For blocked cells: specific diagnosis (ESS collapse? gradient noise? expressiveness?)
- Thesis table comparing our results to DMC references + CTNN stand-ins

### Phase 5: Reporting and thesis integration

- Produce final results table for thesis (matching format of results.tex Table 5.2)
- Generate figures: err% vs ω by N, ESS trajectories, reliability heatmap
- Write analysis of what stabilization techniques worked and why
- Update plans with final outcomes

---

## Current state

**Active phase:** Phase 1 (N=6 baseline reliability map + H1 + H3 + H4)
Active step: Phase 1.2 — Full Baseline A1 launch (N=6, ω=1.0, seed=42)
Last evidence: `CUDA_MANUAL_DEVICE=0 python3.11 src/run_weak_form.py --mode bf --n-elec 6 --omega 1.0 --epochs 2 --n-coll 512 --oversample 4 --micro-batch 256 --seed 42 --tag sanity_p1_a1_n6w1_s42` produced `E=20.236229 ± 0.009092`, `err=+0.382%`, `ESS=34`, `khat=1.18`
Current risk: full-run wall time may exceed single interactive command timeout; must run in persistent async terminal and monitor
Next action: launch 1200-epoch baseline A1 run with phase-1 baseline flags and capture first-epoch diagnostics
Blockers: none

## Scope

**In scope:**
- All 20 cells of canonical (N, ω) grid
- Reliability measurement (3+ seeds per cell)
- Existing code levers only (no new code required)
- Launch scripts for tmux campaigns
- Cross-regime analysis and reporting

**Out of scope:**
- New architectures (settled per DECISIONS.md)
- MCMC/HMC sampling (abandoned per DECISIONS.md)
- MinSR (proven worse than Adam in ablation)
- Neural proposal networks (too complex for thesis timeline)
- ω=0.28 regime (not in canonical grid)

## Risk register

| Risk | Impact | Mitigation |
|------|--------|------------|
| N=20 ω≤0.01 fundamentally blocked | Can't report those cells | Report CTNN-only results; diagnose clearly |
| Robust recipe helps N=6 but not N=12+ | Partial thesis table | Characterize the dimensional scaling wall |
| Transfer cascade introduces bias | Wrong results | Always run final eval from scratch; cascade only for init |
| 60 runs exceed GPU time budget | Campaign incomplete | Prioritize N=6 full + N=12 high-ω; defer N=20 low-ω |
| Reward normalization changes optimal hyperparameters | Need re-tuning | Run Phase 1 as fair A/B with SAME lr/epochs |
