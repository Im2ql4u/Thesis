# Grand Plan: Resolving the Low-Omega / Large-N Frontier

**Date:** 2026-04-26
**Status:** Ready to execute, prioritized
**Supersedes:** `2026-04-19_first-tranche-experiments.md` (absorbs and extends it)

This document is a self-contained execution plan. It summarizes the current state of
the repository, explains why the (large N, low ω) corner fails, and gives a fully
specified sequence of experiments. Each experiment section includes exact flags,
decision criteria, and what to do based on outcomes. A reader with no prior context
should be able to execute this plan from top to bottom.

---

## Part 1 — Current State of the Grid

The following table is the authoritative current-best for the non-MCMC collocation
pipeline. All `%` errors are against the references in `src/config.py`. Surrogate
references are marked `(S)`; all others are DMC.

| N  | ω     | Best E       | Ref E       | Err %    | Ansatz     | Status               |
|----|-------|--------------|-------------|----------|------------|----------------------|
| 2  | 1.0   | 3.000055     | 3.000000    | +0.002   | Jastrow    | **Solved**           |
| 2  | 0.5   | 1.659755     | 1.659770    | -0.001   | Jastrow    | **Solved**           |
| 2  | 0.1   | 0.440808     | 0.440790    | +0.004   | BF+Jas     | **Solved**           |
| 2  | 0.01  | 0.073830     | 0.073839    | -0.014   | Jastrow    | **Solved**           |
| 2  | 0.001 | 0.013774     | 0.013778(S) | -0.029   | Jastrow    | **Solved (surrogate)**|
| 6  | 1.0   | 20.161193    | 20.159320   | +0.009   | BF+Jas     | **Excellent**        |
| 6  | 0.5   | 11.784662    | 11.784840   | -0.002   | BF+Jas     | **Excellent**        |
| 6  | 0.1   | 3.556028     | 3.553850    | +0.061   | BF+Jas     | **Strong**           |
| 6  | 0.01  | 0.691286     | 0.690360(S) | +0.134   | BF+Jas     | **Strong**           |
| 6  | 0.001 | 0.140956     | 0.140832(S) | +0.088   | BF+Jas     | **Strong**           |
| 12 | 1.0   | 65.706011    | 65.700100   | +0.009   | BF+Jas     | **Strong**           |
| 12 | 0.5   | 39.169168    | 39.159600   | +0.024   | BF+Jas     | **Strong**           |
| 12 | 0.1   | 12.282367    | 12.269840   | +0.102   | BF+Jas     | **Strong**           |
| 12 | 0.01  | 2.501093     | 2.473630(S) | +1.110   | BF+Jas     | **Borderline**       |
| 12 | 0.001 | 0.531987     | 0.515365(S) | +3.225   | BF+Jas     | **Rescued, not solved** |
| 20 | 1.0   | 157.283168   | 155.882200  | +0.899   | Jastrow    | **Usable**           |
| 20 | 0.5   | 96.119158    | 93.875200   | +2.390   | Jastrow    | **Weak**             |
| 20 | 0.1   | 31.636564    | 29.977900   | +5.533   | Jastrow    | **Unresolved**       |
| 20 | 0.01  | 12.295074    | 6.146450(S) | +100.035 | Jas+Shell  | **Blocked**          |
| 20 | 0.001 | —            | 1.293033(S) | —        | —          | **No result**        |

### Reading this table

- Cells labeled **Solved** or **Excellent** or **Strong** are thesis-ready. Do not
  re-run them unless you specifically want a robustness check.
- `N=12, ω=0.001` has now been re-run post-fix. The old +72% number was invalid;
  the confirmed current result is `0.531987 ± 0.000224` (+3.225%) from
  `results/arch_colloc/confirmF2_temp075_12k.pt`. This is a real rescue, but
  still not a solved cell.
- `N=20, ω=0.01` is the key diagnostic case. ESS is now recoverable (via
  ShellFlow stabilization), but the energy is still near 2× the reference. The
  sampling problem is partially solved; the ansatz problem is not.
- Surrogate references for N≥6, ω≤0.01 come from the PINN+CTNN model — the same
  family of methods being tested here. The surrogate itself may have 5-20% error
  relative to true DMC. Keep this in mind when interpreting percentage errors in
  those cells; the goal is "close to surrogate" as a proxy, not absolute truth.

---

## Part 2 — Root-Cause Diagnosis

Before launching experiments, understand WHY the failures happen. This section is
required reading. Skipping it means launching experiments blind.

### 2.1 — Sampling is no longer the dominant bottleneck at N=20

The April 2026 ShellFlow campaign showed that all proposal variants now cluster
tightly near +109% error at N=20 ω=0.01:

```
2shell:   +108.9%
epoch2:   +109.2%
ess2:     +109.2%
floor020: +109.3%
3shell:   +109.3%
radius3:  +109.4%
floor010: +110.3%
```

If the problem were primarily the proposal (sampling), you would see large spread
between these variants — the better proposal would give substantially lower error.
The tight clustering means all proposals are now "good enough" for the optimizer
to do its job, but the optimizer is still getting stuck.

**Conclusion:** Stop tuning proposals at N=20 ω=0.01. The bottleneck has shifted to
the wavefunction ansatz and/or the loss landscape.

### 2.2 — The Slater determinant node structure is qualitatively wrong at low ω

This is the deepest issue and has not been explicitly named before.

The ansatz is Ψ(x) = SD(x) × exp(J(x)), where SD is a Slater determinant of
harmonic oscillator orbitals and J is the neural Jastrow factor.

At low ω, the physical ground state enters the Wigner molecule regime. Electrons
localize at specific crystallographic sites. The nodal surface (zeros of the
antisymmetric wavefunction) of the TRUE ground state corresponds to this localized
structure.

The SD of HO orbitals has a DIFFERENT nodal surface — one corresponding to
delocalized wavefunctions. The Jastrow factor can reshape the wavefunction
amplitude, but it **cannot move the nodal surface**. The nodal surface is
determined entirely by the SD.

At N=6, the mismatch between the HO SD nodes and the Wigner crystal nodes is
small enough that the Jastrow correction bridges the gap (+0.088% at ω=0.001).
At N=12 and N=20, the mismatch grows because more shells are filled and the
crystal structure becomes more complex. The Jastrow alone cannot fix this.

The correct remedy is a **backflow transformation**, which effectively evaluates
the SD at electron-position-dependent "quasi-coordinates" instead of the raw
positions. This moves the nodal surface and allows the ansatz to represent the
Wigner crystal ground state.

Backflow at N=20 was tried earlier but failed. That failure was confounded:
  (a) The IS bug was present — gradients were biased.
  (b) BF was not warm-started from a good Jastrow checkpoint.
  (c) BF was trained from scratch at N=20, which is extremely difficult.

None of these confounds have been removed simultaneously and re-tested.

### 2.3 — The radius cap is hiding the relevant geometry

The current ShellFlow scripts use `--jas-input-radius-cap-aho 2.0`. This clips the
Jastrow network's awareness of positions to 2 oscillator lengths from the origin.

In the Wigner molecule regime, the inter-electron distances (in oscillator units)
can be 5-10 a_ho or more. By capping at 2 a_ho, the Jastrow network cannot learn
anything about the spatial structure that matters most at low ω. This cap was
added for stabilization but may be the main reason the Jastrow is stuck.

This is trivially ablatable: run with cap=3.0 and uncapped. Expected result: the
uncapped version should substantially improve energy even without any other change.

### 2.4 — N-curriculum has never been properly tested

The entire training history uses ω-curriculum (start at high ω, cascade to low ω)
but has not seriously exploited N-curriculum (train N=12 first, then transfer
weights to N=16, then to N=20).

The Jastrow network is permutation-equivariant. Its parameters do not hard-code N
— they process particle pairs and pools. A Jastrow trained at N=12 ω=0.01 (which
is at +1.11% and borderline solved) is a plausible warm-start for N=16 ω=0.01,
and N=16 for N=20.

The Slater determinant does change with N (different orbital occupancy). But:
  - In Jastrow-only mode, the SD is not a neural network — it's computed from
    fixed HO orbitals, and the Jastrow warm-start is still valid.
  - The correlation captured by the Jastrow (pair distances, shell structure) is
    qualitatively similar at N=12 and N=20 in the same ω regime.

This is currently labeled "deferred" in `2026-04-19_first-tranche-experiments.md`.
It should be elevated to Priority 2.

### 2.5 — N=20 training budget is ~10× underpowered

N=6 best runs: ~6000 epochs × 8192 n-coll = **49M collocation points**
N=20 overnight runs: 5600 epochs × 1024 n-coll = **5.7M collocation points**

N=20 lives in 40-dimensional space. N=6 lives in 12D. The N=20 optimizer is doing
harder work with 10× less data. The high-sample ablation for N=12 (+1.11% might
improve with more budget) is already planned; the same logic applies to N=20.

### 2.6 — N=12 ω=0.001 is rescued but still sampling-stressed

The only historical N=12 ω=0.001 result (+72.371%) was produced before the
importance-sampling bugfix in March 2026 and is invalid as a performance claim.
The 2026-04-26 rapid/refinement sequence lowered the confirmed energy to
`0.531987 ± 0.000224`, or +3.225% against the surrogate reference.

This is not primarily a high-budget success. The useful path was rapid promotion:
short VMC-probed continuations from the post-fix checkpoint, with lower learning
rates and `oversample=16`. The remaining warning sign is severe importance-weight
pathology throughout the successful runs: raw ESS is usually `1-6` out of 16,384
candidates and PSIS-khat is roughly `8-10`. The model can still be polished, but
the sampler is not healthy.

### 2.7 — Reference quality for low-ω large-N cells is uncertain

For N=6, 12, 20 at ω=0.01 and ω=0.001, no DMC references exist. The surrogate
references come from PINN+CTNN, which is the same model family. The surrogate
errors are unknown but plausibly 10-20% above true DMC at these regimes. This
means the "+100%" number at N=20 ω=0.01 might partly reflect a bad reference, not
just a bad model. This does NOT change the experiment plan — you still want to
minimize energy — but it should temper how you interpret percentage errors in
these cells. The honest goal is "beat the surrogate," not "achieve DMC quality."

---

## Part 3 — Experiment Plan

Experiments are ordered by priority. Do not skip ahead. Each experiment is
gated on the result of the previous one where indicated.

---

### Experiment A — Uncap the Jastrow radius at N=20 ω=0.01

**Priority:** 1 (run first, cheapest diagnosis)
**Question:** Is the `--jas-input-radius-cap-aho 2.0` restriction hiding the
geometry the Jastrow needs to see?
**Gate:** Run before any other N=20 ω=0.01 experiment.
**Expected wall-clock:** ~3 hours on 3 GPUs.

#### Background

The best current N=20 ω=0.01 result (+100%) uses `--jas-input-radius-cap-aho 2.0`.
At low ω, Wigner molecule inter-site distances are 5-10 a_ho. The cap may be
preventing the Jastrow from learning this structure at all.

#### Runs

Launch 3 parallel jobs:

**Job A1 — cap=2.0 (baseline, reproduce current best)**
```bash
python3 src/run_weak_form.py \
  --mode jastrow --n-elec 20 --omega 0.01 \
  --seed 42 \
  --epochs 400 \
  --n-coll 2048 --oversample 64 --micro-batch 256 \
  --loss-type reinforce --direct-weight 0.0 \
  --clip-el 4.0 --grad-clip 1.0 --reward-qtrim 0.01 \
  --lr 8e-5 --lr-jas 8e-6 \
  --lr-warmup-epochs 20 --lr-warmup-init-frac 0.1 --lr-min-frac 0.01 \
  --rollback-decay 0.98 --rollback-err-pct 0.0 --rollback-jump-sigma 5.0 \
  --ess-floor-ratio 0.01 --ess-oversample-max 256 --ess-oversample-step 32 \
  --resample-weight-temp 0.60 --resample-logw-clip-q 0.995 \
  --jas-input-radius-cap-aho 2.0 \
  --vmc-every 50 --vmc-n 15000 --n-eval 20000 \
  --no-pretrained \
  --tag expA1_cap2_n20_w001
```

**Job A2 — cap=3.0**
```bash
# Same as A1 but change the cap flag:
  --jas-input-radius-cap-aho 3.0 \
  --tag expA2_cap3_n20_w001
```

**Job A3 — uncapped (remove the cap flag entirely)**
```bash
# Same as A1 but remove --jas-input-radius-cap-aho entirely
  --tag expA3_uncap_n20_w001
```

#### Evaluation

After all 3 jobs finish, compare:
- The final heavy-VMC energy for each (reported as `--vmc-n 15000` probes, or
  run a dedicated `--n-eval 50000` eval pass on the best checkpoint).
- Report the best probe VMC error against the surrogate reference (6.146450).

**Decision tree:**
- If A3 (uncapped) beats A1 (cap=2) by more than 5 percentage points in err%:
  → Remove the cap from all future N=20 ω=0.01 runs. Update the ShellFlow scripts.
  → The cap was a significant limiter. Proceed to Experiment B with uncapped.
- If A2 or A3 beats A1 but only by 1-3 percentage points:
  → The cap is a mild limiter. Prefer cap=3 as a compromise in future runs.
- If all three are within 2 percentage points of each other:
  → The cap is not the bottleneck. The Jastrow's lack of expressivity is confirmed.
  → The node structure (Section 2.2) is the dominant issue. Proceed directly to
    Experiment C (warm-started BF).

---

### Experiment B — High sample budget ablation at N=12 ω=0.01

**Priority:** 2 (run in parallel with A if GPUs are available)
**Question:** Is the +1.11% gap at N=12 ω=0.01 mostly a sample-budget problem?
**Expected wall-clock:** ~5 hours on 3 GPUs.

#### Background

N=12 ω=0.01 sits at +1.11%, which is tantalizingly close to the <0.5% band
achieved at ω≥0.1. The current best was trained with modest n-coll settings.
Increasing the candidate budget is the cheapest path to improving this cell.

Best checkpoint to resume from:
`outputs/2026-03-20_2114_campaign_v5_10h_fix/logs/v5_n12w001_bf_cascade.log`
(or the corresponding `.pt` file in `results/arch_colloc/`).

#### Runs

**Job B1 — higher n-coll, capped oversample**
```bash
python3 src/run_weak_form.py \
  --mode bf --n-elec 12 --omega 0.01 \
  --seed 42 \
  --epochs 3000 \
  --n-coll 8192 --oversample 16 --micro-batch 512 \
  --loss-type reinforce --direct-weight 0.0 \
  --clip-el 5.0 --grad-clip 1.0 --reward-qtrim 0.02 \
  --lr 1.2e-4 --lr-jas 1.2e-5 \
  --lr-warmup-epochs 60 --lr-warmup-init-frac 0.1 --lr-min-frac 0.01 \
  --ess-floor-ratio 0.03 --ess-oversample-max 32 --ess-oversample-step 4 \
  --rollback-decay 0.97 --rollback-err-pct 0.0 --rollback-jump-sigma 4.5 \
  --vmc-every 100 --vmc-n 20000 --n-eval 60000 \
  --sigma-fs 0.8,1.3,2.0,3.5,6.0 \
  --resume results/arch_colloc/v5_n12w001_bf_cascade.pt \
  --tag expB1_n12_w001_hisamp_cap16
```

**Job B2 — higher n-coll, uncapped oversample (heavier)**
```bash
# Same as B1 but:
  --oversample 32 \
  --ess-oversample-max 64 \
  --tag expB2_n12_w001_hisamp_uncap
```

**Job B3 — very high oversample (maximum pressure)**
```bash
# Same as B1 but:
  --oversample 48 --n-coll 6144 \
  --ess-oversample-max 96 \
  --tag expB3_n12_w001_hisamp_maxpressure
```

#### Evaluation

Report the heavy-VMC energy (`--n-eval 60000`) for each job's best checkpoint.
Compare against reference 2.473630 (surrogate).

**Decision tree:**
- If any B job reaches <0.5% error:
  → N=12 ω=0.01 is solved. Record in the status table. Move on.
- If best B job is in 0.5-1.1% range:
  → Sample budget helps but doesn't fully close the gap. The remaining gap is
    likely ansatz-limited. Flag for later but do not invest more compute here.
- If all B jobs are ≥1.1%:
  → Sample budget does not help. The bottleneck is something else (LR schedule,
    ansatz capacity, or the run has already converged). Do not launch B4+.

---

### Experiment C — Warm-started tiny BF at N=20 ω=1.0 and ω=0.1

**Priority:** 3 (the most important ansatz test; run after Experiment A is evaluated)
**Question:** Does a tiny backflow head on top of the best Jastrow checkpoint
improve N=20 accuracy? If yes, the bottleneck is ansatz expressivity, not sampling.
**Expected wall-clock:** ~8 hours on 4 GPUs.
**Gate:** Run Experiment A first. The best Jastrow checkpoint from A3 (uncapped)
should be the warm-start; if A still gives +100%, use the best available N=20
Jastrow checkpoint from `results/arch_colloc/`.

#### Background

The current N=20 results use Jastrow-only. Backflow was tried before but was
confounded by the IS bug and lack of warm-start. The hypothesis is:

- The Jastrow's nodal surface (from the Slater determinant) is wrong for the
  Wigner regime.
- A tiny backflow can deform the nodal surface with very few parameters.
- Warm-starting from the Jastrow means the BF starts at zero displacement
  (identity) and gradually learns small corrections.

The key is that BF is initialized with `bf_scale_init=0.05` (from the code),
meaning its initial displacement is nearly zero. The Jastrow starts in a good
state and the BF learns small corrections. This is very different from training
BF from scratch.

#### Checkpoint to warm-start from

Use the best N=20 ω=1.0 Jastrow checkpoint:
`results/arch_colloc/n20x2_adam_w1_best.pt`

For ω=0.1, use:
`results/arch_colloc/` — the best checkpoint from the ShellFlow ω=0.01 campaign
may also be a valid start for ω=0.1 warm-start. If no good ω=0.1 Jastrow
checkpoint exists, run from the ω=1.0 checkpoint with a brief ω=0.1 warmup.

#### Runs

**Job C1 — tiny BF (hidden=48) at ω=1.0**
```bash
python3 src/run_weak_form.py \
  --mode bf \
  --bf-hidden 48 --bf-msg-hidden 48 --bf-layers 2 \
  --n-elec 20 --omega 1.0 \
  --seed 42 \
  --epochs 2000 \
  --n-coll 1024 --oversample 8 --micro-batch 128 \
  --loss-type reinforce --direct-weight 0.0 \
  --clip-el 5.0 --grad-clip 1.0 --reward-qtrim 0.01 \
  --lr 6e-5 --lr-jas 6e-6 \
  --lr-warmup-epochs 40 --lr-warmup-init-frac 0.1 --lr-min-frac 0.01 \
  --ess-floor-ratio 0.02 --ess-oversample-max 32 --ess-oversample-step 4 \
  --rollback-decay 0.97 --rollback-err-pct 0.0 --rollback-jump-sigma 5.0 \
  --vmc-every 200 --vmc-n 15000 --n-eval 30000 \
  --init-jas results/arch_colloc/n20x2_adam_w1_best.pt \
  --no-pretrained \
  --tag expC1_n20_w1_tinybf48
```

**Job C2 — smaller BF (hidden=32) at ω=1.0 — memory-lighter**
```bash
# Same as C1 but:
  --bf-hidden 32 --bf-msg-hidden 32 \
  --tag expC2_n20_w1_tinybf32
```

**Job C3 — tiny BF (hidden=48) at ω=0.1**
```bash
# Same as C1 but:
  --omega 0.1 \
  --sigma-fs 0.8,1.3,2.0,3.5,6.0 \
  --init-jas <best_n20_w01_checkpoint.pt> \
  --tag expC3_n20_w01_tinybf48
```

**Job C4 — frozen Jastrow, train BF head only (first 500 epochs), then unfreeze**
This is the staged variant. It avoids the Jastrow getting corrupted early.
Implementation note: use `--freeze-jastrow-epochs 500` if supported; otherwise,
manually run two stages — first a run with frozen Jastrow, then resume with
everything unfrozen.
```bash
# Stage 1: Freeze Jastrow for 500 epochs
python3 src/run_weak_form.py \
  --mode bf --bf-hidden 48 --bf-msg-hidden 48 --bf-layers 2 \
  --n-elec 20 --omega 1.0 \
  --seed 42 --epochs 500 \
  --n-coll 1024 --oversample 8 --micro-batch 128 \
  --loss-type reinforce --direct-weight 0.0 \
  --clip-el 5.0 --grad-clip 0.5 --reward-qtrim 0.01 \
  --lr 6e-5 --lr-jas 0.0 \
  --vmc-every 100 --vmc-n 15000 \
  --init-jas results/arch_colloc/n20x2_adam_w1_best.pt \
  --no-pretrained \
  --tag expC4_n20_w1_bfonly_stage1

# Stage 2: Resume with Jastrow unfrozen
python3 src/run_weak_form.py \
  --mode bf --bf-hidden 48 --bf-msg-hidden 48 --bf-layers 2 \
  --n-elec 20 --omega 1.0 \
  --seed 42 --epochs 1500 \
  --n-coll 1024 --oversample 8 --micro-batch 128 \
  --loss-type reinforce --direct-weight 0.0 \
  --clip-el 5.0 --grad-clip 1.0 --reward-qtrim 0.01 \
  --lr 4e-5 --lr-jas 4e-6 \
  --vmc-every 200 --vmc-n 15000 --n-eval 30000 \
  --resume <checkpoint_from_stage1> \
  --tag expC4_n20_w1_bfonly_stage2
```

#### Evaluation

Compare the final heavy-VMC error against:
- The current Jastrow-only baseline at ω=1.0: **+0.899%**
- The current Jastrow-only baseline at ω=0.1: **+5.533%**

**Decision tree:**
- If tiny BF at ω=1.0 reaches < +0.5% (improving on +0.899%):
  → BF is helping. Bottleneck was ansatz capacity. Scale BF investment at N=20.
  → Proceed to Experiment D (N-curriculum) with BF as the target ansatz.
- If tiny BF at ω=1.0 is within 0.1% of the Jastrow baseline:
  → BF is not adding capacity. The Jastrow is already near-optimal at ω=1.0.
  → Look at ω=0.1 result (C3) — the Wigner crystal regime is where BF should help.
- If tiny BF at ω=0.1 beats Jastrow (+5.533%) by ≥1 point:
  → Confirmed: the nodal surface mismatch is the bottleneck at low ω.
  → Proceed to Experiment D with BF as the target ansatz.
- If BF makes things WORSE:
  → The Jastrow warm-start is being corrupted. Try C4 (frozen-Jastrow stage).
  → If C4 still worsens things, reduce BF learning rate by 10× and re-run.

---

### Experiment D — N-curriculum: 12 → 16 → 20 at ω=0.1

**Priority:** 4 (run after Experiment C is evaluated)
**Question:** Does bootstrapping through intermediate N values unlock N=20 low-ω
accuracy that direct training cannot reach?
**Gate:** Requires N=12 ω=0.1 checkpoint (exists: `+0.102%` strong) and a working
GPU.
**Expected wall-clock:** ~12 hours total on 3 GPUs (sequential jobs).

#### Background

The Jastrow network is permutation-equivariant and its parameters are N-agnostic.
This means you can load a Jastrow trained on N=12 electrons as an initialization
for N=16 electrons — the pair-interaction weights are learned on particle pairs,
and the same pair structure exists in any N. The Slater determinant changes
(different orbital occupancies) but in Jastrow-only mode, the SD uses fixed HO
orbitals and doesn't need to be warm-started.

This is a completely untested training axis. Every campaign so far has varied ω
with fixed N. The jump from N=12 to N=20 is likely too large; go through N=16.

#### Step D1 — Train N=16 ω=0.1 from N=12 ω=0.1 Jastrow weights

```bash
python3 src/run_weak_form.py \
  --mode jastrow --n-elec 16 --omega 0.1 \
  --seed 42 \
  --epochs 3000 \
  --n-coll 2048 --oversample 12 --micro-batch 256 \
  --loss-type reinforce --direct-weight 0.0 \
  --clip-el 5.0 --grad-clip 1.0 --reward-qtrim 0.02 \
  --lr 1e-4 --lr-jas 1e-5 \
  --lr-warmup-epochs 60 --lr-warmup-init-frac 0.1 --lr-min-frac 0.01 \
  --ess-floor-ratio 0.02 --ess-oversample-max 48 --ess-oversample-step 8 \
  --rollback-decay 0.97 --rollback-err-pct 0.0 --rollback-jump-sigma 5.0 \
  --sigma-fs 0.8,1.3,2.0,3.5,6.0 \
  --vmc-every 100 --vmc-n 15000 --n-eval 40000 \
  --init-jas <best_n12_w01_jastrow_checkpoint.pt> \
  --no-pretrained \
  --tag expD1_n16_w01_from_n12
```

Note: There is no DMC reference for N=16. You will have to estimate quality by
comparing the energy to the Thomas-Fermi or PINN estimate, or accept this as an
intermediate step and evaluate quality only at N=20. The goal here is NOT to
have a great N=16 result — it is to have a good warm-start for N=20.

Gate: Run D2 only after D1 has trained for at least 1000 epochs and shows
a VMC probe energy that is not diverging (increasing monotonically for 200+
epochs). If D1 diverges, reduce LR by 2× and restart.

#### Step D2 — Train N=20 ω=0.1 from N=16 ω=0.1 Jastrow weights

```bash
python3 src/run_weak_form.py \
  --mode jastrow --n-elec 20 --omega 0.1 \
  --seed 42 \
  --epochs 3000 \
  --n-coll 1024 --oversample 8 --micro-batch 128 \
  --loss-type reinforce --direct-weight 0.0 \
  --clip-el 5.0 --grad-clip 1.0 --reward-qtrim 0.02 \
  --lr 8e-5 --lr-jas 8e-6 \
  --lr-warmup-epochs 60 --lr-warmup-init-frac 0.1 --lr-min-frac 0.01 \
  --ess-floor-ratio 0.01 --ess-oversample-max 64 --ess-oversample-step 8 \
  --rollback-decay 0.98 --rollback-err-pct 0.0 --rollback-jump-sigma 5.0 \
  --sigma-fs 0.8,1.3,2.0,3.5,6.0 \
  --vmc-every 100 --vmc-n 15000 --n-eval 30000 \
  --init-jas <best_n16_w01_checkpoint_from_D1.pt> \
  --no-pretrained \
  --tag expD2_n20_w01_from_n16
```

#### Evaluation for D

Compare final heavy-VMC energy against:
- Current N=20 ω=0.1 baseline: **+5.533%** (reference 29.977900)

**Decision tree:**
- If D2 beats 5.533% by ≥1 percentage point:
  → N-curriculum works. This is a major finding. Extend to ω=0.01.
  → Proceed to Experiment E.
- If D2 is within 1 point of 5.533%:
  → N-curriculum helps marginally. The bottleneck at N=20 ω=0.1 is likely the
    Jastrow ansatz capacity, not the starting point. Combine with Experiment C
    (BF warm-start after N-curriculum).
- If D2 is worse than 5.533%:
  → N-curriculum is not improving things, possibly because the N=16 step is too
    noisy as a bridge. Try D2 directly from N=12 (skipping N=16) and compare.

---

### Experiment E — N-curriculum extension to ω=0.01 (gated on D)

**Priority:** 5 (run only if D is successful)
**Question:** Can the N-curriculum path reach N=20 ω=0.01 from below +20% error?
**Gate:** D2 must have beaten +5.533% at ω=0.1. Use D2's best checkpoint as
the warm-start, then cascade ω from 0.1 → 0.05 → 0.01.

#### Step E1 — N=20 ω=0.05 from D2

```bash
python3 src/run_weak_form.py \
  --mode jastrow --n-elec 20 --omega 0.05 \
  --seed 42 --epochs 2000 \
  --n-coll 1024 --oversample 16 --micro-batch 128 \
  --loss-type reinforce --direct-weight 0.0 \
  --clip-el 5.0 --grad-clip 1.0 --reward-qtrim 0.01 \
  --lr 6e-5 --lr-jas 6e-6 \
  --lr-warmup-epochs 40 --lr-warmup-init-frac 0.1 --lr-min-frac 0.01 \
  --ess-floor-ratio 0.01 --ess-oversample-max 128 --ess-oversample-step 16 \
  --resample-weight-temp 0.65 --resample-logw-clip-q 0.995 \
  --sigma-fs 0.5,0.9,1.4,2.2,3.8,6.5 \
  --rollback-decay 0.98 --rollback-err-pct 0.0 --rollback-jump-sigma 5.0 \
  --vmc-every 100 --vmc-n 15000 --n-eval 25000 \
  --resume <best_n20_w01_checkpoint_from_D2.pt> \
  --tag expE1_n20_w005_from_D2
```

Note: There is no reference for N=20 ω=0.05. Use the PINN surrogate energy if
available, or treat this step purely as a bridge. What matters is that the
VMC probe energy is decreasing (not diverging).

#### Step E2 — N=20 ω=0.01 from E1

```bash
python3 src/run_weak_form.py \
  --mode jastrow --n-elec 20 --omega 0.01 \
  --seed 42 --epochs 2000 \
  --n-coll 1024 --oversample 32 --micro-batch 128 \
  --loss-type reinforce --direct-weight 0.0 \
  --clip-el 4.0 --grad-clip 1.0 --reward-qtrim 0.01 \
  --lr 5e-5 --lr-jas 5e-6 \
  --lr-warmup-epochs 40 --lr-warmup-init-frac 0.1 --lr-min-frac 0.01 \
  --ess-floor-ratio 0.01 --ess-oversample-max 256 --ess-oversample-step 32 \
  --resample-weight-temp 0.60 --resample-logw-clip-q 0.995 \
  --top1-mass-update-ceiling 0.22 --top10-mass-update-ceiling 0.70 \
  --sigma-fs 0.3,0.6,1.0,1.8,3.2,6.0 \
  --rollback-decay 0.98 --rollback-err-pct 0.0 --rollback-jump-sigma 5.0 \
  --vmc-every 60 --vmc-n 15000 --n-eval 25000 \
  --resume <best_n20_w005_checkpoint_from_E1.pt> \
  --tag expE2_n20_w001_from_E1
```

#### Evaluation for E

Compare against: current N=20 ω=0.01 baseline: **+100.035%** (reference 6.146450).

Any result below +50% is a major improvement. Any result below +20% means the
N-curriculum + ω-cascade together have unlocked this cell. Below +5% would be a
thesis headline result.

---

### Experiment F — Re-run N=12 ω=0.001 post-bugfix

**Priority:** 6 (can run in parallel with D, requires only 1 GPU)
**Question:** What is the actual state of N=12 ω=0.001 with correct importance
sampling weights?
**Background:** The current +72.371% result was trained before the IS bugfix in
March 2026. It is almost certainly wrong. This is a straightforward re-run.

```bash
python3 src/run_weak_form.py \
  --mode bf --n-elec 12 --omega 0.001 \
  --seed 42 \
  --epochs 5000 \
  --n-coll 4096 --oversample 16 --micro-batch 512 \
  --loss-type reinforce --direct-weight 0.0 \
  --clip-el 5.0 --grad-clip 1.0 --reward-qtrim 0.02 \
  --lr 8e-5 --lr-jas 8e-6 \
  --lr-warmup-epochs 60 --lr-warmup-init-frac 0.1 --lr-min-frac 0.01 \
  --ess-floor-ratio 0.05 --ess-oversample-max 64 --ess-oversample-step 8 \
  --resample-weight-temp 0.65 --resample-logw-clip-q 0.995 \
  --rollback-decay 0.97 --rollback-err-pct 0.0 --rollback-jump-sigma 5.0 \
  --sigma-fs 0.3,0.5,0.8,1.4,2.5,4.5,8.0 \
  --vmc-every 100 --vmc-n 20000 --n-eval 60000 \
  --init-jas <best_n12_w001_checkpoint.pt> \
  --no-pretrained \
  --tag expF1_n12_w0001_postfix
```

Warm-start checkpoint: use the best N=12 ω=0.01 checkpoint (`v5_n12w001_bf_cascade.pt`
or the best result from Experiment B if that finishes first).

**Decision tree:**
- If result is <5% error: major improvement post-bugfix. Record and update table.
- If result is 5-30% error: the fix helped but the problem is still hard. The
  Wigner regime at N=12 ω=0.001 requires more work (N-curriculum or BF at N=12).
- If result is >30% error: the IS fix alone wasn't enough. Check ESS during training;
  if ESS is consistently low, add `--adaptive-proposal` with the standard GMM and
  try again with wider sigma-fs.

---

## Part 4 — What NOT to Do

This section is equally important as the experiment plan.

### Do not run more ShellFlow proposal variants at N=20 ω=0.01

The April 2026 sweep showed all variants clustering within 2% of each other
(+108% to +110%). The proposal is no longer the bottleneck. More ShellFlow tuning
will not move the frontier.

### Do not run N=20 ω=0.001 until ω=0.01 is below 20%

N=20 ω=0.001 has no result at all. It makes no sense to attempt it while ω=0.01
is at +100%. The ω cascade must be respected.

### Do not interpret N=20 surrogate-backed % errors as equivalent to DMC-backed % errors

The surrogate reference for N=20 ω=0.01 comes from PINN+CTNN. It may be 10-20%
above the true DMC value. A result of "+50% vs surrogate" might in truth be
"+30-35% vs DMC." This does not change the goal (minimize energy), but it does
mean you should not panic about absolute numbers in these cells.

### Do not start a fresh BF run at N=20 without a Jastrow warm-start

Starting BF from scratch at N=20 puts the ansatz far from any good minimum. The
BF learns to output large displacements early and corrupts the wavefunction. Always
warm-start BF from a Jastrow checkpoint with `--init-jas`.

### Do not consume GPU budget on N=6 or N=2

These cells are solved. Any regression there is a bug in the code, not a physics
problem. If you suspect regression, run a quick smoke test (50 epochs) on one cell
to verify the infrastructure hasn't broken.

---

## Part 5 — Evaluation Protocol

All experiments use the same evaluation protocol. Deviations must be documented.

### Checkpoint selection

- During training, use VMC probes (`--vmc-n 15000`) every 50-200 epochs.
- The "best" checkpoint is the one with the lowest mean VMC probe energy over a
  window of the last `--save-best-window` probes (default 30).
- Do NOT use the collocation-batch energy to select checkpoints — it is biased.

### Final evaluation

After selecting the best checkpoint, run a dedicated heavy evaluation:

```bash
python3 src/run_weak_form.py \
  --mode <mode> --n-elec <N> --omega <omega> \
  --n-eval 50000 \
  --resume <best_checkpoint.pt> \
  --eval-only \
  --tag <experiment_tag>_finaleval
```

If `--eval-only` is not supported, run 0 training epochs with `--epochs 0` and
capture the final VMC energy from the log.

### Reporting

For each experiment, report in this format:
```
Experiment: <tag>
N=<N>, ω=<ω>
Best VMC energy: <E> ± <σ>
Reference: <E_ref> (<type: DMC or surrogate>)
Error: <(E - E_ref) / |E_ref| × 100>%
Epochs trained: <n>
Evaluation samples: <n_eval>
Checkpoint: <path>
```

---

## Part 6 — Success Criteria and Thesis Implications

### Minimum thesis-viable outcomes

These are the results needed for the thesis narrative to hold:

| Cell | Current | Target | Priority |
|------|---------|--------|----------|
| N=12 ω=0.001 | +72% (stale) | < 5% | High — need post-fix number |
| N=20 ω=1.0 | +0.899% | < 0.5% | Medium |
| N=20 ω=0.1 | +5.533% | < 2% | High |
| N=20 ω=0.01 | +100% | < 20% | High |

### What makes a headline result

If N=20 ω=0.01 reaches <20% error via the N-curriculum path (Experiment E),
that is a headline result: it would show that exploiting the N-transfer structure
of the Jastrow network unlocks a regime that was previously completely blocked.

If warm-started BF (Experiment C) improves N=20 ω=0.1 from +5.5% to <2%, that
is also a headline result: it confirms the nodal surface hypothesis and provides
a clear physical explanation for why Jastrow alone is insufficient in the Wigner
regime.

Either of these, combined with the N=6 results already in hand, constitutes a
strong thesis.

### What does NOT constitute a headline result

- Improving N=20 ω=0.01 from +100% to +80% by tuning the proposal further.
  This is incremental and does not resolve any scientific question.
- Showing that all N=20 low-ω variants cluster near +100%. This is already known.
- Confirming N=2 or N=6 cells more precisely. These are already solved.

---

## Part 7 — Execution Order Summary

| # | Experiment | GPU hours | Gate |
|---|-----------|-----------|------|
| A | Uncap Jastrow radius at N=20 ω=0.01 | 3h × 3 GPU | None |
| B | High-sample N=12 ω=0.01 | 5h × 3 GPU | None (parallel with A) |
| C | Warm-started tiny BF at N=20 ω=1.0, 0.1 | 8h × 4 GPU | After A evaluated |
| D | N-curriculum 12→16→20 at ω=0.1 | 12h × 3 GPU | After A evaluated |
| E | Extend N-curriculum to ω=0.01 | 6h × 2 GPU | After D success |
| F | Re-run N=12 ω=0.001 post-bugfix | 8h × 1 GPU | Parallel with D |

Total estimated GPU-hours if all succeed: ~120 GPU-hours.
Total estimated wall-clock if GPUs are available: 3-4 days.

---

## Part 8 — If Everything Fails

If Experiments A through D all fail to improve on the baselines, the following
less-explored directions remain:

1. **Center-of-mass subtraction**: The current parameterization includes global
   drift in the inputs. Subtracting the center of mass from all particle positions
   before feeding to the network removes a spurious degree of freedom and may
   improve learning, especially at large N where global drift is large.

2. **Variance minimization loss**: Replace the REINFORCE (energy minimization) loss
   with a VMC-style variance minimization objective. At low ω, the local energy
   variance is very large relative to the mean, making REINFORCE nearly useless.
   Variance minimization directly targets the quantity that measures wavefunction
   quality and may have a better-conditioned landscape.

3. **Orbital localization prior**: At very low ω, initialize the Slater determinant
   orbitals at the known classical Wigner crystal sites rather than using standard
   HO orbitals. This requires modifying how the SD is computed in `Slater_Determinant.py`.
   The classical minimum-energy positions for N electrons in a parabolic trap are
   known in the literature for N up to ~20.

4. **Pfaffian ansatz**: For the Wigner crystal regime, a geminal or Pfaffian wavefunction
   is more naturally suited than a Slater determinant. This would require a significant
   architecture change but may be the only way to achieve DMC-quality at N=20 ω=0.001.

---

*This document should be updated after each experiment completes. Record results
in the table in Part 1, update the gate status for each experiment, and append
a brief outcome summary at the end of the relevant experiment section.*

---

## Rapid Triage Update — 2026-04-26

The original stage-1 A+B launcher was stopped because it expanded the plan into
completion-scale jobs before decision-level evidence was available. It also hid a
major compatibility issue: old checkpoints were trained with the legacy cusp decay
`exp(-r)`, while the current code defaults to the fixed oscillator-length decay
`exp(-r/a_ho)`. Evaluating old checkpoints without compatibility mode can roughly
double N=20, ω=0.01 energies. Use `--cusp-len-mode legacy` for pre-metadata
checkpoints when comparing against the existing result table.

Rapid diagnostics were run with 30 epochs, small VMC probes, fixed low oversample,
and explicit candidate-count logging:

| Probe | Best quick VMC | Decision |
|-------|----------------|----------|
| A1 cap=2, N=20 ω=0.01 | 12.3006 (+100.13%) | Baseline |
| A2 cap=3, N=20 ω=0.01 | 11.7120 (+90.55%) | Cap=3 helps |
| A3 uncapped, N=20 ω=0.01 | 11.5982 (+88.70%) | Uncapped helps most; promote uncapped/cap=3 |
| B1/B2/B3, N=12 ω=0.01 | 2.4881-2.4890 (+0.58-0.62%) | High sample does not give a fast improvement beyond current best |
| C0 tiny BF, N=20 ω=1.0 | 158.2137 (+1.50%) | Worse than existing tiny-BF/baseline records |
| C1/C2 tiny BF, N=20 ω=0.1 | 31.7683 (+5.97%) | Does not beat Jastrow baseline (+5.533%) in quick probe |
| D1 N=16 ω=0.1 from N=12 | VMC 21.0779 (no ref) | Usable bridge checkpoint, but raw ESS remains poor |
| D2 N=20 ω=0.1 from N=16 | 40.6798 (+35.70%) | N-curriculum helps versus direct transfer but not enough |
| D2b N=20 ω=0.1 direct from N=12 | 59.9173 (+99.87%) | Confirms direct N12→N20 jump is bad |
| F1 N=12 ω=0.001 post-fix | 0.5838 (+13.28%) | Big improvement vs stale +72%, but not solved |

Immediate next actions:

1. Do not continue the original B high-sample ladder; the quick probes all land
   in the same VMC band.
2. Promote Experiment A with uncapped or cap=3 only, and use a real N=20 ω=0.01
   checkpoint plus legacy cusp compatibility when comparing to the existing table.
3. Do not promote tiny BF at N=20 yet; use a frozen-Jastrow BF-only stage or a much
   smaller BF learning rate before spending a long run.
4. Treat N-curriculum as helpful but insufficient by itself. It needs a better
   proposal/ESS fix before D2/E can be meaningful.
5. Continue F from the post-fix checkpoint with better ESS handling; it is the
   most promising near-term cell because the quick VMC is already down to +13%.

---

## Technical Analysis Update — 2026-04-26

### What was broken operationally

The original stage-1 launcher was not a diagnostic launcher. It combined long
epoch counts, high `n_coll`, heavy periodic VMC, and final large VMC evaluations
before any variant had earned promotion. That is why it was drifting toward
20-hour wall-clock behavior. The corrected workflow is:

1. Run short collocation segments with `--n-eval 0`, `--vmc-every 20-50`,
   `--vmc-n 2000-4000`, and `--max-epoch-seconds`.
2. Promote only if small VMC probes improve within the first 1-3 probes.
3. Run a moderate confirmation VMC only after a checkpoint wins.
4. Reserve completion-scale jobs for variants that have already passed the above
   gates.

The implementation now supports this workflow better:

- `scripts/launch_2026_04_26_grand_plan_stage1.sh` refuses to run unless
  `ALLOW_LONG_STAGE1=1` is set.
- `scripts/launch_2026_04_26_grand_plan_rapid_triage.sh` provides small gated
  suites for A/B/C/D/E/F.
- `scripts/summarize_rapid_triage.py` summarizes multiple output directories and
  surfaces best VMC probe values.
- `src/run_weak_form.py` records candidate counts and, for new runs, reports the
  best VMC probe instead of saving only `nan` when `--n-eval 0`.

### Current best F trajectory

| Stage | Checkpoint / log | VMC energy | Error vs surrogate |
|-------|------------------|------------|--------------------|
| Historical pre-fix | `camp2_n12w0001_cascade.log` | 0.888340 | +72.371% |
| Rapid post-fix | `rapidF1_n12_w0001_postfix` | 0.5838 | +13.28% |
| First refinement | `refineF_simple_150.pt` | ~0.55375 | +7.45% |
| Low-LR ov16 polish | `polishF_ov16_lowlr.pt` | 0.53507 | +3.824% |
| Stage-2 selection probe | `polishF2_temp075_lr6_ov16.pt` | 0.53103 | +3.040% |
| Confirmed eval | `confirmF2_temp075_12k.pt` | 0.531987 ± 0.000224 | +3.225% |

This is the strongest concrete improvement from the day: N=12, ω=0.001 moved
from an invalid +72% cell to a confirmed +3.225% cell without a long run.

### Negative or weak branches

- F ShellFlow proposal continuation was worse (`0.5976`, +15.96%). Do not promote
  ShellFlow for this N=12, ω=0.001 branch without a new reason.
- A mistaken direct-term sweep with `β=0.1` landed around `0.5425-0.5431`
  (+5.26% to +5.37%), worse than the clean `β=0` low-LR continuation.
- Increasing F to `n_coll=2048` at the same candidate count did not beat the
  `n_coll=1024, oversample=16` path and roughly doubled epoch cost.
- N=20, ω=0.01 cap removal helps (`cap=2`: +100.13%, `uncapped`: +88.70%) but
  does not solve the cell.
- Restoring the old N=20 ShellFlow proposal reduced PSIS-khat and candidate
  pressure but did not improve VMC energy enough to matter.

### Technical interpretation

The successful F continuation is not evidence that importance sampling is fixed.
It works despite bad raw ESS. The effective ESS printed during training is
inflated by tempering/clipping and is useful as an optimizer-stability signal, but
the raw ESS and PSIS-khat show that candidate overlap is still extremely poor.

The printed collocation epoch energy is also not the promotion metric whenever
resampling weights are tempered or clipped. It is an optimizer diagnostic under a
regularized sample distribution. Small VMC probes are the correct decision signal.

### Next promotion rule

For F, continue only from `polishF2_temp075_lr6_ov16.pt` or the confirmed
checkpoint. Use `--direct-weight 0.0`, `--oversample 16`, `--n-coll 1024`,
`--resample-logw-clip-q 0.995`, and small learning rates. If the next two VMC
probes do not beat the confirmed +3.225% by at least ~0.3 percentage points, stop
polishing and move to the sampler problem.

For N=20, do not return to long stage-1 jobs yet. The current evidence says:
uncapping helps, N-curriculum helps relative to a direct jump, but raw ESS remains
too poor and the energy gap is still large. The next N=20 work should be a sampler
or objective diagnostic, not another completion-scale replay of A/B.
