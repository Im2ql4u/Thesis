# Plan: Consistency Campaign — Remaining Phases (3.5 through 7)

Date: 2026-03-30
Status: confirmed

## Objective

Close the gap from the current best N=6 heavy-VMC errors (+0.158% at ω=0.1, +0.212% at ω=0.001) toward the ≤0.01% target, scale the winning recipe to N=12 and N=20, and produce a multi-seed consistency matrix that either demonstrates the target or honestly characterizes the achievable boundary.

Success condition: A defensible, seed-wise consistency statement across the target N×ω matrix, where "defensible" means every number is heavy-VMC validated with understood provenance.

## Context

Phases 0–3 of the Consistency Campaign are complete. Key findings:

**What the numbers say (heavy-VMC validated, 30k samples):**

| Phase | Tag | N | ω | err (%) | Recipe |
|---|---|---:|---:|---:|---|
| P3 | p3b_fdcolloc_n6w01 | 6 | 0.1 | +0.158 | FD-colloc + proximal, 1500ep |
| P3 | p3a_nogate_n6w001 | 6 | 0.001 | +0.212 | no-gate, oversample=32, 1133ep |
| P3 | p3a_nogate_n6w01 | 6 | 0.1 | +0.328 | no-gate, oversample=32, 370ep |
| P3 | p3c_adam_n6w01 | 6 | 0.1 | +0.383 | patient Adam, LR warmup, 486ep |
| P2D | diag_scratch_n6w01 | 6 | 0.1 | +0.414 | scratch init, 855ep |
| P2D | diag_xfer_n6w01 | 6 | 0.1 | +0.590 | transfer from bf_ctnn_vcycle, 467ep |
| P3 | p3c_adam_n6w001 | 6 | 0.001 | +1.890 | patient Adam, LR warmup, 573ep |

**Historical bests (pre-campaign, from COLLOCATION_BEST_RESULTS.md):**

| N | ω | Best err (%) | Method |
|---:|---:|---:|---|
| 6 | 1.0 | +0.002 | best_probe, regime campaign |
| 6 | 0.5 | +0.002 | best, SR sweep |
| 6 | 0.1 | +0.091 | final, transfer campaign |
| 6 | 0.01 | +0.193 | best_probe, regime campaign |
| 6 | 0.001 | +0.334 | best_probe, regime campaign |
| 12 | 1.0 | +0.018 | final, long campaign |
| 12 | 0.5 | +0.028 | final, long campaign |
| 12 | 0.1 | +0.122 | final, transfer campaign |

**Phase 1 reproducibility finding:** None of 8 multi-seed continuation runs met ≤0.01%. Best phase 1 result was N=12 ω=1.0 at +0.051%. The historical +0.002% results were not reproduced under multi-seed conditions.

**Established hypotheses:**
- **H4 (ESS-gate deadlock):** confirmed strongest — removing gate enabled ω=0.001 improvement from +2.158% → +0.212%
- **H9 (transfer basin):** scratch slightly beats transfer at ω=0.1 (+0.414% vs +0.590%)
- **H5 (REINFORCE SNR collapse):** weaker than expected
- FD-colloc consistently outperforms REINFORCE at ω=0.1 by a small margin

**Known bugs:**
- Rolling-best checkpoint (`_best.pt`) missing metadata (omega, n_elec, mode, e_dmc, seed, tag). All `_best.pt` heavy-VMC evaluations ran at wrong omega, producing nonsense. Lines 1036-1042 of `src/run_weak_form.py`.

**Early termination:** p3a_nogate_n6w01 stopped at 370/2000 epochs, p3c_adam_n6w01 at 486/2000 — likely hit patience=300 threshold. Suggests these recipes converge quickly but may need longer patience or LR decay to push further.

## Approach

The original Phases 4–6 (scale winning recipe, low-omega push at high N, final matrix) assumed Phase 3 would produce a recipe that reaches ≤0.01% at N=6. That did not happen — the gap is 15–20×. Scaling a recipe that is 15× away from target at N=6 to N=12 would amplify the problem, not solve it.

The revised plan inserts a bugfix/recovery phase (3.5), a recipe deepening phase at N=6 (4), and a full N=6 grid consolidation (5) before scaling. Phases 6–7 remain scaling + final matrix but now have explicit go/no-go gates with realistic fallback targets.

**Regime-specific winning recipes going forward:**
- **ω ≥ 0.1:** FD-colloc + proximal + no-gate (hybrid of p3b + p3a signals)
- **ω ≤ 0.01:** no-gate + high oversample (p3a signal dominant)
- **ω ≥ 0.5:** Historical recipes already near target; re-verify with multi-seed

**Strategic constraint:** Do not scale to higher N until the N=6 recipe is either (a) within 3× of target across ω, or (b) the gap is understood and proven to be a fundamental method limitation at that ω. In case (b), the boundary itself is a valid thesis finding.

## Foundation checks (must pass before new code)

- [x] Data pipeline known-input check (Phase 0 — DMC lookup guards)
- [x] Split/leakage validity check (not applicable — no train/test split)
- [x] Baseline existence (Phase 1 — reproduction baseline established)
- [ ] Rolling-best checkpoint bug fixed (Step 1.1)
- [ ] Re-evaluated `_best.pt` files confirm whether mid-training states are better (Step 1.3)

## Scope

**In scope:**
- Fix rolling-best checkpoint bug
- Extended training runs at N=6 for the two winning recipes
- Full N=6 ω grid coverage with multi-seed
- Scale to N=12 with the consolidated recipe
- N=20 boundary characterization
- Final consistency matrix  

**Out of scope:**
- Architecture changes (backflow hidden size, new network types)
- New sampling schemes (MCMC, normalizing flows)
- SR/natural-gradient exploration (decision: REINFORCE/Adam-only for low omega, per DECISIONS.md 2026-03-26)
- N=2 results (trivially easy, not thesis-relevant)

---

## Steps

### Phase 3.5 — Bugfix and Recovery

#### Step 1.1 — Fix rolling-best checkpoint metadata

**What:** Add `mode`, `n_elec`, `omega`, `e_dmc`, `seed`, `tag` to the `_bst` dict in the rolling-best checkpoint save logic. These values are already available in the `train_weak_form()` scope.
**Files:** `src/run_weak_form.py` lines 1036-1042
**Acceptance check:** Save a dummy best checkpoint, load it, verify all keys present: `python3 -c "import torch; d=torch.load('test_best.pt', map_location='cpu'); assert all(k in d for k in ['omega','n_elec','mode','e_dmc','seed','tag'])"`
**Risk:** None — pure bugfix.

#### Step 1.2 — Investigate early termination

**What:** Check the Phase 3 epoch JSONL files for p3a_nogate_n6w01 (370 ep) and p3c_adam_n6w01 (486 ep). Determine if these stopped due to the `--patience` parameter or an error. Extract the training curves and identify whether they were still improving when they stopped.
**Files:** `outputs/consistency_campaign/phase3/p3a_nogate_n6w01_epochs.jsonl`, `outputs/consistency_campaign/phase3/p3c_adam_n6w01_epochs.jsonl`
**Acceptance check:** A clear statement of why each run stopped and whether longer training is warranted.
**Risk:** If the runs hit patience correctly, extending them may not help without LR decay.

#### Step 1.3 — Re-evaluate all `_best.pt` checkpoints with correct metadata

**What:** After Step 1.1, either re-run the buggy checkpoints through the evaluator with manual omega/mode overrides, or patch the saved files to add the missing metadata, then run `eval_checkpoint_matrix.py`. The goal is to know whether the rolling-window best states are materially better than the final states.
**Files:** `scripts/eval_checkpoint_matrix.py`, `results/arch_colloc/*_best.pt`
**Acceptance check:** A comparison table: for each run, final-checkpoint err vs best-checkpoint err. Quantified gap.
**Risk:** The best checkpoints may not be significantly better (the regression might be modest at N=6). If so, this doesn't change the strategy.

#### Step 1.4 — Commit and update report

**What:** Commit the bugfix. Update `CONSISTENCY_CAMPAIGN_REPORT_2026-03-29.md` with Phase 2D and Phase 3 findings, corrected `_best.pt` evaluations, and the early-termination investigation.
**Files:** `src/run_weak_form.py`, `outputs/consistency_campaign/CONSISTENCY_CAMPAIGN_REPORT_2026-03-29.md`
**Acceptance check:** `git diff --stat` shows exactly the expected files. Campaign report has Phase 2D + Phase 3 sections with heavy-VMC validated numbers.
**Risk:** None.

---

### Phase 4 — Deepen N=6 Winning Recipes

**Goal:** Push N=6 ω=0.1 from +0.158% toward ≤0.05%, and N=6 ω=0.001 from +0.212% toward ≤0.1%. These are intermediate targets that de-risk scaling.

**Go/no-go gate to enter Phase 4:** Phase 3.5 complete, `_best.pt` evaluations are valid, early termination understood.

#### Step 2.1 — Design extended-training recipe matrix

**What:** Design a small recipe matrix for extended N=6 runs at ω=0.1 and ω=0.001. Each recipe variant targets one hypothesis about why the current best plateaued. Candidates:

| ID | Regime | Recipe change vs Phase 3 best | Hypothesis |
|---|---|---|---|
| 4a | ω=0.1 | FD-colloc, n-coll=8192, 4000ep, cosine LR decay to 1e-5 | More samples and LR schedule break plateau |
| 4b | ω=0.1 | FD-colloc, n-coll=8192, resume from p3b _best.pt, 3000ep | Best intermediate state is better starting point |
| 4c | ω=0.001 | No-gate, oversample=64, 4000ep, patience=1000 | More patience + higher oversample pushes further |
| 4d | ω=0.001 | No-gate, oversample=64, resume from p3a _best.pt, 3000ep | Best intermediate state + more budget |

**Files:** New launcher script `scripts/launch_consistency_phase4.sh`
**Acceptance check:** 4 experiments defined, each with concrete command lines, GPU assignments, and expected wall-clock time.
**Risk:** Extended runs at 4000 epochs will take 4-6 hours each. If LR decay doesn't help, we've spent GPU-hours without progress.

#### Step 2.2 — Run Phase 4 experiments

**What:** Launch the 4 experiments in tmux across 4 GPUs. Include `--save-best-window 20` (now with the fix from 1.1).
**Files:** Launch script, tmux session `consistency_p4`
**Acceptance check:** All 4 summary JSONs exist with `epochs_logged ≥ 3000` (or converged via patience). No NaN in final energy.
**Risk:** GPU contention from other users on GPU 0.

#### Step 2.3 — Heavy-VMC evaluation of Phase 4

**What:** Run `eval_checkpoint_matrix.py` on all 8 Phase 4 checkpoints (4 final + 4 _best.pt).
**Files:** `scripts/eval_checkpoint_matrix.py`
**Acceptance check:** Comparison table with Phase 3 → Phase 4 deltas for both final and best checkpoints. At least one of:
- ω=0.1 drops below +0.10%
- ω=0.001 drops below +0.15%
If neither is met, this is a meaningful negative finding that needs interpretation before proceeding.
**Risk:** The current recipe may have reached a capacity or optimization floor. See "Phase 4 failure protocol" below.

#### Phase 4 failure protocol

If Step 2.3 shows no meaningful improvement over Phase 3:
1. Check whether the training curve is flat (capacity limit) or noisy (optimization limit)
2. If capacity: the BF+Jastrow ansatz at N=6 cannot reach ≤0.01% at this omega. This is a thesis finding.
3. If optimization: investigate gradient quality (var_EL, SNR, ESS trajectory) and consider whether SR is needed at ω=0.1 (reversing the REINFORCE-only decision, with evidence).
4. Present the finding and ask for direction before continuing.

---

### Phase 5 — N=6 Full Grid Consolidation

**Goal:** Cover the full ω grid at N=6 with the winning recipe from Phase 4, multi-seed ×3 per regime. Produce the N=6 evidence bundle.

**Go/no-go gate to enter Phase 5:** Phase 4 showed improvement at ω=0.1 and/or ω=0.001 (even if the ≤0.01% target is not reached).

#### Step 3.1 — Identify the consolidated recipe per regime

**What:** Based on Phase 4 results, define one recipe per omega band:
- **ω ∈ {0.5, 1.0}:** Existing historical recipe (REINFORCE + ESS gate, the "easy" regime). These were near-target in historical bests.
- **ω = 0.1:** Phase 4 winner (likely FD-colloc variant)
- **ω ∈ {0.001, 0.01}:** Phase 4 winner (likely no-gate variant)

**Files:** Recipe specification document or config files in `scripts/`
**Acceptance check:** One clearly specified recipe per omega, with exact command-line arguments.
**Risk:** The best recipe may differ between ω=0.001 and ω=0.01. If so, test both quickly at ω=0.01 before committing.

#### Step 3.2 — Multi-seed N=6 campaign (3 seeds × 5 omega values)

**What:** Run 15 experiments (5 omega values × 3 seeds {42, 11, 77}). Group into GPU batches of 7 (all available GPUs). Batch 1: 7 runs, Batch 2: 7 runs, Batch 3: 1 run (or overlap).
**Files:** `scripts/launch_consistency_phase5.sh`
**Acceptance check:** All 15 summary JSONs exist. For each omega, the 3-seed standard deviation is computed as a consistency measure.
**Risk:** 15 runs × 4h = ~10h if fully parallelized in 3 batches. Realistic wall clock: ~12-14h with evaluation.

#### Step 3.3 — Heavy-VMC evaluation of full N=6 grid

**What:** Evaluate all 30 checkpoints (15 final + 15 best). Compute per-omega statistics: mean, std, min, max across seeds.
**Files:** `scripts/eval_checkpoint_matrix.py`
**Acceptance check:** A summary table:

| ω | Seed-mean err (%) | Seed-std (%) | Best single err (%) | 3/3 below X%? |
|---|---|---|---|---|

Where X is the declared target at each omega (either ≤0.01% or the revised realistic target from Phase 4).
**Risk:** High seed-to-seed variance would indicate the training process is not robust. This is the single most important measurement in the campaign.

#### Step 3.4 — N=6 evidence bundle and target revision

**What:** Based on Step 3.3, declare one of:
- **Target met:** ≤0.01% achieved 3/3 seeds at omega X. This is a strong thesis claim.
- **Consistent but above target:** All 3 seeds converge to similar errors, but above 0.01%. State the actually-achievable bound (e.g., "consistently ≤0.2% at ω=0.1"). This is still a useful finding.
- **Inconsistent:** Seed variance is large relative to mean. The method is not stable at this regime. This constrains which regimes can appear in the final matrix.

Archive as `outputs/consistency_campaign/phase5/n6_evidence_bundle.json`.
**Files:** Evidence bundle JSON + updated campaign report
**Acceptance check:** The evidence bundle exists and has been committed.
**Risk:** None — this is an assessment step.

---

### Phase 6 — Scale to N=12

**Goal:** Transfer the N=6 winning recipe to N=12 and establish whether the method's accuracy scales with N.

**Go/no-go gate to enter Phase 6:** Phase 5 complete. At least ω ∈ {0.1, 0.5, 1.0} have consistent multi-seed results at N=6 (even if above the ≤0.01% target).

#### Step 4.1 — N=12 recipe adaptation

**What:** Adapt the N=6 recipe for N=12:
- Increase batch size or n-coll to compensate for N² scaling of electron interactions
- Adjust sigma_fs auto-widening for 24D (N=12, d=2)
- Consider memory constraints (11GB RTX 3080 with BF at N=12: bf-hidden may need reduction)
- Warm-start from best N=12 existing checkpoints where available (e.g., `long_n12w1` at +0.018%)

**Files:** `scripts/launch_consistency_phase6.sh`
**Acceptance check:** Recipe spec with verified GPU memory fit (≤10GB).
**Risk:** N=12 backflow may OOM at n-coll=8192. Plan for n-coll=4096 fallback.

#### Step 4.2 — N=12 multi-seed campaign

**What:** Run the adapted recipe across ω ∈ {0.1, 0.5, 1.0} × 3 seeds = 9 experiments. Deprioritize ω≤0.01 at N=12 initially — only attempt if ω=0.1 succeeds.
**Files:** Launch script, tmux session
**Acceptance check:** All 9 summary JSONs. Per-omega seed statistics.
**Risk:** N=12 runs are 2-3× slower than N=6 per epoch. Budget ~8h per run, ~16-24h total for 2 batches.

#### Step 4.3 — N=12 heavy-VMC evaluation

**What:** Evaluate all N=12 checkpoints. Compare with historical bests from COLLOCATION_BEST_RESULTS.md.
**Files:** `scripts/eval_checkpoint_matrix.py`  
**Acceptance check:** Comparison table: Phase 6 results vs historical best at each (N=12, ω) point. Improvement or regression quantified.
**Risk:** N=12 ω=0.1 historical best is +0.122%. If Phase 6 cannot improve on this, the recipe may not be transferring well.

---

### Phase 7 — N=20 Boundary Characterization and Final Matrix

**Goal:** Attempt N=20 (Jastrow-only, per DECISIONS.md 2026-03-19), characterize the achievable boundary, and produce the final consistency matrix for the thesis.

**Go/no-go gate to enter Phase 7:** Phase 6 complete. N=12 multi-seed results exist for at least ω ∈ {0.5, 1.0}.

#### Step 5.1 — N=20 feasibility runs

**What:** Run the Jastrow-only recipe at N=20 ω=1.0 (the easiest N=20 regime). Use bf-hidden=64, bf-layers=2, micro-batch=128, n-coll=1024 (memory-constrained).
**Files:** New launcher script
**Acceptance check:** Run completes without OOM. Heavy-VMC error quantified. Compare with historical best (+32.889% at ω=1.0 — aiming for significant improvement).
**Risk:** N=20 Jastrow-only historically reached +1.32% at ω=1.0 with polishing. If the winning recipe doesn't help here, N=20 may need different treatment.

#### Step 5.2 — N=20 extended attempts (if feasible)

**What:** If Step 5.1 shows reasonable results, extend to ω=0.5 and ω=0.1. Use 1-2 seeds only (not full 3-seed matrix) to characterize feasibility.
**Files:** Launcher script extension
**Acceptance check:** Error quantified at each attempted omega. Clear boundary identification.
**Risk:** N=20 ω=0.1 historically at +5.5%. Getting below 1% would be a significant achievement; ≤0.01% is almost certainly not possible with current architecture.

#### Step 5.3 — Compile final consistency matrix

**What:** Combine all evidence into the final matrix:

| N | ω | Mean err (%) | Std err (%) | Best err (%) | Seeds | Status |
|---|---|---|---|---|---|---|
| 6 | 1.0 | ? | ? | ? | 3/3 | ? |
| 6 | 0.5 | ? | ? | ? | 3/3 | ? |
| ... | ... | ... | ... | ... | ... | ... |

Status categories:
- **consistent:** 3/3 seeds below threshold
- **achievable:** best seed below threshold, others close
- **boundary:** method cannot reliably reach here with current architecture

**Files:** `outputs/consistency_campaign/FINAL_CONSISTENCY_MATRIX.md`, updated campaign report
**Acceptance check:** The matrix is complete, all numbers are heavy-VMC validated, every status has a link to the evidence checkpoint/eval.
**Risk:** None — this is the final synthesis.

#### Step 5.4 — Session close and archival

**What:** Final git tag `result/2026-XX-XX-consistency-campaign-final`. Update SESSION_LOG.md, JOURNAL.md, DECISIONS.md. Archive campaign artifacts.
**Files:** All log files
**Acceptance check:** `git tag` shows the result tag. Campaign report has a "Final Summary" section.
**Risk:** None.

---

## Risks and mitigations

- **The ≤0.01% target may be unachievable at low omega with current architecture:** Mitigation — Phase 4 failure protocol defines how to detect this early and reframe the thesis finding as "consistent bound at X%" rather than "≤0.01% everywhere." The achievable boundary is scientifically more honest than a cherry-picked single-seed number below the threshold.

- **GPU contention (GPU 0 occupied by fmwestby):** Mitigation — plan all campaigns for GPUs 1-7 only (7 GPUs). If more GPUs become unavailable, reduce seeds from 3 to 2 in early phases (can re-run missing seed later).

- **Extended runs (4000 epochs) may not improve over 1500-2000 epoch runs:** Mitigation — include cosine LR decay as the primary intervention, not just more epochs. LR decay is the most likely mechanism to break a plateau.

- **Rolling-best checkpoints may not be significantly better than finals:** Mitigation — this is a testable hypothesis (Step 1.3). If true, the regression problem was smaller than the epoch-trace suggested, and we can simplify by just using final checkpoints.

- **N=12 may regress from historical bests under the new recipe:** Mitigation — warm-start from existing good N=12 checkpoints rather than training from scratch. Historical N=12 ω=1.0 at +0.018% is already near target.

## Success criteria

**Minimum viable outcome (thesis-defensible):**
- N=6: Multi-seed consistency statement at ω ∈ {0.1, 0.5, 1.0} with quantified bounds
- N=12: At least 1-seed heavy-VMC at ω ∈ {0.5, 1.0}
- Honest boundary characterization for regimes where ≤0.01% is not achieved

**Target outcome:**
- N=6: ≤0.05% consistent across 3/3 seeds at ω ∈ {0.1, 0.5, 1.0}
- N=6: ≤0.3% consistent at ω=0.001
- N=12: ≤0.1% at ω ∈ {0.5, 1.0} (multi-seed)
- N=20: Below +2% at ω=1.0 (single seed, boundary finding)

**Stretch outcome:**
- Any (N, ω) point with ≤0.01% at 3/3 seeds
- N=6 full grid with ≤0.1% everywhere

## Current State

**Active step:** Step 3.1 — ω=0.01 bridge probe running in tmux (`consistency_p5_probe`); full Phase 5 launch is prepared and waiting on that recipe choice.
**Last evidence:** `outputs/consistency_campaign/phase4/eval_matched_20260401_075211.json` confirmed the Phase 5 gate; `scripts/launch_consistency_phase5_w001_probe.sh` launched `p5probe_fd_n6w001_s42` and `p5probe_ng_n6w001_s42` at 2026-04-01 08:03 on GPUs 1/2 in tmux.
**Current risk:** the ω=0.01 bridge remains the only unresolved N=6 recipe decision; launching the full 15-run matrix before that probe resolves would mix incompatible low-omega assumptions.
**Next action:** let the probe finish (~1.3h), compare heavy-VMC at ω=0.01, launch Phase 5 in tmux with `LOW_OMEGA_RECIPE=<winner>`, then run the prepared Phase 5 evaluation and keep Phase 6 queued.
**Blockers:** None.
