# Journal

Research journal for this project. Each entry documents an experiment, a significant result, or a meaningful shift in understanding. Entries are dated and cumulative — this is a scientific record, not a changelog.

The model reads this to understand what has been tried, what worked, what failed, and what remains open. Write entries as if they will be read by a technically capable person who has not been following the project day-to-day.

---

## Format

```
### [YYYY-MM-DD] — <experiment or event title>

**Motivation:** <why we ran this — what question we were trying to answer>
**Method:** <what was done — concisely but precisely>
**Results:** <what the outputs actually showed — numbers where relevant>
**Interpretation:** <what this means — not just what happened but what it implies>
**Caveats:** <what might be wrong about this interpretation, what was not controlled for>
**Output reference:** <path to result files, e.g. results/2024-03-15_run01/>
**Next question:** <what this result makes us want to investigate next>
```

---

## Journal

### [2026-07-06b] — Convergence probe: collocation floors at var(E_L)≈0.30 UNDER THREE OPTIMIZERS (Adam, diag-Fisher, CG-SR); my honest heavy-VMC (+0.21–0.31%) is ~7–10× the thesis honest-final (+0.031%) — the wavefunctions are NOT converged, and the gap is not an optimizer-tuning problem

**Motivation:** The user challenged the +0.4% repro: "not good enough — if that's what we replicate we'd
declare the method unstable; are we sure the wavefunctions are fully converged?" Test convergence directly.
**Method:** Resumed the N=6 ω=1 from-scratch checkpoint (var(E_L)=0.30, +0.39%) and continued three ways,
each with LR annealing + rolling-best + heavy 30k-VMC eval: (A) Adam +REINFORCE 3000 ep; (B) diagonal
natural-gradient 1500 ep; (C) FULL CG-SR polish 500 ep (`--sr-mode cg`, the thesis's actual endgame
optimiser). Watched whether var(E_L) breaks the ≈0.30 level toward a converged eigenstate (~0.02).
**Results:**
- **The initial run was under-converged (still descending), BUT continuing reveals a hard floor.** All
  three optimisers PLATEAU at var(E_L)≈0.28–0.38 with energy BOUNCING at +0.1–0.4% (heavy-VMC): Adam
  early-stopped ep 645 (+0.311%); diag-Fisher bounced (~+0.2–0.4%); CG-SR bounced +0.14→+0.83% and
  early-stopped ep 394 (+0.212%). var(E_L) never approached the ~0.02 of a tight eigenstate.
- **best-probe vs honest gap reproduced:** each run's rolling-best (lucky) probe read +0.12–0.14% while
  the honest 30k heavy-VMC read +0.21–0.31%. Confirms best_probe over-selects on a high-variance state.
- **I did NOT reproduce the thesis honest number:** thesis `long_n6w1` FINAL (heavy VMC) = +0.031%; my
  best honest = +0.212% (CG-SR) — ~7× worse. So the thesis reached a genuinely tighter state than mine.
**Interpretation — two things established, one fork open:**
- **(established) var(E_L)≈0.30 is optimiser-invariant** → the plateau is NOT a wrong-optimiser or
  under-training artifact. Ruled out: architecture (this vcycle IS the faithful `bf_ctnn_vcycle` stack,
  not the drifted PINN); optimiser (3 tried).
- **(established) the +0.002% headline is best_probe selection bias** on a var≈0.3 state (my best-probe
  0.12–0.14% vs honest 0.21–0.31% is the same mechanism, milder because fewer probes). The honest thesis
  metric is +0.031%, and I'm ~7× above even that.
- **(fork) the remaining gap to +0.031% is EITHER the collocation MEASURE (fixed-proposal importance
  sampling can't resolve var below ~0.3 — same class of net reaches var~0.02 under |Ψ|²-VMC in the
  2026-06-15 analysis runs, strong circumstantial evidence for measure) OR PATH-DEPENDENCE (the thesis
  +0.031% came from a multi-stage continuation chain `bf_ctnn_vcycle→bf_joint_reinf_v3→bf_resume_lr_v1
  →bf_hardfocus_v1b` from a canonical init now LOST in the Jun-13 rewrite; a single from-scratch lineage
  may simply not reach the same basin).** These are not exclusive — likely both.
**So, honest answer to "are the wavefunctions converged": NO.** Not in variance (var 15× a tight
eigenstate), not reproducibly in energy (my honest +0.21% vs thesis +0.031%). This is a real
reproducibility + convergence concern for the collocation Q3 claims, surfaced before the thesis leans on
them further. It does NOT by itself prove the method is fundamentally unstable — one from-scratch lineage
flooring at 0.3 is not the same as the method being incapable — but it means the good numbers are
path/selection-dependent and not currently reproducible on the working tree.
**Decisive next tests (not yet run — a direction fork for the user):**
1. **Measure vs capacity (gold standard):** VMC-SR (`train_model_sr_energy`, |Ψ|²-Metropolis, SAME E_L
   reward) on the IDENTICAL net+checkpoint. If var 0.3→~0.02 → the collocation MEASURE is the limiter
   (clean Q3 result: collocation cannot resolve the eigenstate as tightly as VMC, ~15× var floor). If it
   also floors at 0.3 → ansatz capacity. (Needs a ~40-line driver reusing run_weak_form.setup + the
   builders; feasible, some debug risk.)
2. **Path-dependence:** reconstruct a continuation chain (staged ω-anneal / LR-stage resumes) and see if
   staged annealing breaks var 0.3 where single-lineage didn't.
**Caveats:** single seed per lineage; N=6 ω=1 only; the CG-SR polish used lr 2e-2 / diagonal-damping 1e-3
(not swept — a different CG-SR schedule might do better, but the point is 3 optimisers all plateau at the
SAME var, not that each is perfectly tuned); "same-class net reaches var~0.02 under VMC" is the analysis
ctnn_vcycle_big (80k) not the identical 25k net — test 1 removes this confound.
**Output reference:** [results/analysis/2026-07-06_colloc_repro/](../results/analysis/2026-07-06_colloc_repro/)
(contAdam_n6w1.log, natdiag_n6w1.log, cgsr_n6w1.log); checkpoints results/arch_colloc/{contAdam,natdiag,
cgsr}_n6w1{,_best}.pt.
**Next question:** run test 1 (measure vs capacity) to convert this from "not converged" into the
mechanism — is it the sampling measure or the ansatz — then decide whether collocation Q3 claims need
re-scoping around the honest (+0.03% at best, path-dependent) numbers.

### [2026-07-06] — Reproduce the thesis collocation trainer on today's code: N=6 ω=1 reaches +0.4% (recipe is SOUND); the "awfully far" energies are a NEW-trainer regression, not a broken paradigm

**Motivation:** The overnight collocation (new minimal `src/analysis/train.py::train_collocation_weak`)
was +1.0–1.7% at ω≥0.1 and DIVERGED at ω=0.01 — "awfully far" from the historically-reported
+0.002–0.19%. Before proposing fixes, settle the prior question: does the ACTUAL thesis trainer
(`src/run_weak_form.py`) still reach the reported range on today's code+env?
**Method:** Got `run_weak_form.py` running (env: `source /etc/profile.d/z00_lmod.sh; module load
PyTorch/2.1.2-foss-2023a-CUDA-12.1.1` — the debug-note `lmod.sh` path is stale; and `module load`
must NOT be piped or the eval'd env is lost to the subshell). Two faithful N=6 ω=1 runs, 800 epochs,
standard REINFORCE + ESS-adaptive resampling (n-coll 4096, oversample 10→24 on ESS floor 0.1,
clip-el 5, direct-weight 0.1, lr 5e-4/5e-5, warmup 20), heavy 30k-sample exact-VMC final eval:
(a) from scratch; (b) thesis backflow warm-started (`--init-bf` from `official_models/6p/w_10/
backflowCTNN.pt`, wrapped as bf_state).
**Results:**
- **From scratch: E=20.2381±0.0036, err=+0.391%.** BF-warm: E=20.2472±0.0036, err=+0.436%.
  var(E_L)≈0.30–0.35, ESS≈62%, k̂≈0.46–0.74 (healthy). Correct energy SCALE (≈20.24 vs DMC
  20.15932), sub-1%, no divergence.
- Reported thesis range is +0.002% (best_probe) to +0.03% (long from-scratch) — so today's quick
  recipe is ~10–15× worse than the headline but the SAME order and physically correct. BF warm-start
  gave no gain (its displacements were tuned to the thesis PINN Jastrow, not our from-scratch 25k one).
- Online training probe read +0.26% at the last epoch but heavy VMC gave +0.39–0.44% — the known
  probe-optimism gap (DECISIONS 2026-03-14: heavy VMC is authoritative).
**Interpretation:** The collocation paradigm is NOT broken and the trainer is sound — it lands at the
right energy from scratch. The overnight "awfully far" (+1% → divergence) is specifically the NEW
minimal `train_collocation_weak` (fixed single Gaussian, no resampling, weak-Rayleigh-only, no ESS
adaptation, no continuation), which costs ~2–3× at ω=1 and stability at ω=0.01 vs the real recipe.
This empirically confirms the 2026-07-06 diagnosis: a recipe regression, not a paradigm failure. The
remaining gap to the +0.002% headline is carried by the ingredients now MISSING from the working tree:
the canonical pretrained init `results/arch_colloc/bf_ctnn_vcycle.pt` (wiped in the Jun-13 origin
force-rewrite; `*.pt` gitignored), continuation chains, best-probe selection, and natural-gradient
CG-SR polish.
**Caveats:** A bit-exact reproduction is impossible now: (1) the canonical init is gone; (2) the
Jun-13 re-sync also DRIFTED the Jastrow arch — `build_jastrow_model()` builds a 25,562-param net,
but the thesis `f_netCTNN.pt` is a 53,933-param `PINN` (φ/ψ/g ScaledPINN), so the Jastrow can't be
warm-started from the thesis checkpoints (only the 182,403-param `CTNNBackflowNet` backflow matches
exactly). Runs are single-seed; no continuation cascade yet; ω=0.1/0.01 not yet run (need the ω=1
checkpoint as warm-start).
**Output reference:** [results/analysis/2026-07-06_colloc_repro/](../results/analysis/2026-07-06_colloc_repro/)
(repro_n6w1_scratch.log, repro_n6w1_bfwarm.log); checkpoints results/arch_colloc/repro_n6w1_{scratch,
bfwarm}.pt; plan [plans/2026-07-06_collocation-conditioning-diagnosis.md](../plans/2026-07-06_collocation-conditioning-diagnosis.md).
**Next question:** (1) push one run harder (natural-grad CG-SR polish + longer + best-probe) to see if
it closes to +0.03%; (2) cascade ω=1→0.1→0.01 to confirm the low-ω range and whether ω=0.01 stays
stable under the real recipe (it should — the divergence was the minimal trainer); (3) then port the
winning ingredients (resampling + ESS adaptation + strong-form/zero-var reward) back into the analysis
trainer so the Q3 dual-track is a fair fight.

### [2026-07-03] — Overnight campaign: arch ablation (Group A) is clean; collocation UNDER-CONVERGED (paradigm/N-scaling confounded); and the paradigm-backflow thread is RESOLVED — it's ARCHITECTURE, not paradigm

**Motivation:** First big overnight run (13 chains, GPUs 1-6): Group A architecture ablation (VMC,
{ctnn,deepset}×{bf,nobf}×2 seeds), Group B paradigm=collocation, Group C optimizer, Group D N=12
scaling. Then settle the headline paradigm-backflow thread by loading a thesis checkpoint.
**Method:** `scripts/orchestrate_overnight.py` (fixed a Group-A optimizer=sr divergence bug via the
smoke), `scripts/analyze_overnight.py` -> master.csv (energy/var/d_eff/backflow-rank/message-kinetic
per checkpoint). Then reconstructed + loaded the thesis N=6 ω=0.001 checkpoint.
**Results — Group A (VMC, converged, TRUSTWORTHY):**
- CTNN compression gap holds matched-recipe: d_eff ctnn 1.2-2.5 vs deepset 3.0-4.3 (~2.5× at ω=1).
- var ordering ctnn-bf < ctnn-nobf < deepset-bf < deepset-nobf (MP and backflow both improve quality).
- Message ablation (CTNN): zeroing messages costs +0.67-0.81 kinetic at ω=1 (consistent with M0).
- Backflow rank collapses at low ω for BOTH arches (6-8 → 1).
- SURPRISE: the no-backflow arms reach near-DMC ENERGY at N=6 (ctnn-nobf ω=0.01 = -0.015%) — so the
  backflow's ENERGY contribution at N=6 is small; its role is variance/quality + Wigner lattice work,
  NOT energy. Softens the earlier "backflow needed for nodes" claim at N=6.
**Results — collocation (Groups B, D) DID NOT CONVERGE:** ctnn+bf collocation reached only +1.1%/+1.6%
at ω=1/0.1 and DIVERGED at ω=0.01 (err 5e5%, var 3e7) — the ESS-collapse (Q3) biting the TRAINING with
a single fixed Gaussian proposal. So the paradigm-internal-structure and N-scaling comparisons from this
run are NOT trustworthy (under-converged).
**Results — the paradigm-backflow thread RESOLVED (architecture, not paradigm):** loaded the thesis
N=6 ω=0.001 checkpoint -> f_net class='PINN' (φ/ψ/g ScaledPINN), backflow class='BackflowNet'
(CONVENTIONAL per-particle) -- a DIFFERENT architecture from my CTNN-V-cycle + message-passing backflow.
Loaded the conventional BackflowNet (clean load, 0 missing): its displacement rank = 6.23 at ω=0.001 and
11.26 at ω=1 -- HIGH at both, never collapsing. So "rank-10 (thesis) vs rank-1 (mine)" was
conventional-vs-message-passing backflow, NOT VMC-vs-collocation. The paradigm hypothesis is FALSE.
**Interpretation (the better finding):** the rank-1 collapse at Wigner is a MESSAGE-PASSING signature:
at the crystal the message-passing backflow discovers the optimal correction is a single coordinated
COLLECTIVE mode (rank-1) -- repositioning all electrons together -- while a conventional backflow,
lacking inter-particle communication, applies independent per-particle corrections (rank ~6). This fits
the "message passing = collective coordination" theme and is cleaner than the paradigm claim.
**Caveats:** conventional backflow rank measured on Gaussian samples (not its own |Ψ|²), so 6.23 is
approximate -- but clearly HIGH (not 1), which is the point; ω=1 rank 11.26 matches the thesis table.
Collocation non-convergence blocks the paradigm/N-scaling threads until the proposal is fixed. N=12
energy% in master.csv uses the wrong (N=6) reference -- ignore.
**Output reference:** [results/analysis/2026-07-02_overnight/](../results/analysis/2026-07-02_overnight/)
(master.csv, orchestrator.log, 34 checkpoints); thesis checkpoint results/models-Copy1/6p/w_00/.
**Next question:** fix the collocation recipe (adaptive/mixture proposal + continuation) to unblock the
Q3 collocation-vs-VMC and N-scaling threads; converged N=12 (warm-start VMC from collocation, or fixed
proposal) + N=12 references; debug collocation+SR (WIP).

### [2026-07-11] — RETRACTION: the "message-passing backflow" findings were never message passing; the CTNN backflow was DEAD (tanh saturation x COM projection)

**Two independent errors, both now fixed. The entries below marked [RETRACTED] are false.**

**Error 1 — mislabeling.** `backflow_arch` defaulted to `"conv"` and the `ctnn` option did not exist
until 2026-07-04. So EVERY campaign before that (Group A, the overnight run, the whole
"paradigm-backflow" thread) used the CONVENTIONAL `BackflowNet`. My headline reading —
"the rank-1 collapse at Wigner is a MESSAGE-PASSING signature ... the message-passing backflow
discovers a single coordinated COLLECTIVE mode" (entries 2026-07-02i and the paradigm entry above) —
was measured on a conventional per-particle backflow. There was no message passing in it.
**RETRACTED.** The rank-1-vs-rank-6/11 contrast is my-training-vs-thesis-training, not architecture.

**Error 2 — the CTNN backflow was dead the moment I did wire it up.** `CTNNBackflowNet.forward`
does `dx = tanh(dx_head(h_v))` (PINN.py:650) and then subtracts the per-particle mean to conserve
the centre of mass (PINN.py:669). My analysis trainer ran Adam at lr=3e-3, ONE param group, clip=5.
That drives the dx_head pre-activation to ~148; tanh saturates; every particle then gets the
*identical* +-1; and the zero-mean projection cancels identical values to EXACTLY zero. tanh' is
then 0, so no gradient ever returns: the backflow is dead and cannot recover. Measured live:

    [adam 00000] |dx|/ell=0.0058  tanh_sat=0.00   <- alive
    [adam 00100] |dx|/ell=0.0000  tanh_sat=1.00   <- fully saturated, dead

It dies in under 100 steps. The v1 PINN campaign (`results/analysis/2026-07-04_pinn_ansatz`) is
therefore INVALID: its CTNN arm had Delta_x identically zero, so "PINN+CTNN-bf vs PINN+conv-bf"
compared a backflow-less ansatz against a real one. That is also why conv appeared to *beat* ctnn.

**Root cause is my trainer, not the architecture.** The thesis pipeline (`run_weak_form.py`) never
hits this trap: lr 5e-4 on the backflow, 5e-5 on the Jastrow (separate param groups), grad_clip=1.0.
Those numbers keep the pre-activation inside tanh's linear region. Adopted verbatim; backflow sizing
also corrected to the thesis's msg_hidden/hidden=128 (I had 64). `backflow_health()` now prints
|dx|/ell and tanh_sat on every Adam and SR log line, so this failure can never again survive a run.

**What SURVIVES.** `Thesis/results_kernel.tex` is unaffected: its Q1 comparison holds the backflow
*identical and fixed* across the CTNN-Jastrow and DeepSet-Jastrow arms (a conventional backflow in
both), which is a valid controlled experiment and is described as such. The CTNN-Jastrow vs DeepSet
compression gap, the mode-naming, and the message-ablation results are all Jastrow-side and correctly
labeled. Only the *backflow*-side claims are retracted.

**Next:** rerun the PINN ansatz with the fixed trainer (`scripts/launch_pinn_ansatz_v2.sh`, into
`results/analysis/2026-07-11_pinn_ansatz_v2`; v1 is kept, not overwritten), then redo the CTNN-vs-
conventional *backflow* contrast on an ansatz whose backflow is actually alive.

### [2026-07-02i] [RETRACTED 2026-07-11 — conventional backflow, mislabeled as message-passing] — Phase M0 (T1.2): feature-rank ≠ tangent-d_eff (two distinct low-dims); backflow rank collapses 10→1 at Wigner (route-dependent, to verify)

**Motivation:** Are the correlator FEATURE rank (thesis r_eff≤3), the tangent d_eff (my work), and the
intrinsic dim the same low-dimensional object? And quantify the low-rank-correlator / high-rank-backflow
"two feature spaces" in one place.
**Method:** `scripts/run_dim_unification.py` — on the V-cycle CTNN across ω, capture the readout (f_head)
input feature Z; report r_eff(Z) (linear), TwoNN intrinsic dim ID(Z), tangent d_eff (QGT), and the
backflow displacement rank.
**Results:** r_eff(Z) ≈ 1.07 at every ω (nearly rank-1 — one collective readout coordinate; flatter
than the thesis's φ/ψ/g r_eff≤3, a DIFFERENT architecture); ID(Z) ≈ 6–8 (nonlinear — Z is a wiggly ~1D
curve); tangent d_eff = 1.21→1.81→3.71 (rises toward Wigner). So **feature-rank (~1) and tangent-d_eff
(1.2–3.7) are NOT the same object**: the correlator reads out along ~1 direction while the parameters
can move the state along more — two distinct low-dimensionalities. Backflow rank = 10/12 (ω=1,0.1) but
**1/12 at ω=0.01** — collapses to a rank-1 collective displacement at Wigner, coherent with the
message-ablation (backflow does zero kinetic work at Wigner → a rank-1 lattice-repositioning, not a
high-rank per-particle correction).
**Interpretation:** answers "are the intrinsic dimensions the same" — no; the readout coordinate is ~1D
(collective), the tangent space is a few-D, the backflow is high-rank at weak coupling and collapses to
rank-1 collective at Wigner. Unifies with the backflow kinetic-switch (2026-07-02h).
**Verification (same day):** the backflow rank-collapse at Wigner is ROBUST — all 3 ω=0.01 seeds AND
the original cascade checkpoint give rank 1.00 (|Δx|/ℓ ≈ 0.045–0.057), vs rank ~10 at ω=1/0.1. So it is
seed- and route-robust WITHIN my (VMC/SR-trained) analysis checkpoints.
**NEW open thread (Q3 × mechanism):** this rank-1 (VMC/SR) contradicts the thesis backflow table
(rank≈10 at ω=0.001), whose checkpoints are COLLOCATION-trained. So the two *paradigms* may produce
different internal backflow structure at Wigner *at matched energy* — a much deeper collocation-vs-VMC
result than the energy tie (mirrors the N=2 dual-track "same Ψ" but suggests DIFFERENT internal
mechanism at N=6 Wigner). To nail: load an actual collocation checkpoint at ω=0.001 and measure its
backflow rank; confounded by ω=0.01-vs-0.001, so also check VMC at ω=0.001.
**Caveats:** ID(Z) TwoNN on 512 pts; V-cycle arch (not the φ/ψ/g ScaledPINN of the thesis
correlator-geom table); the paradigm claim is a hypothesis until a collocation checkpoint is measured.
**Output reference:** [results/analysis/2026-07-02_dim_unification/](../results/analysis/2026-07-02_dim_unification/).
**Next question:** measure a collocation-trained checkpoint's backflow rank (the paradigm thread); T1.3
decode the edge scalar beyond distance; then the decisive training ablations (M1) and SR-on-collocation (O).

### [2026-07-02h] — Phase M0 (Grand Mechanism Program): message passing is worth ~10% energy 100% KINETIC, confound untangled — and the BACKFLOW's kinetic role VANISHES at Wigner (new)

**Motivation:** New spine (DECISIONS 2026-07-02c): finish/explain the CTNN mechanism. Two no-training M0
probes: (T1.1) the confound-free message ablation — what does the graph compute, and is my "energies
tie" a backflow confound? (T1.5) is the V-cycle a scale separator (bottleneck=global, fine=local)?
**Method:** `scripts/run_message_ablation.py` — on trained N=6 CTNN checkpoints (ω=1 acc, ω=0.1 casc,
ω=0.01 ×3 seeds), zero the inter-particle rho_* maps (→ pairwise Jastrow) and/or disable the backflow;
evaluate E=<T>+<V_trap>+<V_Coul> with weak-form kinetic T=½|∇logΨ|² (no Laplacian) on common samples
from the full |Ψ|². 2×2 {msg on/off}×{bf on/off}. `scripts/run_scale_separation.py` — hook fine
(node/edge_embed) vs coarse (node/edge_down bottleneck), linear-probe vs global/local observables.
**Results — message ablation (decisive):**
- **Removing all message passing costs +7–11% energy, 100% KINETIC** at every ω (ω=1 +10.4%, ω=0.1
  +7.5%, ω=0.01 +11.4%; V_Coul change = 0 by construction — same samples). Reproduces the
  DIAGNOSTIC_SUMMARY "100% kinetic" claim robustly (smaller magnitude: my no-MP baseline keeps the
  pairwise Jastrow geometry).
- **Confound untangled:** at ω=1, Jastrow messages (ΔT +0.78) and backflow (ΔT +1.13) both contribute,
  ~additively (Jastrow-msg cost barely changes with bf on/off: 0.78 vs 0.95). So the earlier "CTNN ≈
  DeepSet energy" was a from-scratch DeepSet COMPENSATING, not messages being worthless. In the trained
  net, messages carry a real kinetic reduction.
- **NEW — division-of-labor shift:** the backflow's kinetic contribution collapses with ω:
  ΔT(bf) = +1.13 (ω=1) → +0.05 (ω=0.1) → **+0.00** (ω=0.01, all 3 seeds). At the Wigner crystal the
  backflow does ZERO kinetic work; it repositions electrons to lattice sites (orbital/nodal correction),
  not smoothing. Fresh mechanistic angle on the Fig-J force regime-switch: BF = kinetic-smoother at
  weak coupling, non-kinetic lattice corrector at Wigner. The Jastrow messages remain the kinetic
  workhorse across all ω.
**Results — scale separation (honest negative):** the "V-cycle bottleneck = global, fine = local"
hypothesis is NOT supported by read-out decoding. The fine edge feature is ~rank-1 = pair distance,
which already determines the breathing coordinate (Σr_ij² = N·Σr_i² in COM), so there is no clean
global/local split in the decode. Real bits: edge scalar identified as distance (confirms edge-rank
~1.4); message passing builds local density into nodes (R² 0.70→0.17, DECAYING toward Wigner = the
3-body collapse); bottleneck adds only modest extra shell structure. The decisive V-cycle test is
ABLATION (energy/d_eff cost), deferred to M1 — decoding is confounded.
**Interpretation:** The graph's benefit is a smoother (lower-curvature) wavefunction → lower kinetic →
this IS the var(E_L) discriminator (symptom↔mechanism connected at last). The Jastrow messages and the
backflow are complementary kinetic contributors at weak coupling; toward Wigner the backflow switches
off (kinetically) and becomes a lattice corrector while messages keep smoothing. My prior "energies
tie" is reconciled: ablation (can't compensate) vs from-scratch (can) measure different things.
**Caveats:** trained-net ablation = mechanistic reliance, NOT from-scratch capacity (the clean 2×2
from-scratch is M1). Magnitude is checkpoint-dependent (10% here vs 22–30% in DIAGNOSTIC_SUMMARY, older
checkpoints/baseline). Weak-form kinetic (mean = true kinetic by parts). N=6 only.
**Output reference:** [results/analysis/2026-07-02_message_ablation/](../results/analysis/2026-07-02_message_ablation/)
(ablation.csv, summary.json, fig), [results/analysis/2026-07-02_scale_separation/](../results/analysis/2026-07-02_scale_separation/).
**Next question (M0 cont.):** T1.2 feature-rank ↔ tangent-d_eff unification; T1.3 decode the edge scalar
beyond distance (spin/environment modulation); T1.7 extend force-alignment/attribution/3-body to
N=12,20 (does the backflow kinetic-switch move with N?). Then M1 ablations (bottleneck, from-scratch 2×2).

### [2026-07-02g] — Q3 dual-track (N=2 exact): VMC and weak-form collocation reach the SAME ground state (overlap² 0.9992), collocation at ~5× higher var(E_L)

**Motivation:** The roadmap's dual-track (§7): does collocation reach a different WAVEFUNCTION than
VMC, or the same state by a noisier route? Answer it at the exact N=2 anchor (step 2 / Q3).
**Method:** Added `train_collocation_weak` to `src/analysis/train.py` — weak-form (Rayleigh) collocation:
fixed Gaussian proposal q, importance weights w=|Ψ|²/q (clamped), exact self-normalised IS gradient of
the Rayleigh energy <½|∇logΨ|²+V>_w (direct term + REINFORCE score term); never samples |Ψ|².
`scripts/run_dual_track_N2.py` trains the identical CTNN (same init) via train_vmc_adam and
train_collocation_weak at N=2 ω=1 (900 steps each), compares overlap² vs exact and vs each other,
var(E_L), d_eff, and the learned Jastrow u(r).
**Results:**
- **Same state, both routes:** overlap²(VMC,exact)=0.99998, overlap²(colloc,exact)=0.99920,
  **overlap²(VMC,colloc)=0.99918**. Learned u(r) overlays exact for both. The paradigms do NOT find
  different wavefunctions — they find the same GS.
- **Quality differs:** var(E_L) VMC 1.1e-2 vs colloc 5.1e-2 (~5× higher — a marginally worse eigenstate).
  d_eff same (1.06 vs 1.13, as expected for the same state).
- **Weak-energy readout is unreliable:** collocation's weak-form E settled at 2.94 (below exact 3.0)
  even though the state is accurate — the zero-variance point made concrete (weak estimator unbiased in
  principle, noisy in practice; here ESS≈37%, benign at N=2).
**Interpretation:** The collocation–VMC distinction is not about WHICH state is found but the VARIANCE
at which it is resolved — closing the Q3 loop with the measure/Laplacian story: same manifold, same
solution, different estimator quality. The new collocation trainer converged cleanly at N=2 ω=1
(ESS ~37%), validating it against exact truth.
**Caveats:** N=2 ω=1 only (benign ESS); the interesting low-ω/high-N regime (where ESS collapses to
0.1%) would stress the two routes apart and is the natural extension. Single Gaussian proposal;
overlap² via self-normalised IS (finite-sample). The weak trainer is validated at N=2 but not yet at
scale.
**Output reference:** [results/analysis/2026-07-02_dual_track_N2/](../results/analysis/2026-07-02_dual_track_N2/)
(summary.json, fig_dual_track.png); `src/analysis/train.py::train_collocation_weak`,
`scripts/run_dual_track_N2.py`. Added the dual-track subsection to results_kernel.tex.
**Next question:** dual-track at low ω / N=6 (does ESS collapse pull the two states apart?); the honest
cross-paradigm energy reckoning at N=6 with heavy-VMC eval.

### [2026-07-02f] — Q2 low-ω SR-vs-Adam trend (N=2, ω=0.05/0.03): the advantage is MODEST, not a growing decisive win; ω=0.01 diverges with the light recipe

**Motivation:** Push the SR-vs-Adam onset (resolved at ω=0.1, ~1.5σ) to lower ω — does SR's advantage
grow decisively toward Wigner (the pre-registered Q2 prediction)?
**Method:** `scripts/exp_sr_mechanism.py --omega {0.05,0.03} --seeds 4` (N=2, cusp-on; common warm-up →
Adam-only vs SR-only branch from the same checkpoint; measured vs exact). ω=0.01 attempted but the
from-scratch lightweight recipe DIVERGES (E→−2e18, var→1e35) — needs the warm-started annealed cascade.
**Results (final-step energy err %, 4 seeds, ±1 s.d.; dist-to-exact):**
- ω=1:    Adam −0.004 [−0.11,+0.11] | SR +0.056 [+0.01,+0.11]   (tie); dist 0.0084 vs 0.0066
- ω=0.1:  Adam +0.163 [+0.06,+0.27] | SR −0.028 [−0.11,+0.05]   (SR wins ~1.5σ); dist 0.0390 vs 0.0298
- ω=0.05: Adam +0.102 [+0.03,+0.18] | SR −0.000 [−0.12,+0.12]   (bands OVERLAP); dist 0.0715 vs 0.0655
- ω=0.03: Adam +0.063 [+0.01,+0.11] | SR +0.064 [−0.05,+0.18]   (TIE); dist 0.0983 vs 0.0882
- **Energy edge is resolved only at ω=0.1**; at 0.05/0.03 the 4-seed energy bands overlap/tie.
- **Distance-to-exact: SR consistently ~10–20% lower at EVERY ω** (the less noisy signal).
**Interpretation:** The pre-registered "SR advantage grows decisively toward Wigner" is NOT supported in
the tested range. SR helps modestly and consistently (a slightly closer state everywhere, a resolved
energy edge at ω≈0.1) but not decisively. Honest verdict: with a physics-informed ansatz the optimiser
is largely immaterial — conditioning lives in the ansatz (cusp, backflow) and the measure (|Ψ|²), not
the optimiser. This is the publishable-null character the roadmap pre-registered for Q2.
**Caveats:** 4 seeds (energy bands wide); ω=0.01 (the true Wigner point) untested — the diagnostic
recipe diverges there, and a proper test needs the warm-started cascade. dist-to-exact advantage is
consistent but I did not compute its seed bands. N=2 only.
**Output reference:** [results/analysis/2026-07-02_SRmech_N2_w0p05/](../results/analysis/2026-07-02_SRmech_N2_w0p05/),
[results/analysis/2026-07-02_SRmech_N2_w0p03/](../results/analysis/2026-07-02_SRmech_N2_w0p03/);
trend figure results/figures/results/kernel/sr_vs_adam_trend.png. Updated Q2 section of results_kernel.tex.
**Next question:** ω=0.01 SR test with the hardened cascade recipe; N=6 SR-vs-Adam; Q3 dual-track.

### [2026-07-02e] — Q1 N-scaling (at init): the CTNN-vs-DeepSet compression gap GROWS with N (1.7×→2.6× to N=20); trained-state N≥12 blocked by training/memory (deferred)

**Motivation:** Does the compression advantage grow with system size (step 4)? The trained-state d_eff
at N≥12 is blocked — from-scratch N=12 training diverges (E→1e21, rank collapse) and the exact Laplacian
OOMs even at batch 512. But the Phase-0 finding was that the architectural gap is present at
INITIALISATION, and d_eff at init needs only first derivatives (no Laplacian, no training) — so the
inductive-bias scaling is cheaply measurable.
**Method:** `scripts/run_dinit_scaling.py` — build the UNTRAINED f_net tangent space (ctnn_vcycle_big,
deepset_big + backflow) on a common Gaussian probe at oscillator scale, fair d_eff via build_O (CPU
SVD), N∈{6,12,20}, ω=1, 3 random inits each.
**Results:**
- **CTNN d_eff(init) falls with N:** 2.09±0.77 (N=6) → 1.63±0.47 (N=12) → 1.44±0.24 (N=20).
- **DeepSet d_eff(init) is rigid:** 3.60±0.79 → 3.49±0.61 → 3.69±0.97 (N-independent).
- **Gap GROWS:** 1.7× → 2.1× → 2.6×. Message passing compresses harder the more particles there are;
  the separable pairwise sum does not.
**Interpretation:** The architectural inductive-bias dimension scales the right way — CTNN's compression
advantage widens with N, supporting the thesis prediction that message passing wins more at scale. This
is the INIT bias, not the trained solution (init≠solution: trained CTNN N=6 ω=1 is ~1.4, below its init
2.1), so it probes the architecture's raw parametrisation, which is exactly the "present at init"
inductive bias.
**Caveats:** Initialisation only (trained-state N-scaling deferred behind the large-N training
infrastructure: stable warm-started recipe + Laplacian chunking). Seed scatter at init is large (±0.5–1.0,
untrained nets vary) but the gap and monotone trend are clear. Gaussian probe (not |Ψ|²); ω=1 only.
**Output reference:** [results/analysis/2026-07-02_dinit_scaling/](../results/analysis/2026-07-02_dinit_scaling/)
(dinit.csv, summary.json, fig_dinit_scaling.png); `scripts/run_dinit_scaling.py`.
**Next question:** The deferred trained-state N=12 scaling (needs a stable N=12 recipe + Laplacian
memory fix). Q2 low-ω trend (ω=0.05/0.03 running) and Q3 dual-track still open.

### [2026-07-02d] — N=6 mode-naming (operator decomposition): leading modes ARE physical collective modes; at Wigner DeepSet's leading mode goes NON-physical (seed-robust) — corroborates the cross-projection

**Motivation:** The N=2 naming (2026-07-02) was exact. Extend "what are the manifolds" to N=6, where
there is no analytic solution, by naming the tangent modes against a basis of physical operator
generators (step 3 of the Q1 program).
**Method:** `scripts/run_mode_naming_N6.py` — build the net's top NTK eigenfunctions on a probe, project
onto {monopole Σr_i², quartic Σr_i⁴, quadrupole Σ(x²−y²)/Σ2xy, correlation-hole Σ_{i<j}exp(−r_ij²/2σ²)
(σ=0.5,1.0 ℓ), pair-Coulomb Σ1/r_ij, pair-linear Σr_ij}, centered on the probe. Leading-mode R² and the
dominant operator, for CTNN and DeepSet at ω=1 (acc/casc checkpoints) and ω=0.01 (seeded s0). Then
seed-verified the ω=0.01 leading-mode R² across all 3 seeds per arch.
**Results:**
- **ω=1: both leading modes are physical.** CTNN leading mode R²=0.99 (top operators pair_linear 0.98,
  hole 0.97, monopole 0.91); DeepSet R²=1.00 (hole 0.98, pair_linear 0.97, monopole 0.91). The N=2
  breathing/correlation picture carries to N=6.
- **ω=0.01: CTNN stays physical, DeepSet does NOT (seed-robust).** CTNN leading-mode R² = 0.95/0.92/0.92
  (mean 0.93, dominated by pair/Coulomb/breathing); DeepSet leading-mode R² = 0.01/0.02/0.01 (mean 0.01)
  — its leading tangent direction is essentially orthogonal to the physical-operator span, with the
  physical content demoted to a subdominant mode (operator basis is adequate: it captures DeepSet's
  mode-3 at 0.97).
**Interpretation:** Independently corroborates the A3 cross-projection via a different method: at strong
correlation the separable net's DOMINANT variational direction drifts off the physical collective modes,
while message passing keeps its leading direction oriented to the physics. Names the N=6 modes
(pair-correlation / breathing / correlation-hole) and completes "what are the manifolds" beyond N=2.
**Caveats:** Operator basis is finite (8 generators) — "non-physical" means "outside this span" (but the
basis demonstrably captures physical modes when present, so R²=0.01 is a genuine miss). Jastrow (f_net)
tangent only; single checkpoint per arch at ω=1 (ω=0.01 is 3-seed).
**Output reference:** [results/analysis/2026-07-02_mode_naming_N6/](../results/analysis/2026-07-02_mode_naming_N6/)
(summary.json); `scripts/run_mode_naming_N6.py`.
**Next question:** Q2 — the low-ω SR-vs-Adam test needs a hardened recipe (the ω=0.01 diagnostic run
diverged: fixed-lr from-scratch training blows up at ω=0.01, var→1e35). Warm-start + anneal + clip, or
run at ω=0.05/0.03 first. Then Q3 dual-track, Q1 N=12 scaling.

### [2026-07-02c] — Q1 anchor seeded (independent basins): the weak-coupling compression gap is CONFIRMED — CTNN 1.40±0.17 vs DeepSet 3.25±0.03 (~11σ)

**Motivation:** After the ω=0.01 inversion was retracted (2026-07-02b), the last open Q1 claim was the
weak-coupling compression gap itself (CTNN d_eff ~1.2 vs DeepSet ~3.4 at ω=1), still single-seed. Seed
it — and do so with INDEPENDENT basins (stronger than the crossover's shared warm-start).
**Method:** No new training. The Phase-0/1 data already holds independent ω=1 runs: 5 CTNN-big
(eq_seed1, eq_seed2, 2x2_adam, 2x2_sr, acc) and 3 DeepSet-big (2x2_adam, 2x2_sr, casc) — genuinely
separate training runs. Generalised the seeded analyser (`scripts/run_phaseB_seeded_analysis.py
anchor`) to measure fair common-probe d_eff + var(E_L) + NO-PR on all 8, mean±s.d. per arch.
(Fixed a GPU-OOM by moving the SVD to CPU and freeing O per iteration — 8 nets on a shared GPU.)
**Results:**
- **Compression gap confirmed, independent-basin:** CTNN d_eff **1.40 ± 0.17** (5 runs, range
  1.18–1.62) vs DeepSet **3.25 ± 0.03** (3 runs, range 3.23–3.30). Gap 2.3×, separation ~11σ —
  overwhelmingly robust. DeepSet-big's d_eff is remarkably tight across independent runs (±0.03).
- **NO count architecture-independent:** both ~2.1–2.2 at ω=1 (trace ~3.06, well-resolved, unlike the
  grid-truncated ω=0.01). CTNN ratio 0.63, DeepSet 1.54 → DeepSet over-complete, CTNN compressed.
- **var(E_L):** CTNN ~2.2e-2 vs DeepSet ~7.1e-2 (~3×) — CTNN lower, consistent with everywhere.
**Interpretation:** Q1's dimensional story is now fully seeded end-to-end: a strong compression gap at
weak coupling (1.40 vs 3.25, ~11σ, independent basins) that CLOSES toward the Wigner crystal (3.70 vs
3.84, converged). Combined with the surviving var(E_L) discriminator, the cross-projection (orthogonal
subspaces at the crystal), and the exact N=2 naming, the architecture question is robustly answered:
CTNN compresses the correlator at weak coupling and finds a better (lower-variance, different-subspace)
state at strong coupling; it never simply "has more/fewer dimensions" in a way that inverts.
**Caveats:** ω=1 and ω=0.01 seeded; intermediate ω still single-seed (sweep). ω=0.01 NO grid-truncated.
The 5 CTNN runs mix adam/sr/eq configs (same ctnn_vcycle_big arch) — independent but not identical recipe.
**Output reference:** [results/analysis/2026-07-02_phaseB_anchor/](../results/analysis/2026-07-02_phaseB_anchor/)
(seeded.csv, summary.json); analyser `scripts/run_phaseB_seeded_analysis.py`.
**Next question:** Q1 is locked; move to the N=6 mode-naming (extend the reference solver to N≥6), the
Q2 low-ω SR sweep, and the Q3 dual-track.

### [2026-07-02b] — Phase B (Gate B): the Wigner d_eff INVERSION does not survive seeding — CTNN and DeepSet CONVERGE to ~3.7 at ω=0.01; CTNN's crystal advantage is variance, not dimension

**Motivation:** The Phase-A headline (single-seed) was a dramatic d_eff "inversion" at the Wigner
crystal — CTNN 5.21 overtaking DeepSet 3.24. That is exactly the kind of single-seed claim the
feedback discipline says to seed BEFORE promoting. Phase B (crossover-first): retrain 3 seeds ×
{CTNN, DeepSet} at ω=0.01 to matched analysis grade and put error bars on it, and confirm the DeepSet
low value isn't an under-converged-checkpoint artifact.
**Method:** `scripts/launch_phaseB_crossover.sh` — 6 runs on GPUs 1–6, each warm-started from its
arch's ω=0.1 cascade checkpoint (steps 400 / polish 120 / SR-polish 200), then
`scripts/run_phaseB_crossover_analysis.py` measures energy error, var(E_L), fair common-probe d_eff,
and NO participation ratio per seed → mean±s.d. Also re-ran the A3 cross-projection on the matched-grade
checkpoints. (Seeds share the ω=0.1 warm-start → measurement + optimisation variance around the
crossover, not independent basins from ω=1.)
**Results:**
- **The inversion is OVERTURNED.** All 6 are genuine GS (CTNN err −0.012/−0.044/−0.033%, DeepSet
  +0.082/+0.059/+0.075%). Seeded d_eff: **CTNN 3.70 ± 0.04 vs DeepSet 3.84 ± 0.08** — equal within
  error, NOT an inversion. The single-seed 5.21 vs 3.24 was a checkpoint/optimisation-path artifact:
  at ω=0.01 the landscape is flat enough that d_eff depends on which GS a run settles into.
- **var(E_L) discriminator SURVIVES (seeded):** CTNN 1.8e-5 vs DeepSet 3.3e-5 (1.8×), and better
  energy. At the crystal the two nets reach the SAME dimension but CTNN finds a lower-variance state.
- **A3 cross-projection SURVIVES / strengthens (matched checkpoints):** at ω=0.01 the leading modes are
  nearly orthogonal (mutual top-1 capture ~0.01); CTNN's leading mode needs 3 DeepSet directions,
  DeepSet's needs all 6 of CTNN's; top-3 subspace overlap 0.42. Same dimension, DIFFERENT subspaces.
- **NO count** ~equal (both ~6.1, grid-truncated trace ~1.85) — architecture-independent, as before.
**Interpretation:** The honest, seeded Q1 picture is a weak-to-strong CONVERGENCE, not an inversion:
CTNN compresses hard at weak coupling (d_eff 1.2 vs 3.4 at ω=1) and that gap CLOSES toward Wigner,
where both reach ~3.7–3.8 and CTNN's remaining edge is variance/quality plus a different (better)
tangent subspace. This is cleaner and more defensible than the fragile inversion, and it keeps the
through-line intact (the var(E_L) discriminator — a genuine positive — survives everywhere). The
retraction is the seeding discipline working: caught before the thesis, not after.
**Caveats:** The ω=1 compression gap (1.2 vs 3.4) is itself still single-seed — needs the same seeding
(though it's a large gap). Seeds share the ω=0.1 warm-start (not independent ω=1 basins). ω=0.01 NO
count grid-truncated. Updated results_kernel.tex throughout (sweep table dagger + seeded-correction
table `tab:crossover-seeded` + figure; interpretation, synthesis, conclusions, caveats all corrected).
**Output reference:** [results/analysis/2026-07-02_phaseB_crossover/](../results/analysis/2026-07-02_phaseB_crossover/)
(crossover.csv, summary.json, fig_phaseB_crossover.png); checkpoints in
`results/analysis/2026-07-02_N6_w001_{ctnn,deepset}_s{0,1,2}/`.
**Next question:** Seed the ω=1 anchor (confirm the 1.2-vs-3.4 compression gap) and the full ω-sweep
with independent basins (cascade ×3 from ω=1); the N=6 response-projection (name the modes at N=6);
then Q2 low-ω SR sweep and Q3 dual-track.

### [2026-07-02] — Phase A (Gate A): the Q1 mechanism is CLOSED — DeepSet's tangent dimension is rigid and CROSSES the physical mode count; and the manifold is NAMED (breathing mode) at the exact N=2 anchor

**Motivation:** Close the "why CTNN > FFNN / what are the manifolds / why can't FFNN find them"
interpretation on existing checkpoints (no training), before spending Phase-B compute. Four closers on
the N=6 cascade + the exact N=2 anchor: A2 physical mode count for BOTH arches; A3 cross-architecture
tangent projection; A4 name the modes; A5 harden the crossover. (Also: consolidated + committed a
week of in-flight Phase-0/1/2 work, wrote the thesis results chapter `Thesis/results_kernel.tex`,
corrected the run_weak_form conditioning claim.)
**Method:** `scripts/run_phaseA_closers.py` (A2/A3/A5) and `scripts/run_response_projection.py` (A4).
A2: `physics_probes.natural_orbital_occupations` (1-RDM participation ratio) on CTNN AND DeepSet across
ω, grid half-width scaled to the sampled density; paired with fair common-probe tangent d_eff. A3: on a
common pooled probe, build O for both arches, form the NTK K=OOᵀ (same R^B function space), compare the
top NTK eigenvectors by subspace/principal-angle overlap. A4: extended `analysis/reference.py` with a
Coulomb strength λ and excited relative states; projected the N=2 net's Jastrow tangent modes onto the
exact {∂_ω J\*, ∂_λ J\*, φ_n=u_n/u_0}. A5: `fair_dimension.dimension_convergence` at ω∈{1,0.01}.
**Results:**
- **A2 — the physical mode count is architecture-independent; only CTNN's tangent tracks it.** NO
  participation ratio is ~equal for the two nets at each ω (ω=1: CTNN 2.10 / DeepSet 2.34; ω=0.01:
  6.19 / 5.94) — it is a property of the *state*, rising ~2→6 toward Wigner. CTNN's tangent d_eff/NO
  ratio ≈ 0.58, 0.57, 0.52, 0.84 (ω=1→0.01) — it allocates directions in step with the physics.
  DeepSet's ratio = 1.45, 1.26, 1.08, 0.55 — **it crosses unity around ω=0.1**: over-complete
  (redundant) at weak coupling, under-complete (can't reach the ~6 physical modes) at Wigner. This is
  the mechanism: DeepSet is pinned near d_eff~3.3 regardless of the physics.
- **A3 — at weak coupling both nets share the single leading collective coordinate; at Wigner they
  diverge.** ω≥0.1: DeepSet top mode captures ≥0.98 of CTNN's leading mode and vice versa (k90=1) —
  they agree on the dominant direction, differ only in DeepSet's redundant extras. ω=0.01: CTNN's
  leading mode needs TWO DeepSet directions (top-1 captures only 0.10); DeepSet's leading mode never
  reaches 90% even in CTNN's top-6 (0.69) — at strong correlation DeepSet's dominant variational
  direction is no longer a physical collective mode.
- **A4 — the manifold is NAMED.** At N=2 ω=1 (overlap²=1.00000, E=3.00000, solver validated; excited
  E_rel = 2.0/3.85/5.75), the net's leading Jastrow tangent mode is captured by
  span{∂_ω, ∂_λ, φ_1} to **1.000** (top-2 subspace 0.990). Least-squares naming: R²=1.000, dominated by
  the **breathing** response (∂_ω coef −0.83) + first relative excitation (φ_1 −0.16); ∂_λ, φ_2 ≈ 0.
  All four physical modes lie in the net's top-6 span (0.92–1.00); the other net directions are
  tiny-eigenvalue numerical modes. So the ~1-D manifold at weak coupling *is the breathing mode*, and
  the effective 2–3D tangent space is {breathing, correlation-hole, first relative excitation}.
- **A5 — the crossover is not a sampling artifact.** DeepSet ω=0.01 d_eff is flat at 3.20/3.30/3.25/3.24
  over n=256/512/1024/2048; CTNN plateaus ~4.99–5.75 (→5.21). The DeepSet pin at ~3.24 (≪ the 5.9
  physical count) is real and sample-converged.
**Interpretation:** Q1 is answered mechanistically. "Low-dimensional = good" was the wrong frame; the
right one is "the tangent dimension should MATCH the physical collective-mode count." CTNN's does
(adaptive, tracks the NO count, converging to it at Wigner); DeepSet's is rigid and crosses it, which
is simultaneously its redundancy at weak coupling and its under-expressiveness (higher var(E_L)) at
Wigner. A4 grounds the whole story physically: the dimensions are the low collective modes of H
(breathing / correlation-hole / relative excitation). Also unifies with Q2: the leading mode is the
smooth *breathing* (soft), the correlation-hole/cusp is the stiffer subdominant direction — precisely
what SR's whitening should help with as correlation grows.
**Caveats:** ω=0.01 NO count is on a grid-truncated 1-RDM (trace ~1.87 of 3) — absolute value
uncertain, but the CTNN/DeepSet ratio is fair (shared grid). A4 naming is exact only at N=2; the N=6
response-projection needs the finite-difference reference (pending). N=6 still single-seed (Phase B).
A3's "DeepSet leading mode is non-physical at Wigner" is on the single low-ω checkpoint, whose GS
quality is worse than CTNN's.
**Output reference:** [results/analysis/2026-07-02_phaseA_closers/](../results/analysis/2026-07-02_phaseA_closers/)
(modecount.csv, convergence.csv, summary.json, fig_phaseA_closers.png),
[results/analysis/2026-07-02_response_projection/](../results/analysis/2026-07-02_response_projection/)
(summary.json, fig_response_projection.png); write-up in
[Thesis/results_kernel.tex](../Thesis/results_kernel.tex).
**Next question:** Phase B — seed the ω-sweep (≥3 seeds at ω∈{1,0.1,0.01}, both arches) and retrain the
DeepSet low-ω checkpoints to analysis grade, to firm the crossover with error bars; then the N=6
response-projection (name the modes where it matters); then Q2 low-ω SR sweep and Q3 dual-track.

### [2026-06-27] — Phase 2 Q2 (low-ω onset): at N=2 ω=0.1 SR finally beats Adam — first statistically-resolved SR advantage; and the xfill completed the ω-sweep

**Motivation:** Gate 1 concluded that the SR-vs-Adam advantage, null at ω=1 (cusp on *and* off),
should switch on as ω drops and κ(S)/var(E_L) grow. Test the first low-ω point at the exact N=2
anchor. (This entry back-fills two runs done Jun 23–27 that were never journalled.)
**Method:** `scripts/exp_sr_mechanism.py` at N=2 ω=0.1, cusp-on, 4 seeds; identical protocol to
Jun-20/23 (common Adam warm-up → branch into Adam-only vs SR-only from the same checkpoint; measure
energy error, distance-to-exact, and the stiff/soft split in the FIXED ω=1 NTK eigenbasis). Separately,
the `2026-06-23_N6_w{028,005,003}_xfill` runs filled the intermediate ω of the Phase-2 sweep.
**Results:**
- **SR beats Adam at ω=0.1 (first resolved advantage).** Energy error: Adam **+0.163%** [+0.058,+0.268]
  vs SR **−0.028%** [−0.108,+0.052] — the ±1 s.d. seed bands are nearly disjoint (Adam-lo +0.058 >
  SR-hi +0.052), a ~1.5σ separation. Distance-to-exact: Adam 0.039 vs SR 0.030. SR also has tighter
  seed bands. This is the first regime where SR's advantage is *resolved*, not merely descriptive.
- **But not via the pre-registered stiff-mode channel:** stiff-mode error ties (Adam 1.66e-2 vs SR
  1.67e-2); SR's gain is in the SOFT modes (0.882 vs 0.673) and overall. Caveat: the stiff/soft basis
  is the ω=1 frame, so its decomposition is only loosely meaningful at ω=0.1 — the robust signal is the
  energy/dist advantage.
- **ω-sweep completed** (xfill ω=0.28,0.05,0.03): CTNN eff-rank rises more smoothly than the original
  4-point version suggested — 1.21, 1.27, 1.24, 1.80, 1.79, 2.56, 5.21 (ω=1→0.01) — but still jumps
  ~2× over the last factor-3 in ω. DeepSet stays ~3.2–4.3 (non-monotone; 4.31 at ω=0.05, 3.24 at 0.01).
**Interpretation:** Consistent with the Q2 thesis: with a physics-informed ansatz SR is immaterial at
weak coupling and earns its keep toward Wigner, where whitening bites. The mechanism is broader than
"collapse the cusp mode" — at low ω the *whole* tangent space is ill-conditioned. The onset is caught;
it must now be pushed to lower ω and N=6 to see whether the ~1.5σ becomes decisive.
**Caveats:** N=2, single ω=0.1, 4 seeds, cusp-on; SR at 256 samples/500 steps. The ω-sweep is still
single-seed per (arch,ω) and the DeepSet ω=0.01 point may be under-converged (Phase B firms it).
**Output reference:** [results/analysis/2026-06-27_SRmech_N2_w0p1/](../results/analysis/2026-06-27_SRmech_N2_w0p1/)
(summary.json, raw.json, fig_sr_mechanism.png); sweep xfill in `results/analysis/2026-06-23_N6_w*_xfill/`.
**Next question:** Phase 2 proper — the low-ω SR-vs-Adam sweep at N=2 and N=6 (does the advantage grow
and become significant?), and Phase A closure of Q1 (name the modes; DeepSet mode count).

### [2026-06-23b] — Phase 2 (no-training half): the CTNN-vs-DeepSet effective-dimension relation INVERTS across the Wigner crossover — CTNN's d_eff is adaptive, DeepSet's is rigid

**Motivation:** Phase-2 test of the headline Wigner predictions, on existing N=6 backflow-cascade
checkpoints (no training): Q1 does the CTNN/DeepSet eff-rank gap grow toward Wigner; Q3 does
var(weak)/var(strong) track ω.
**Method:** `scripts/run_phase2_omega_sweep.py` over the cascade {ω=1,0.5,0.1,0.01} × {CTNN,DeepSet}
(configs identical across ω: ctnn_vcycle_big 66498 / deepset_big 34354 f_net params). Q1: fair eff-rank
on a common pooled probe per ω. Q3: var(E_L) (chunked detached Laplacian) and var(e_w) per arch.
**Results:**
- **Q1 — the gap inverts (prediction was WRONG, truth is richer).** eff-rank(S): CTNN
  **1.21 → 1.27 → 1.80 → 5.21** (ω = 1 → 0.5 → 0.1 → 0.01); DeepSet **flat ~3.3** (3.38, 3.38, 3.62,
  3.24). The DeepSet/CTNN gap goes 2.78× → 2.66× → 2.02× → **0.62×** (CTNN overtakes at ω=0.01).
  So CTNN's effective dimension is **adaptive** — minimal at weak correlation (compresses to ~1
  collective coordinate), rising to ~5 toward the Wigner crystal — while DeepSet's is **rigid** (~3.3
  regardless of ω). CTNN wins by *compression* at weak coupling, by *adaptive expressivity* at strong.
- **Q3 — var(E_L) drops cleanly toward Wigner** (CTNN 2.0e-2→7.7e-3→7.9e-4→8.5e-6; DeepSet ~2–4×
  higher at every ω: CTNN is the lower-variance wavefunction throughout). var(e_w)/var(E_L) stays
  30–6500× (weak form always catastrophic) but is too heavy-tailed for a clean ω-trend.
**Validation (prompted by the "is ω=0.01 an outlier?" challenge):** the ω=0.01 CTNN checkpoint is a
genuine GS — E=0.68935 (−0.146% vs ref), var(E_L)≈1e-5, message-ablation dE=0.066 / var×138. The
eff-rank ~5.25 is independently reproduced (Jun-15 battery `effective_rank`=5.255; my sweep 5.21;
convergence plateau ~5.0 over n=512→4096). **Physics corroboration:** the same battery's
natural-orbital participation ratio = **5.77** at ω=0.01 ⇒ the 1-RDM spreads over ~5–6 orbitals at the
Wigner crystal, and **d_eff(tangent) ≈ natural-orbital count** — a direct preview of the Phase-4
"d_eff = physical collective-mode count" claim.
**Interpretation:** The ω=1 anchor told a misleading "CTNN = low-dim" story; the sweep reveals
"CTNN = *adaptive*-dim, tracking the physical collective-mode count, while DeepSet is rigid." This is
more physical and unifies Q1 with the natural-orbital structure.
**Caveats:** single-seed per (arch,ω); the ω=0.1→0.01 jump (1.80→5.21) is abrupt vs the smooth rise
below — could be a genuine sharp Wigner crossover OR the ω=0.1 point under-resolved; needs the
intermediate ω (0.28, 0.05) and a seed before anchoring. var(e_w) heavy-tailed.
**Output reference:** [results/analysis/2026-06-23_phase2_omega_sweep/](../results/analysis/2026-06-23_phase2_omega_sweep/)
(omega_sweep.{csv,json}, fig_phase2_omega_sweep.png).
**Next question:** fill the crossover (ω=0.28, 0.05) to resolve smooth-vs-sharp and confirm the
inversion; then Phase-2 training half (Q2 low-ω SR-vs-Adam — does SR finally beat Adam where var(E_L)
and κ explode?).

### [2026-06-23] — Phase 1 Q2 (cusp-OFF 2×2) + Gate 1: the cusp prior is a stiff-mode preconditioner; SR≥Adam cusp-off but within seed noise at ω=1 → decisive SR test moves to low ω

**Motivation:** Complete the {cusp ON/OFF}×{Adam/SR} grid (D2): the Jun-20 null tested only cusp-ON
(the quadrant where SR is predicted not to help). Run cusp-OFF — where the stiff cusp mode must now be
*learned* — and consolidate all three questions at the anchor (Gate 1).
**Method:** `scripts/exp_sr_mechanism.py --no-cusp --seeds 4` (added a `--no-cusp` flag setting
`use_analytic_cusp=False`; pinned all per-seed Systems + the fixed NTK basis to one device via
`CUDA_VISIBLE_DEVICES=0` after a cuda:0/cuda:1 device-split crash). Same protocol as Jun-20: common
Adam warm-up, branch into Adam-only vs SR-only from an identical checkpoint, log energy error,
distance-to-exact, and the stiff/soft split in a FIXED NTK eigenbasis. (Run note: a first launch died
overnight to a session teardown after seeds 0–2; relaunched fully detached via `setsid`.)
**Results (4 seeds, mean ± seed-std):**
- **Cusp removal inflates the stiff-mode error ~2–3×** (cusp-ON ~3–5e-3 → cusp-OFF ~1.1e-2) for both
  optimizers — direct confirmation that the fixed cusp removes the stiffest direction from the
  *learning* problem (D2 core claim).
- **The optimizer ranking flips:** cusp-ON Adam ≥ SR on stiff modes (3.4e-3 vs 5.0e-3, energy tie);
  cusp-OFF SR ≤ Adam on *every* metric — stiff 1.06e-2 [0.94,1.19] vs 1.13e-2 [1.05,1.21], energy
  −0.216% [−0.24,−0.19] vs −0.277% [−0.35,−0.20], dist 5.65e-3 vs 7.32e-3 — and SR is markedly more
  reproducible (energy seed-std ≈0.025% vs Adam ≈0.074%).
- **But the SR-vs-Adam gap is within seed noise** (bands overlap on every metric). SR is consistently
  but not significantly better cusp-off at ω=1.
**Interpretation:** The cusp prior is a function-space preconditioner doing part of SR's job; removing
it flips the optimizer ranking toward SR (mechanism direction = D2's prediction) but the advantage is
not significant in the easy regime. Honest statement: *with a physics-informed ansatz the optimizer is
largely immaterial at ω=1 even cusp-off; SR's edge (and its lower seed variance) is real but small.*
The decisive "SR collapses the stiff mode while Adam stalls" must be tested where κ(S)/var(E_L) explode
— **low ω** (Phase 2).
**Gate 1 (all three questions at the anchor):** Q1 fair d_eff gap survives (CTNN 1.2–1.7 vs DeepSet
2.9–3.7, measure-robust); Q2 as above; Q3 weak form loses zero-variance 400–11000× (means agree).
Common thread: the easy regime under-resolves the effects → all three point to the ω→Wigner sweep.
**Caveats:** N=2 ω=1 only; 4 seeds; stiff/soft basis is the cusp-ON converged reference (a fixed
frame); SR used 256 samples/500 steps. var-ratio (Q3) heavy-tailed (order-of-magnitude only).
**Output reference:** [results/analysis/2026-06-23_SRmech_N2_nocusp/](../results/analysis/2026-06-23_SRmech_N2_nocusp/)
(summary.json, raw.json, fig_sr_mechanism.png, GATE1.md).
**Next question:** Phase 2 — ω-sweep {1, 0.5, 0.28, 0.1, 0.01} for all three: does the d_eff gap grow
toward Wigner (Q1), does SR finally beat Adam where var(E_L) explodes (Q2), does var(e_w) and the
collocation-vs-VMC gap track ω (Q3)?

### [2026-06-22c] — Phase 1 (no-training half): Q1 fair-dimension gap survives the common-probe protocol; Q3 the weak form sacrifices zero-variance by 400–11000×

**Motivation:** Execute the no-training half of Phase 1 on existing checkpoints: (Q1) re-measure the
CTNN-vs-DeepSet effective-dimension gap *fairly* (common probe set, not each net's own |Ψ|²; sample-
convergence; measure sensitivity); (Q3) demonstrate the Laplacian / zero-variance anatomy
(var(strong)→0 vs var(weak)) and the integration-by-parts mean-equality.
**Method:** New sanctioned code `src/analysis/fair_dimension.py` (common-probe eff-rank wrapper over
`build_O`/`kernel_spectrum`, f_net tangent space only — backflow is identical ~13.3k params across
arches) + `scripts/run_fair_dimension.py`; `scripts/run_estimator_variance.py` (strong via the chunked
detached `local_energy`, weak via chunked `residual_local_energy`). Pipeline validated: reproduces the
stored own-density eff-rank (CTNN 1.64 vs 1.60; DeepSet 3.29 vs 3.35). N=6 ω=1 checkpoints (3 CTNN
seeds, DeepSet ladder 20k→164k); pooled probe = mixture of CTNN+DeepSet |Ψ|² (3072 pts).
**Results:**
- **Q1 — the gap survives fair measurement (Gate-1 PASS).** Under the common pooled probe set,
  eff-rank(S): **CTNN 1.20–1.66 (3 seeds) vs DeepSet 2.94–3.65** (sizes 20k–164k). For *every* model
  the value is within ±0.04 whether probed on CTNN-density, DeepSet-density, or pooled points ⇒ the
  ~2× gap is **not an own-density artifact**, it is architectural. DeepSet stays ~3 independent of size
  (xl 150k → 2.94); CTNN stays ~1.4. Outputs: `results/analysis/2026-06-22_fair_dimension_N6w1/`.
- **Q3 — zero-variance is the Laplacian's gift; the weak form throws it away.** var(weak)/var(strong)
  = **2071× (N=2 exact), 10957× (N=6 CTNN), 403× (N=6 DeepSet)**. At the near-exact N=2 GS
  var(E_L)=1.4e-3 (≈0) while var(e_w)=2.9. Mean-gaps (+0.05 to +0.50) are ~1–1.5 SE of the (very
  noisy) weak estimator → consistent with the integration-by-parts identity ⟨E_L⟩=⟨e_w⟩, but the weak
  form is too noisy to confirm it tightly — which *is* the point: the weak form is unbiased in the mean
  yet a catastrophic estimator. Outputs: `results/analysis/2026-06-22_estimator_variance/`.
**Interpretation:** Q1's central claim ("CTNN compresses the correlator into a ~2× lower-dimensional
tangent space") is now fair, seeded, and measure-robust at N=6 ω=1. Q3 makes the
collocation/Laplacian story concrete and quantitative: "training still works without the Laplacian"
(integration by parts preserves the mean) but at a 400–11000× variance cost (zero-variance lost) —
explaining why the winning recipe keeps the forward-only E_L reward and β small.
**Caveats:** N=6 ω=1 only (ω-sweep is Phase 2); var(e_w) is heavy-tailed and poorly estimated at 2048
samples — the cross-model var(e_w) ordering (CTNN>DeepSet) is NOT robust, only the
orders-of-magnitude gap is. Q1 DeepSet sizes are single-seed (CTNN has 3 seeds).
**Output reference:** [results/analysis/2026-06-22_fair_dimension_N6w1/](../results/analysis/2026-06-22_fair_dimension_N6w1/)
(fair_table.csv, convergence.csv, fig_fair_dimension.png),
[results/analysis/2026-06-22_estimator_variance/](../results/analysis/2026-06-22_estimator_variance/).
**Next question:** Phase 1's training half — Q2 cusp-OFF 2×2 at N=2 (the unrun decisive SR test). Then
Phase 2: sweep ω→Wigner for all three questions (does the d_eff gap grow? does var(e_w) and the
SR-advantage track ω?).

### [2026-06-22b] — Phase 0 (Gate 0): consolidated the un-journalled N=6 ω=1 runs — CTNN tangent space is ~1.5-dim vs DeepSet ~3.8 (architectural, present at init); SR≈Adam null extends to N=6

**Motivation:** First execution step of the three-questions roadmap: collate the finished-but-
unwritten `2026-06-15_{2x2,eq}_*` runs into a per-question picture before any new training.
**Method:** No training. Extracted all `summary.json` (12 runs: the {CTNN,DeepSet}×{Adam,SR} 2×2, the
DeepSet equivalence ladder s/m/match/xl, 2 CTNN seeds, N=2 anchor) into
`results/analysis/2026-06-22_phase0_consolidation/consolidation_table.csv`, and recovered the per-step
`data_alignment_trajectory.csv` series (cos_sr, cos_plain, κ(S), eff-rank(S) vs step).
**Results:**
- **Q1 (CTNN vs FFNN):** CTNN eff-rank(S) = **1.49 [1.39–1.60]** (4 replicates) vs DeepSet **3.78
  [3.24–4.76]** (sizes 20k–164k). The gap **does not shrink with DeepSet capacity** and is **present
  at initialization** (CTNN ≈1.1, DeepSet ≈3.9 at step 25) ⇒ architectural inductive bias, not learned.
  var(E_L) ~0.026 (CTNN) vs 0.030–0.097 (DeepSet; xl at 2× params reaches 0.033, still short).
- **Q2 (SR vs Adam):** 2×2 shows **SR ≈ Adam for both arches at N=6 ω=1** (CTNN err 0.029% both;
  DeepSet 0.071/0.078; var tied) — the cusp-on null extends from N=2 to N=6. New dynamic: κ(S)
  *decreases* over training for CTNN (2e9→8e7) but *increases* for DeepSet (1.4e9→2e10); cos_plain
  stays ~0.06–0.17 (plain grad misaligned, but Adam's diagonal precond reaches SR's endpoint here).
  Descriptive SR=whitening holds; prescriptive SR>Adam does not at ω=1. X2 inconclusive at ω=1.
- **Q3:** not advanced (all runs are VMC). Stands at the Jun-21 ESS-collapse state; dual-track is Phase 1+.
- **D4 (lazy vs rich, recovered):** low dimensionality is baked in at init (both arches lazy in
  *dimension*); the evolving quantity is conditioning (κ), oppositely for the two.
**Interpretation:** Q1 has a real, seeded, architectural result ready to harden; Q2's easy-regime null
is consolidated and the cusp-OFF/low-ω test is the correct next decisive move; Q3 must be carried as a
first-class Phase-1 target, not dropped.
**Caveats:** eff-rank on each net's own |Ψ|², 768 samples (fairness confound → Phase-1 common probe
set); DeepSet not matched-accuracy (var comparison needs the control); trajectory eff-rank mixes
num_rank 383/767; zv extrapolation unreliable for DeepSet; N=6 ω=1 only, DeepSet sizes unseeded.
**Output reference:** [results/analysis/2026-06-22_phase0_consolidation/](../results/analysis/2026-06-22_phase0_consolidation/)
(GATE0.md, consolidation_table.csv).
**Next question:** Phase 1 — does the eff-rank gap survive the fair common-probe protocol with seeds,
and does the cusp-OFF 2×2 at N=2 show SR collapsing the stiff-mode error while Adam stalls?

### [2026-06-22] — Consolidation + audit: CTNN's low-dimensional tangent space is the (unwritten) answer to "why CTNN > FFNN"; program rebalanced to three co-equal questions under one kernel lens

**Motivation:** After a strategic review (the program had drifted into single-seed promotion →
retraction → re-planning), audit what we have actually answered vs glossed/missed against the
2026-06-13 kernel plan, and decide a stable spine. The user flagged the core need: measure the
*dimension* of CTNN vs DeepSet fairly, scale it to Wigner and higher N, and figure out what the
dimensions physically are.
**Method:** No new training. (1) Graded every plan question (T1–T6, D1–D6, A/B/C/X) by status.
(2) Audited untracked `results/analysis/2026-06-15_{2x2,eq}_*` dirs never entered in the journal.
(3) Read the eff-rank measurement path (`diagnostics.build_O` / `kernel_spectrum`,
`scripts/run_depth_analysis.py`) to check cross-architecture fairness.
**Results:**
- **Unconsolidated finding surfaced (N=6 ω=1, SR-trained, ≥2 CTNN seeds):** CTNN-vcycle (80k params)
  reaches the GS in an effective tangent space of **eff-rank(S) ≈ 1.5** with var(E_L) ≈ 0.026; DeepSet
  needs **eff-rank 3.4–4.8** and, even at 2× params (xl, 164k), var(E_L) 0.033 — never matching CTNN.
  cos_plain ≈ 0.07–0.16 (NTK-whitening / gradient-misalignment holds at N=6, both arches). This
  answers X2 (trainability), D6 (expressivity) and B3 (FFNN-equivalence: >2× params and still short)
  in one table. **κ(S) is NOT a clean discriminator** across DeepSet sizes (xl has lower κ than match);
  eff-rank and var(E_L) are.
- **Fairness audit:** the participation-ratio formula, rel_tol, centering and sample count (768) are
  identical across arches, and eff-rank (1.5–4.8) ≪ 768 ⇒ resolved (numerical_rank=767 is sample-
  capped and must NOT be reported as the dimension). One confound: each net's O is built on samples
  from its *own* |Ψ|² — a clean comparison needs a common probe set. At higher N the rank can become
  sample-limited ⇒ must report eff-rank vs n_samples.
**Interpretation:** All three angles plus the collocation/Laplacian thread are faces of one object —
the **low-dimensional, cusp/kinetic-dominated tangent space**. CTNN wins by compressing the state into
fewer collective directions (lower κ AND lower var(E_L) at fewer params). The program is organised
around **three co-equal questions** — Q1 (CTNN vs FFNN), Q2 (SR vs Adam), Q3 (collocation vs VMC) —
each pursued through this shared kernel lens; the low-dimensional tangent space is the synthesis they
converge on, not the organizing axis (rebalanced from an initial dimension-centric framing; see
DECISIONS 2026-06-22b). The "nulls" (SR≈Adam cusp-on; k⁴/k² unconfirmed) are boundary conditions of
this, not refutations.
**Caveats:** the headline eff-rank values are single-/two-seed, own-density-sampled, 768 points,
ω=1 only — they motivate the program, they are not yet the fair measurement. κ comparisons are noisy.
The dimension's *physical identity* is conjectured (collective modes), not yet measured.
**Output reference:** [STATUS_REPORT_2026-06-22.md](STATUS_REPORT_2026-06-22.md),
[plans/2026-06-22_dimension-program-and-roadmap.md](plans/2026-06-22_dimension-program-and-roadmap.md);
source dirs `results/analysis/2026-06-15_{2x2,eq}_N6w1_*`.
**Next question:** Does the ~1.5-vs-3.5 gap survive the fair common-probe-set protocol with seeds
(Phase 1)? Then: does d_eff drop / the gap grow toward Wigner (Phase 2), grow sub-linearly in N
(Phase 3), and can each dimension be named as a physical response/excitation mode of H (Phase 4)?

### [2026-06-21b] — Smart proposal vs simple Gaussian: ESS is the conditioning lever, and the adaptive mixture is OFF the efficient frontier

**Motivation:** If ESS/measure governs collocation conditioning (2026-06-21a), how much does the
trainer's "smart" adaptive Gaussian mixture actually buy over a plain Gaussian? Quantify the
coverage-vs-efficiency-vs-conditioning exchange-rate.
**Method:** `scripts/exp_proposal_compare.py` on the N=6 cascade. Proposals: single Gaussian at
sigma_f/sqrt(omega) for sigma_f in {1.0 narrow, 1.3 matched (= run_weak_form default), 2.5 broad}
vs the adaptive multi-width mixture. Per (proposal, omega): ESS fraction (4096-draw), coalescence
coverage (frac with min pair dist < 0.5 ell), tail reach (95th pct max radius / ell), and
kappa(S)/kappa(A_strong) under that measure (omega in {1, 0.1}).
**Results:** (i) **ESS controls conditioning directly**: every proposal with ESS >~1% has a
*resolved* kappa(A)~1e5; every proposal with ESS <1% *floors* at kappa~1e8 (effective rank ~ ESS).
(ii) **A single matched Gaussian (1.3) dominates the smart mixture on BOTH axes**: at omega=0.1,
matched ESS=1.6% / kappa(A)=1.5e5 (resolved) vs mixture ESS=0.67% / kappa(A)=9.6e7 (floored) -- ~600x
better conditioned and ~2x more efficient. (iii) **But matched wins by not covering the hard
region**: its tail reach is fixed at 4 ell while the density needs 9.5 ell at omega=0.1 and 22 ell
at omega=0.01; only the mixture's broad components reach there. Narrow (1.0) has the best
coalescence coverage (60%) but the worst tail (3.1 ell).
**Interpretation:** "Smart" sampling, as configured (adapt_sigma_fs tiers with very broad
components), buys *negative* conditioning -- it pays ~600x in kappa and ~2x in ESS to gain tail
coverage. The matched Gaussian's good conditioning is partly **coverage bias** (it never penalises
the residual in the uncovered tails, so the solution can be wrong there). The real tradeoff is
**conditioning vs unbiased coverage**, and the current mixture sits far off the efficient frontier:
the broad components over-pay for reach. Actionable: the lever is the proposal; a matched core + a
few targeted broad/coalescence components, or a learned flow (shellflow), should give the needed
coverage at a fraction of the ESS/conditioning cost.
**Caveats:** static adapt-tier widths (the trainer also *refits* the gmm / uses shellflow -- not
tested here); coverage proxies are heuristic; kappa at 96 samples (resolved cases are trustworthy,
floored cases are lower-bounded). Coverage "bias" is argued, not yet measured against exact tails.
**Output reference:** [results/analysis/2026-06-21_proposal_compare/](results/analysis/2026-06-21_proposal_compare/)
(compare.json, fig_proposal_compare.png)
**Next question:** Does the matched Gaussian's better conditioning actually train to a *better or
biased* solution vs the mixture (the conditioning-vs-coverage tradeoff, in real energy error)? And
can a learned/shellflow proposal reach the efficient frontier (full coverage at matched-like ESS)?

### [2026-06-21] — Collocation-conditioning Phase 0: the sampling measure is the dominant conditioning lever (not strong-vs-weak)

**Motivation:** The VMC SR-null redirected the thesis "conditioning not depth" theory to the
collocation picture, where the strong-form residual operator `A = J*J` (`J = ∂_θ R`) is supposed
to be `κ~k⁴`-catastrophic (vs weak-form `k²`). Test it directly on converged checkpoints (load-time,
no training). New probe: `diagnostics.residual_jacobian` (strong = `E_L`; weak = Rayleigh
`½|∇logΨ|²+V`), spectrum via `kernel_spectrum`; driver `scripts/exp_conditioning_A.py` with
`--measure {psi2, mixture}`.
**Method:** N=2 ω=1 and the N=6 ω∈{1,0.5,0.1,0.01} CTNN+backflow cascade. Two sampling measures:
`|Ψ|²` (VMC) and the collocation Gaussian mixture q (training-faithful; widths σ/√ω covering the
tails/near-node regions). 128 then 320 samples. Metric: condition number over the resolved
spectrum (rel_tol 1e-8, to avoid the 1e-12 float64 floor that pins every κ at ~1e12) + log-log
power-law slope.
**Results:** (i) Under **|Ψ|²**, the collocation operators are *mild* — N=6 ω=1: κ(S)=6.6e4,
κ(A_weak)=4.7e4, κ(A_strong)=4.8e3 (strong even BETTER than weak). (ii) Under the **mixture**, they
**explode**: same N=6 ω=1 gives κ(A_weak)≈5e4→**≳1e8**, κ(A_strong)≳1e8 (floor-exceeding at both 128
and 320 samples), while κ(S) stays moderate (2e6, full rank 319/320, genuinely resolved). So the
*same operator* gets a **≥2000× conditioning penalty purely from the sampling measure**.
**Interpretation:** The dominant conditioning lever is **the sampling measure, not the strong/weak
form**. `|Ψ|²`-sampling (VMC) is itself the preconditioner — it down-weights exactly the
low-density tail/near-node points where the residual Jacobian blows up; covering configuration space
(which collocation must do) is what wrecks conditioning. This explains the VMC SR-null (VMC already
well-conditioned → SR has nothing to do) AND why collocation needs continuation chains + ESS +
natural gradient (it fights a κ≳10⁸ operator). NOTE this **contradicts** `run_weak_form.py:8`'s
claim that weak-form "eliminates the conditioning catastrophe" — weak-form is *also* κ≳10⁸ under q.
**Caveats:** At 128–320 samples on an ~80k-param net the fine structure is not reliably resolved:
exact κ for A is floor-limited (≳1e8, true value larger), the strong-vs-weak `k⁴/k²` law is NOT
confirmed (ratios noisy 0.4–6.3; ranks/slopes threshold-dependent), and — importantly — the probe
uses the **unweighted** operator over q, whereas the trainer uses the **importance-weighted**
operator (`w=|Ψ|²/q`), which down-weights the bad tail points. So κ≳1e8 is an *upper bound*; the
real training operator lies between the |Ψ|² value (~5e4) and this, and the ESS controllers exist to
keep it usable. The robust, threshold-independent claim is the measure-driven ordering and the
≥2000× gap, not precise exponents.
**Decisive sharpening (same day) — the bottleneck is ESS collapse, a threshold-free number.**
Adding the importance-weighted operator (`w=|Ψ|²/q`) made *every* operator (S, A_weak, A_strong)
floor at ~1e8 — not conditioning but **ESS collapse** (effective rank ≈ ESS). Measuring ESS
directly (`scripts/exp_ess_collapse.py`, 4096-draw, N=6 cascade): ESS fraction
**3.0% (ω=1) → 2.3% (0.5) → 0.59% (0.1) → 0.11% (0.01)** — at ω=0.01 only ~5 of 4096 points
carry the weight. Monotone, clean, unambiguous. This is the real collocation bottleneck: the broad
mixture q (needed to cover space) mismatches the sharp |Ψ|², so the importance-weighted estimator
runs on a handful of effective points. It unifies everything: VMC is easy (|Ψ|² sampling →
weights=1 → ESS=N, well-conditioned operator); collocation faces a dilemma between an
ill-conditioned *unweighted* operator (κ≳1e8) and an *ESS-collapsed* weighted one; and it
quantitatively explains the known "Adam+ESS beats natural-gradient at low ω" (SR needs a stable
Fisher → needs ESS) and the low-ω/large-N collocation failures.
**Output reference:** [results/analysis/2026-06-21_conditioning_A/](results/analysis/2026-06-21_conditioning_A/)
(summary_{psi2,mixture}.json, spectra_*.npz), [results/analysis/2026-06-21_ess_collapse/](results/analysis/2026-06-21_ess_collapse/)
(ess.json, fig_ess_collapse.png)
**Next question:** Phase 1/2 — does the load-time ESS *predict* the actual training failure (final
energy error) across ω and N? If ESS is the causal lever, the fix is a better proposal/adaptive
resampling (raise ESS), not a fancier optimizer — testable directly.

### [2026-06-20] — Decisive SR-vs-Adam mechanism (N=2, exact GT): NULL in the cusp-on regime

**Motivation:** Settle the long-muddy question "is SR better than Adam, and what does it do to the gradients?" with a pre-registered, seeded, tuned experiment measured against the *exact* N=2 ground state — not single-seed energy noise. Pre-registered: (+) SR collapses the stiff (low-NTK-eigenvalue) error Adam stalls on; (0) SR ≈ Adam → advantage is regime-confined.
**Method:** `scripts/exp_sr_mechanism.py`. Common short Adam warm-up, then branch into Adam-only vs SR-only from the *identical* checkpoint; 4 seeds. On a fixed probe set, log vs step: energy error, distance-to-exact ‖δ‖ (δ = log|Ψ_exact|−log|Ψ|, centred), and δ decomposed in a *fixed* NTK (S) eigenbasis (built once from the converged N=2 checkpoint; rank 124) split at the median eigenvalue into stiff (low-μ) / soft (high-μ) halves. SR: annealed lr 0.2→0.01, damping 1e-2→1e-4, trust-region 0.05→0.005, sample-space Woodbury solve.
**Results:** Across 4 seeds the two optimizers **tie**: final energy err Adam +0.047% vs SR +0.056%; final ‖δ‖ both → 0.015. The mechanism prediction **fails**: SR does not preferentially shrink the stiff modes — Adam's stiff-mode error decreases more (3.4e-3 vs SR 5.0e-3). The earlier single-seed cusp×SR result (Adam +0.100% → SR −0.016%) does **not** survive seeds.
**Interpretation:** With the **cusp prior**, the stiff high-frequency directions are already removed from the residual (consistent with the exact-alignment test, where the *plain* gradient was better aligned than SR), so there is no ill-conditioning left for SR's whitening to exploit — and Adam reaches the same floor. Conditioning lives in the *ansatz*, not the optimizer, in this regime. The "SR breaks the 0.1% wall" claim is retracted as single-seed noise.
**Caveats:** N=2 is nodeless (no fermion sign); cusp-on only; SR run used 256 SR-samples / 300 steps (cheap, GPU-contended); step-0 diag values are walker-equilibration noise — the reliable signal is the converged trend (a tie). Says nothing yet about cusp-OFF, low-ω, or high-N where var(E_L) is large.
**Output reference:** [results/analysis/2026-06-20_SRmech_N2/](results/analysis/2026-06-20_SRmech_N2/) (summary.json, fig_sr_mechanism.png, raw.json)
**Next question:** Rerun the identical protocol **cusp-OFF** (stiff cusp mode must be learned): if SR then collapses the stiff-mode error while Adam stalls, we have localized *where* SR earns its keep; if not, the honest thesis statement is "with a physics-informed ansatz, optimizer choice is largely immaterial."

### [2026-06-15b] — CTNN vs FFNN at matched high accuracy: variance is the discriminator

**Motivation:** Answer "why CTNN > FFNN" with both ansatze trained to the same (below-DMC) accuracy,
so the comparison is fair.
**Method:** DeepSet-big + backflow omega cascade (1->0.5->0.1->0.01), identical recipe to the CTNN-big
cascade (Adam + annealed SR, warm-started). Compare energy error and var(E_L) vs omega.
**Results:** Energies tied for omega>=0.1 (both below DMC within stderr): CTNN/DeepSet err =
w1 -0.028/-0.018, w0.5 -0.001/-0.011, w0.1 -0.009/+0.005. **var(E_L) is consistently lower for CTNN:
w1 1.09e-2 vs 2.51e-2 (2.3x), w0.5 4.6e-3 vs 1.0e-2 (2.2x), w0.1 5.6e-4 vs 1.1e-3 (2.0x), w0.01
4.1e-6 vs 3.0e-5 (7.4x).** At omega=0.01 (Wigner) CTNN is also lower in energy (-0.131% vs +0.022%,
~1.2e-3 Ha).
**Interpretation:** At matched accuracy, total energy is a poor discriminator; var(E_L) is the real
one. Since var(E_L)->0 only for an exact eigenstate, CTNN's 2-7x lower variance means a
closer-to-eigenstate (smoother) wavefunction — consistent with the message-passing 'smoothing'
(100%-kinetic) picture. The advantage is regime-dependent: ~2x at weak/intermediate correlation,
growing to ~7x (plus an energy edge) in the strongly-correlated Wigner regime. So 'CTNN > FFNN' is
about wavefunction *quality*, not (mostly) energy, and it strengthens with correlation.
**Caveats:** Single seed; N=6 only; var ratios at w0.01 are on tiny absolute numbers; need seed
repeats for error bars on the ratios.
**Output reference:** results/analysis/2026-06-15_N6_w*_deepset_big_bf_casc/ vs ..._ctnn_big_*/.
**Next question:** Do the variance ratios hold across seeds and grow further at N=12/N=20?

### [2026-06-15] — Accurate omega cascade (below DMC) + message-decode collapses at Wigner

**Motivation:** The first DMC-quality N=6 runs sat at +0.05% with var(E_L)~0.03 — not accurate enough;
omega=0.1 wouldn't train from scratch. Push to consistently below-DMC accuracy across omega and
analyse what the (now genuine) ground states encode.
**Method:** (1) Diagnosed the two limiters: SR was pinned at the trust-region clip every step
(bouncing) and the small net floored the variance. Added lr/trust-region annealing to train_sr,
sr_samples subsampling, chunked local_energy (exact-Laplacian OOM on big nets), and a higher-capacity
config (ctnn_vcycle_big, ~80k params). (2) Added --init warm-start and ran a sequential omega cascade
1->0.5->0.1->0.01 (CTNN-big + backflow + Adam + annealed SR). (3) Added --load to the depth driver and
ran depth analysis on the accurate checkpoints in parallel across GPUs.
**Results:**
- **Energies all below reference:** w1 -0.028%, w0.5 -0.001%, w0.1 -0.009%, w0.01 -0.131% (w0.01 has
  no DMC; below the thesis PINN+CTNN value). **var(E_L) collapses 1.1e-2 -> 4.6e-3 -> 5e-4 -> 3.9e-6**
  toward low omega. Annealed SR settles (|dtheta| 2e-3 -> 5e-5) instead of bouncing.
- **Cascade unlocks low omega:** omega=0.1/0.01 (which diverged/plateaued from scratch) converge in
  ~60 warm-started steps.
- **Message-decode vs omega (D6):** CTNN message linear-probe R^2 for local quantities falls from
  high omega to Wigner: nn_distance 0.91->0.17, density 0.73->0.13, coulomb 0.82->0.44, spin
  0.53->0.05. The message encodes the local environment where geometry fluctuates (high omega) and
  stops encoding it where geometry freezes (Wigner crystal) — mirrors the known 3-body collapse.
**Interpretation:** Accuracy was an optimisation+capacity issue, fixed by annealed SR + bigger net +
cascade. The message-decode trend is a clean "what the network learns reflects the physics" result:
the many-body representation is information-rich exactly where many-body correlation is dynamically
active. CTNN VMC+SR now reaches or beats DMC consistently across omega.
**Caveats:** Single seed; w0.01 ref is not DMC. CTNN-vs-FFNN at matched high accuracy still pending
(DeepSet-big cascade running). --load depth omits the lazy-vs-rich trajectory (no training ckpts).
**Output reference:** results/analysis/2026-06-15_N6_w{1,05,01,001}_ctnn_big_bf_{acc,casc}/ and
2026-06-15_depthLOAD_N6_w*_ctnn_big/.
**Next question:** Does CTNN beat DeepSet at matched high accuracy at low omega? Then N=12/N=20.

### [2026-06-14] — Depth layer + backflow reaches DMC; SR validated; N=6 CTNN-vs-FFNN (prelim)

**Motivation:** Move from "SR helps + it's a GS" to the mechanistic questions (what the network
learns, what SR does to gradient space, why CTNN>FFNN), and ensure analysed wavefunctions are
genuine (~DMC) ground states.
**Method:** Built the depth diagnostics (src/analysis/representation.py: NTK eigenmodes in real
space, delta_psi SR-vs-plain maps, effective coordinate, circuit decode, lazy-vs-rich CKA, message
decode), an additive cusp on/off flag, and a natural-gradient trainer (src/analysis/fast_sr.py:
vmap O-builder verified to machine precision vs _score_rows; sample-space Woodbury SR step;
train_sr warm-start polish). Ran N=2 depth; N=6 Jastrow-only CTNN vs DeepSet; then N=6 CTNN with
coordinate backflow (Adam + polish).
**Results:**
- **Mechanism (N=2):** NTK eigenmodes ordered by smoothness; the plain gradient = a smooth global
  tilt proportional to r, while the residual (= the SR update) is **localised at the cusp**. Cusp
  prior carries short range, learned net carries long range. Dominant effective coordinate = a
  monotone-in-r "correlation strength" knob. edge eff_rank~1.4; CKA(first,final)~0.93.
- **Backflow was the missing piece:** Jastrow-only N=6 omega=1 plateaued at +0.09% (fixed-node
  error); **CTNN + backflow reached E=20.1577 (-0.008%, below DMC 20.15932)**, zero-var extrap
  20.156 (-0.015%). So a nodeless Jastrow cannot fix the N>=6 nodal surface; backflow can.
- **SR validated:** clean *monotonic* descent (N=2: +0.77% -> +0.064% in 60 steps, var 7e-4 ->
  1.4e-5), unlike Adam's bounce. Sample-space SR solve is practical (~5-7 s/step).
- **N=6 CTNN vs DeepSet (Jastrow-only, sub-DMC, preliminary):** energies tied at omega=1 (CTNN
  20.1785 vs DeepSet 20.1737) -> message-passing gives no energy edge at weak correlation; but the
  CTNN message *decodes local physics* (probe R2: nn_distance 0.88, local_density 0.72); CTNN lazy
  (CKA 0.94) vs DeepSet rich (0.54).
**Interpretation:** "Gradient orthogonal to the Hamiltonian" is concrete and real-space: the plain
gradient occupies the single smooth soft NTK mode (global tilt), the physics lives in high-frequency
cusp-localised stiff modes; SR whitens and puts the update at the cusp. The CTNN advantage is
regime-dependent (vacuous at omega=1, 1 effective collective coordinate); expect it to emerge at
lower omega. To analyse genuine GS we need backflow (nodes) + SR (consistency).
**Caveats:** N=6 comparison so far is Jastrow-only and sub-DMC; redoing with backflow + SR-polish
(DMC quality) for both architectures, in progress. var(E_L)~0.04 at N=6 even at DMC energy (ansatz
not exact). Small CTNN config; single seed.
**Output reference:** results/analysis/2026-06-14_{depth_N2_w1_ctnn_vcycle, depth_N6_w1_ctnn_vcycle,
depth_N6_w1_deepset, N6_w1_ctnn_bf}/ ; DMC-quality comparison -> ..._depthDMC_N6_w1_*_bf/.
**Next question:** Does CTNN beat DeepSet at lower omega (Phase B/C multi-omega), where many-body
correlation matters? Does SR-polish lower var(E_L) and lock both at DMC?

### [2026-06-14] — Kernel-analysis program Phase A: N=2, omega=1.0 anchor

**Motivation:** Validate the entire kernel-picture analysis toolchain against an exact
ground truth before scaling. N=2, omega=1.0 has the analytic Taut solution
(E=3.0, Psi ~ exp(-(r1^2+r2^2)/2)(1+r12), Jastrow cusp dJ/dr|0 = 1).
**Method:** Built generalizable analysis package `src/analysis/` (reference exact 2e solver for
any omega via finite-volume radial diagonalisation; System builder reusing the repo Slater/psi_fn;
diagnostics for O/S/K spectrum, SR-vs-plain alignment, GS-quality, learned-correlation, exact
overlap; fast Adam-VMC trainer). Driver `scripts/run_phase_analysis.py` (any N, omega). Trained a
small CTNN-VCycle Jastrow (9,842 params) for N=2, omega=1.0 by Adam-VMC, 600 steps.
**Results:** E = 2.99586 +/- 0.00009 Ha (clipped est.; raw ~3.0 to within the variational bound);
**|<Psi_net|Psi_exact>|^2 = 0.999984**; learned Jastrow lies exactly on the exact curve, learned
cusp recovered. Kernel: **QGT/NTK effective rank ~1.2 out of 9,842 params (numerical rank ~175),
kappa(S) ~ 9.5e11**. **cos(SR, imaginary-time flow) = 1.000 vs cos(plain gradient, imaginary-time)
falling 0.26 -> 0.05 over training.** NTK-whitening figure: plain gradient weights modes by mu_a
(starves all but the top few); SR weights all supported modes equally; the physics residual has
power spread across all modes -> plain gradient is misaligned, SR is not.
**Interpretation:** Confirms, against exact truth, the core thesis spine: SR = NTK whitening, and it
steps along the projected imaginary-time (Hamiltonian) flow while the plain gradient does not, with
the gap driven by the extreme NTK ill-conditioning (kappa ~ 1e12). The ~1-dimensional effective
tangent space at the solution (eff_rank 1.2/9842) is the N=2 instance of the low-rank-correlator
result and is *why* SR is so effective. Expressivity is not the bottleneck at N=2 (cos_sr=1);
conditioning is.
**Caveats:** Energy -0.14% below exact is a MAD-clip bias in the estimator (removes high-E_L
coalescence spikes), not a variational violation; overlap/cusp confirm the wavefunction is exact.
cos_sr=1.0 is partly because n_eval_samples < numerical rank (overcomplete tangent space) -- the
robust, informative quantity is cos_plain. Single seed; CTNN-VCycle only. Adam (not SR) used for
training speed; SR remains the diagnostic and an optional polish.
**Output reference:** results/analysis/2026-06-14_N2_w1_ctnn_vcycle/ (initial validation run).
**Update (polish, same day):** added a driver-only final settle (single lower-lr Adam call; trainer
untouched), unclipped + zero-variance energy reporting, alignment on 2048 samples (> score-matrix
rank ~156 so cos_sr is a true representable fraction), and full data dumps (plot_data.npz + 5 CSVs +
summary.json; every plot input saved). Result: **E_unclipped = 3.000127 +/- 0.000024 Ha (+0.004%,
variational bound respected; clipped == unclipped), overlap^2 = 1.000000, cos(SR)=0.997 vs
cos(plain)=0.041 at convergence (plain gradient nearly orthogonal to the Hamiltonian flow).**
Output: results/analysis/2026-06-14_N2_w1_ctnn_vcycle_polished/.
**Next question:** Phase B -- sweep the full omega span at N=2 to watch kappa(S) and the
cos(SR)-vs-cos(plain) gap evolve through the Wigner crossover.

### [2026-05-16] — Architecture diagnostics: input attribution, effective rank, REINFORCE vs FD-Colloc

**Motivation:** The thesis methods section argues at length for specific design choices (safe pair features, short-range gate, REINFORCE loss) but had no empirical figures backing these claims. Two explicit `[TODO]` markers in results.tex promised ablation figures. This session was devoted to producing those figures using existing trained N=6 checkpoints — no retraining.

**Method:** Wrote `scripts/diagnose_input_attribution.py` (~820 lines). Four analyses, all on N=6 checkpoints already in `results/arch_colloc/`:
1. **Controlled radial scan**: slid electron 0 from r=0.02 to r=5 a_ho toward electron 1 in 20 random orientations; computed per-channel Jastrow attribution ∂logΨ_jas/∂channel at each r. Also computed total ||∂logΨ/∂x_0|| for four checkpoints (REINFORCE, FD-Colloc, no-gate, best ω=1.0).
2. **MCMC attribution by ω**: drew 1000 MCMC samples from |Ψ|² at ω=1.0 and ω=0.001; computed channel attributions (Jacobian wrt each input channel, normalised).
3. **Activation effective rank**: hooked edge and node embeddings, ran SVD on activation matrix, computed k_eff = (Σσ)²/Σσ².
4. **Gradient norms**: REINFORCE vs FD-Colloc, 300 MCMC configs, accumulated ∂logΨ/∂θ norms per chunk.

**Results:**
- **Feature attribution by regime**: at ω=1.0, |r| edge channel dominates (0.14); at ω=0.001, spin attribution jumps from 0.03→1.05 and y-position from 0.006→0.14. The Wigner crystal regime makes positional and spin ordering the dominant input signal — the network learns this automatically.
- **Dead channels**: all channels have non-negligible attribution; none are fully dead.
- **Effective rank**: node_embed eff_rank≈2.3/24, edge_embed eff_rank≈1.4/24 at ω=1.0. The network projects a 24-dimensional hidden space onto a ~2D manifold, consistent with the PCA/CKA analysis in results.tex §4.1.
- **REINFORCE vs FD-Colloc gradient norms**: REINFORCE mean ||∇_θ logΨ|| = 173, FD-Colloc = 413 (2.4× higher). FD-Colloc differentiates through kinetic energy (second derivatives of logΨ), amplifying parameter gradients — directly demonstrating the instability that REINFORCE avoids.

**Interpretation:** The four results together form a coherent story validating the architecture design:
1. The network spontaneously adapts input utilisation to the physical regime without explicit supervision.
2. The latent space is low-dimensional (eff_rank≈2), which explains why the collocation training needs fewer samples than VMC-SR: the loss landscape is intrinsically low-dimensional.
3. FD-Colloc's 2.4× gradient inflation is the mechanism behind its slower convergence and lower accuracy (energy 0.079% vs 0.364% but FD took 4× more epochs in the long-run comparison).

**Caveats:** The "near-coalescence" gradient comparison (r_min < 0.3 a_ho) returned NaN because MCMC samples at ω=0.1 have typical pair distances ~5 a_ho — no natural coalescence events. A more targeted comparison would need forced close-pair configurations, which the controlled scan provides in Figure A.

**Output reference:** `results/figures/architecture_diagnostics/architecture_diagnostics.pdf`

**Next question:** Write the CTNN architecture diagram (TODO in results.tex line 428) and integrate these four figures into the appropriate thesis sections.

---

### [2026-04-15] — N=20 low-omega ShellFlow collapse diagnosis and stabilized relaunch

**Motivation:** The first direct N=20 Jastrow+ShellFlow low-omega diagnostics failed with ESS pinned at 1, shell radii drifting outward, and NaN final evaluation. The immediate question was whether this was a shell-geometry choice problem or a deeper sampling/refit failure.
**Method:** Compared the four direct diagnostics at `omega={0.01, 0.001}` and `2shell/3shell`, inspected the trainer/refit implementation, then patched the code so ShellFlow refits on weighted raw candidate clouds rather than already-resampled points and skips refits under catastrophic importance-weight collapse. Relaunched the same four-run comparison with tempered/clipped resampling, adaptive oversampling, and mass-gate ceilings via [scripts/launch_shellflow_n20_jastrow_diag.sh](scripts/launch_shellflow_n20_jastrow_diag.sh).
**Results:** In the original run set under `outputs/2026-04-15_shellflow_n20_jastrow_diag_v1/`, all four runs showed `ESS=1` from epoch 0 onward, proposal-only training for hundreds of epochs, outward shell-radius drift, and NaN final metrics for completed `omega=0.01` jobs. In the stabilized relaunch under `outputs/2026-04-15_shellflow_n20_jastrow_diag_v2/`, epoch-0 ESS jumped to roughly `1969-1971` for `omega=0.01` and `3284-3286` for `omega=0.001`, with PSIS-khat still elevated (`0.58-0.87`) and energies still physically poor but no immediate ESS collapse.
**Interpretation:** The dominant failure was not 2-shell vs 3-shell geometry. The deeper issue was an implementation-level feedback loop: with ESS collapsed, the proposal refitter was learning from a degenerate resampled batch and amplifying collapse into shell drift. The relaunch shows that resampling stabilization plus weighted raw-candidate refits can restore usable ESS immediately, which is a necessary precondition for any honest architecture comparison.
**Caveats:** The relaunched jobs were only checked at the earliest epochs during this session; the current evidence is about ESS recovery, not final scientific quality. Initial energies remain wildly wrong, so restored ESS alone does not establish that the N=20 low-omega recipe is good.
**Output reference:** `outputs/2026-04-15_shellflow_n20_jastrow_diag_v1/`, `outputs/2026-04-15_shellflow_n20_jastrow_diag_v2/`, `scripts/launch_shellflow_n20_jastrow_diag.sh`
**Next question:** Does the stabilized relaunch maintain non-collapsed ESS beyond the first few epochs, and if so does either `2shell` or `3shell` show a credible advantage in energy/error rather than just sampling statistics?

### [2026-04-06] — Higher-N Phase 1 smoke execution and N=20 post-bugfix ESS gate

**Motivation:** Execute the active phase of the higher-N scaling plan to test whether the N=6 DiagFisher+REINFORCE win transfers to N=12 and to check if N=20 is still blocked by sampling quality after the importance-sampling bugfix.
**Method:** Added and ran [scripts/launch_higher_n_phase1.sh](scripts/launch_higher_n_phase1.sh) with six parallel jobs: N=12 DiagFisher smokes at omega {0.1, 0.5, 1.0}, one N=12 Adam+REINFORCE control at omega 0.1, and two N=20 Adam+REINFORCE diagnostics at omega {0.1, 1.0}; generated summary artifact at `outputs/higher_n/phase1/phase1_summary.txt`.
**Results:** All six runs completed (N=12: 100 epochs each, N=20: 200 epochs each). N=12 best VMC errors were +0.292% (w0.1, DiagFisher), +0.090% (w0.5, DiagFisher), +0.056% (w1.0, DiagFisher), and +0.287% (w0.1, Adam control). N=20 ESS means were 6.75 (w0.1) and 4.80 (w1.0), with minima of 1 in both runs; N=20 best VMC errors remained very large (+64.662% at w0.1, +34.036% at w1.0).
**Interpretation:** Phase 1 passed for N=12 recipe viability (no failures, expected artifacts). For N=20, post-bugfix behavior remains strongly sampling-limited at omega 1.0 (ESS below planned gate), so Phase 3 of the higher-N plan remains conditionally blocked unless the ESS gate is relaxed.
**Caveats:** These are short diagnostics (100-200 epochs) and not final quality runs; VMC sampling during phase checks was 10k, not heavy-VMC 100k.
**Output reference:** `outputs/higher_n/phase1/`, `outputs/higher_n/phase1/phase1_summary.txt`
**Next question:** Should we proceed directly to Phase 2 (N=12 full campaign) while deferring N=20 to a sampling-focused plan, or run an additional N=20 diagnostic with stronger oversampling before deciding?

### [2026-03-29] — Consistency campaign Phase 0-2 synthesis and queued Phase 3 intervention matrix

**Motivation:** The consistency campaign needed two things before pushing deeper: a trustworthy synthesis of what Phases 0-2 actually established, and an automatic Phase 3 launch path that preserves diagnostic gates without requiring manual babysitting.
**Method:** Consolidated the campaign into a written report, added reward-normalized REINFORCE support and LR warmup controls to the trainer, prepared `scripts/launch_consistency_phase3.sh`, and queued `scripts/queue_consistency_phase3_after_phase2.sh` in tmux session `consistency_p3` so Phase 3 starts only after Phase 2 completion markers and heavy-VMC eval are present.
**Results:**
- Phase 2 heavy-VMC eval finished for six completed jobs.
- Best completed `omega=0.1` Phase 2 result is `diag_fdcolloc_n6w01` at `+0.236%`, with `diag_reinf_n6w01` close behind at `+0.271%`.
- Completed `omega=0.001` jobs remain far from target: `diag_ess_n6w001` at `+2.158%` and `diag_snr_n6w001` at `+2.626%`.
- Phase 3 is queued but blocked correctly on the two pending Phase 2D summaries (`diag_scratch_n6w01`, `diag_xfer_n6w01`).
**Interpretation:** The campaign now has a disciplined bridge from diagnosis into intervention. The strongest current signal is that removing ESS gating allows useful movement, while estimator choice at `omega=0.1` matters but only modestly so far. Phase 3 is appropriately broad because no single Phase 2 branch has yet produced a decisive win.
**Caveats:** Phase 2D is still incomplete, so transfer-basin conclusions remain provisional. Phase 3C and 3D introduce new trainer controls and therefore need fresh empirical validation.
**Output reference:** `outputs/consistency_campaign/CONSISTENCY_CAMPAIGN_REPORT_2026-03-29.md`, `outputs/consistency_campaign/phase2/eval_summary.json`, `scripts/launch_consistency_phase3.sh`, `scripts/queue_consistency_phase3_after_phase2.sh`
**Next question:** Which Phase 3 branch produces the first heavy-VMC result that materially outperforms `+0.236%` at N=6 `omega=0.1`, and does any branch shrink the `omega=0.001` error by an order of magnitude?

### [2026-03-28] — Adaptive-sigma deployment and 8h N=12 low-omega rescue campaign

**Motivation:** Determine whether low-omega transfer failure is primarily a sampling-overlap issue and recover N=12 `omega=0.001` transfer within a strict 8-hour wall-clock budget.
**Method:** Implemented adaptive proposal-width activation in trainer runtime, validated with targeted ESS diagnostics, then executed tmux multi-profile N=12 campaigns (A/B/C/D) over available GPUs with bridge `omega=0.005 -> transfer omega=0.001` and hard timeout limits.
**Results:**
- Bridge completed for profiles A/C/D with final energies around `1.58-1.85` (DMC shown as NaN for those logs).
- Profile B bridge did not reach completion marker.
- Transfer stages (A/C/D) started but were dominated by repeated `ESS < min_ess` skip/revert loops and did not produce successful final transfer checkpoints.
- Only bridge checkpoints were produced for v17 (`v17_n12w0005_bridge_A/C/D.pt`), with no transfer checkpoint artifacts.
**What the numbers actually mean:** Adaptive sampling helped overlap diagnostics in easier settings, but did not by itself unlock robust N=12 `omega=0.001` optimization under current gating/training policy.
**What we cannot explain:** Why N=12 transfer remains trapped in skip loops after adaptation, and whether the blocker is primarily overlap, ESS thresholding policy, reference mismatch, or interaction among all three.
**Caveats:** Low-omega reporting is constrained by missing/NaN DMC handling in some N=12 contexts; worker summary logs were incomplete due wrapper-shell exit behavior, so completion state required reconstruction from stage logs.
**What a skeptic would say:** This session delivered infrastructure and diagnostics improvements but still no end-to-end scientific win at the target regime.
**Output reference:** `outputs/2026-03-27_1036_campaign_v17_n12_8h/`, `results/arch_colloc/v17_n12w0005_bridge_A.pt`, `results/arch_colloc/v17_n12w0005_bridge_C.pt`, `results/arch_colloc/v17_n12w0005_bridge_D.pt`
**Next question:** Which minimal transfer policy change (ESS floor, oversample, gating schedule, or reference handling) yields measurable accepted-step progress for N=12 `omega=0.001` in a short ablation run?

### [2026-03-26] — Low-omega REINFORCE-only reruns: `omega=0.01` stable, `omega=0.001` still failing

**Motivation:** Test whether removing SR and enforcing focused low-omega schedules can recover target quality, then continue to harder regime transfer and higher N.
**Method:** Ran sequential low-omega REINFORCE-only chains in tmux (`v13`, `v14`) and then launched higher-N chain (`v15`) using staged `omega=0.01` warmup/polish followed by `omega=0.001` transfer. Avoided `omega=0.005/0.002` in focused reruns when requested.
**Results:**
- `v14_n2w001_polish_reinf`: `E=0.073983 +/- 0.000022`, `err=+0.194%` (good at `omega=0.01`)
- `v14_n2w0001_transfer_reinf`: `E=0.013899 +/- 0.000002`, `err=+90.391%` (still bad at `omega=0.001`)
- Prior chain (`v13`) showed the same pattern: good at `0.01`, large error at `0.001`.
**What the numbers actually mean:** The pipeline can still optimize and hold precision in the easier low-omega anchor (`0.01`) but fails to cross to the ultra-low regime (`0.001`) under current transfer/sampling/training settings.
**What we cannot explain:** Why repeated retunes and direct `0.01 -> 0.001` transfer preserve high error near +90% instead of moving into the expected band.
**Caveats:** Some low-omega reporting in previous runs was confounded by snapped references for unsupported omega values (`0.005/0.002`), but the `0.001` reference is explicitly present and this failure is real under current configuration.
**What a skeptic would say:** This session improved orchestration discipline but not scientific understanding of the `0.001` failure mechanism; too much effort went into relaunching without a deep diagnostic gate.
**Output reference:** `outputs/2026-03-25_0933_campaign_v13_lowomega_reinforce_only/`, `outputs/2026-03-26_0723_campaign_v14_lowomega_2stage_reinforce_only/`, `outputs/2026-03-26_1403_campaign_v15_n6_lowomega_2stage_reinforce_only/`
**Next question:** Which foundation-layer issue dominates `omega=0.001` failure now: proposal overlap/ESS collapse, reward-weight distribution pathology, or mismatch in transfer initialization between regimes?

### [2026-03-24] — Critical importance sampling bug: component density vs mixture density

**Motivation:** Low-omega VMC training consistently produced high errors despite the physics being numerically simpler (negligible kinetic energy, irrelevant Coulomb singularity). Higher N also degraded faster than expected. External sources confirmed ω≪1 should be *easier* than ω=1. Something was fundamentally wrong.

**Method:** Systematic audit of the sampling and optimization pipeline. Traced the importance weight computation from `sample_mixture()` through `importance_resample()` to the loss functions.

**Results:** `sample_mixture()` returned `log_q` as the density of the individual Gaussian component that generated each sample, not the mixture density q(x) = (1/K)Σ_k N(x;0,σ_k²I). The correct computation (via `logsumexp` over all components) was already implemented in `eval_mixture_logq()` but was never called — dead code since it was written.

Quantitative impact: For a point at distance r from the origin, with components σ₁=0.8/√ω and σ₂=2.0/√ω, the log-density difference between component and mixture can be O(Nd) where Nd = N×d is the dimensionality. At N=6, d=2, ω=0.01, this is a bias of O(12) nats in log-space — the importance weights were off by factors of e^12 ≈ 160,000.

**Interpretation:** This single bug is the root cause of:
1. Low-omega training failure (exponentially wrong weights as ω→0)
2. Higher-N degradation (exponentially wrong weights as N×d increases)
3. SR instability at low omega (noisy gradients from biased sampling)
4. The "necessity" of disabling SR below ω=0.1 (a downstream workaround)

The reason ω=1.0 worked tolerably well: the three Gaussian widths (0.8, 1.3, 2.0 in oscillator units) overlap significantly, so mixture density ≈ component density to within a modest factor.

**Caveats:** All previous checkpoints were trained with biased sampling. Results may have been lucky (the bias sometimes partially cancelled) or systematically too high. Retraining is needed to establish true baselines.

**Output reference:** No experiment outputs yet — this is a code fix. Validation runs pending.

**Next question:** Re-run the full N×ω grid with correct sampling and SR enabled. Expect dramatic improvement at low omega and higher N. Also worth revisiting whether Langevin failure (2026-03-19) was partly caused by this bug (the Langevin path set lq=0, bypassing the component-density bug but introducing flat-proposal bias).

### [2026-03-19/20] — Langevin proposal sampling: implementation, failure, and lessons learned

**Motivation:** At N=20 with low ω, the Gaussian mixture proposal has near-zero overlap with |Ψ|² in 40 dimensions, producing ESS≈1 on every epoch. Hypothesis: running K steps of overdamped Langevin dynamics on proposal samples before importance resampling would push them toward high-|Ψ|² regions and fix the sampling catastrophe.
**Method:** Implemented `langevin_refine_samples()` in [src/functions/Neural_Networks.py](src/functions/Neural_Networks.py). The update rule is x' = x + ε·∇log|Ψ|² + √(2ε)·η, with per-sample gradient norm clipping (clip=1.0) and NaN guards. After Langevin, use flat proposal weights (lq=0) since samples should be approximately |Ψ|²-distributed. Tested with K=10-20 steps at ε=0.01-0.05 across N=6 ω=0.001, N=20 ω=0.1, and N=20 ω=1.0.
**Results:** Langevin was consistently worse than standard importance resampling:
| Config | Standard (VMC) | Langevin (VMC) |
|--------|---------------|----------------|
| N=20 ω=0.1 | +5.4% | +152% (catastrophic) |
| N=20 ω=1.0 | +1.3% | +5.2% |
| N=6 ω=0.001 | +0.22% | +0.30% |
**Interpretation:** K=10-20 Langevin steps from a Gaussian starting point in 40D are far from equilibrium. The resulting sample distribution is neither |Ψ|² nor q(x), so the flat-proposal importance weights are wrong. The biased gradients push the wavefunction to match the biased distribution (positive feedback loop), explaining the +152% catastrophe at N=20 ω=0.1. Langevin proposal refinement requires either (a) much longer chains (K=100+, impractical), (b) a proper Metropolis-Hastings acceptance step, or (c) should be replaced with proper MCMC within training. This was a useful negative result.
**Caveats:** Only tested with short chains; properly equilibrated Langevin (or HMC) might work but at MCMC-level computational cost, negating the advantage of the collocation-only approach.
**Output reference:** [outputs/2026-03-19_1909_campaign_v3/logs/](outputs/2026-03-19_1909_campaign_v3/logs/)
**Next question:** Is the sampling bottleneck the fundamental limit for collocation at high N, or can architecture improvements (backflow) compensate without fixing sampling?

### [2026-03-19/20] — N=20 Jastrow polishing: significant accuracy improvement

**Motivation:** Historical N=20 Jastrow checkpoints from March 14 transfer campaigns showed +2.6% (ω=1.0), +7.0% (ω=0.5), +5.9% (ω=0.1). These appeared undertrained — could lower LR and patient polishing improve them?
**Method:** Resumed from best checkpoints with LR reduced 2-5x (1e-4 to 5e-5), relaxed rollback thresholds, Adam optimizer. Multiple seeds per omega for cross-validation. Ultra-polish at ω=1.0 uses n-coll=8192 and LR=5e-5.
**Results:**
| N | ω | Before | After (best VMC) | Improvement |
|---|---|--------|-----------------|-------------|
| 20 | 1.0 | +2.63% | **+1.32%** | 2.0× |
| 20 | 0.5 | +7.0% | **+2.38%** | 2.9× |
| 20 | 0.1 | +5.9% | **+5.44%** | 1.1× |
**Interpretation:** The March 14 checkpoints were far from convergence — they had capacity but needed more optimization with smaller LR. The Jastrow architecture at N=20 can reach ~1% at ω=1.0 and ~2.4% at ω=0.5. The ω=0.1 result (+5.4%) shows less improvement, suggesting the Jastrow ansatz may be approaching its capacity limit at this N/ω combination (or sampling quality is limiting). Ultra-polish run with 8192 collocation points is still running and may push ω=1.0 below 1%.
**Caveats:** VMC probes use 15-20k samples; final heavy evaluation will be needed for definitive numbers. The collocation energy often shows lower error than VMC (biased importance sampling), so VMC probes are the reliable metric.
**Output reference:** [outputs/2026-03-19_1909_campaign_v3/logs/](outputs/2026-03-19_1909_campaign_v3/logs/), checkpoints in [results/arch_colloc/](results/arch_colloc/)
**Next question:** Can the ultra-polish push N=20 ω=1.0 below 1%? Should we attempt backflow at N=20 with the smaller architecture (bf-hidden=48) now that we have a strong Jastrow warm-start?

### [2026-03-17] — Post-catch-22 synthesis in thesis appendix and hard-regime stabilisation rollout

**Motivation:** After resolving core BF-conditioning issues, the project needed two things: (i) a complete scientific narrative of what was tried and why, and (ii) a targeted stabilisation rollout for unresolved low-omega/high-N regimes.
**Method:** Added a new thesis appendix chapter (`app:postcatch22`) in `Thesis/appendix.tex` with an explicit map of attempted sampling schemes, architecture families, loss/objective styles, optimiser/preconditioner variants, and campaign chronology (natgrad sweeps, SR sweeps, cascade waves, coalescence matrix, and stabilisation runs). In parallel, launched `scripts/stabilize_hard_regimes.py` for N=6 ω=0.1, N=12 ω=0.5, and N=20 ω=1.0 using stricter CG-SR controls (damping anneal, trust-region caps), ESS-adaptive resampling, tempered/clipped importance weights, and rollback logic.
**Results:** Appendix chapter committed and pushed. Stabilisation rollout started successfully; N=6 ω=0.1 transfer branch is actively training near sub-1% error band early in run, while N=12/N=20 branches required relaunch after stale-process OOM and over-strict epoch-0 rollback settings. Corrected v2 launches are now active with relaxed blocking thresholds.
**Interpretation:** The scientific record is now aligned with actual campaign history rather than single-run anecdotes. Operationally, hard-regime stability is strongly coupled to process hygiene (stale workers) and guard aggressiveness (`min_ess`, rollback triggers), not only optimiser choice.
**Caveats:** N=12/N=20 stabilisation branches are in-progress; no final claim update yet. The hard-regime summary remains provisional until final heavy-VMC evaluations complete.
**Output reference:** `Thesis/appendix.tex`, `outputs/2026-03-17_1202_stabilized_hardregimes/`
**Next question:** Can the corrected hard-regime policy deliver monotonic improvement vs cascade baselines for N=12 ω=0.5 and N=20 ω=1.0 without reintroducing variance blowups?

### [2026-03-17] — CG-SR 12-hour campaign: closing the gap to DMC

**Motivation:** Previous work established CG-SR as the best optimizer (Final E=20.165, +0.029% from DMC at N=6 ω=1.0), but three gaps remained: (1) probe-eval discrepancy where VMC probes showed better energies than the final heavy eval, (2) complete failure at ω=0.1 (+43% error due to narrow Gaussian mixture sampling), and (3) no scaling to N=12/N=20.

**Method:** Three fixes applied before launching a 3-phase, 8-GPU, 12-hour campaign:
1. **Auto-widened sigma_fs for low omega** — when ω<0.5 and sigma_fs is at default (0.8,1.3,2.0), automatically widen: ω≤0.15 → (0.4,0.7,1.0,1.5,2.5,4.0), ω≤0.05 → (0.3,0.5,0.8,1.2,2.0,3.5,6.0).
2. **Heavier VMC evaluation** — final eval: burn_in 400→800, thin 3→5, sampler_steps 80→120. Probes: burn_in 200→400, thin 2→3, sampler_steps 40→60. Reduces selection bias (~1.5σ) from picking minimum of ~15 noisy probes.
3. **N=20 architecture fit** — bf-hidden=64, bf-layers=2, micro-batch=128, n-coll=1024 to fit 11GB GPUs.

Campaign structure (scripts/cgsr_campaign.py):
- Phase 1 (0-2h): Fix verification + smoke tests — all 8 GPUs
- Phase 2 (2-8h): Main training from Phase 1 checkpoints — 1200 epochs
- Phase 3 (8-12h): Ultra-low LR refinement — polish Phase 2 winners

Targets: N=6 × {ω=1.0, 0.5, 0.1}, N=12 × {ω=1.0, 0.5}, N=20 × {ω=1.0}. All CG-SR with damping annealing.

**Results (Phase 1 early, ~30 epochs):**
- N=6 ω=1.0: +0.02% (epoch 30) — effectively at DMC reference (20.159)
- N=6 ω=0.5: +0.84% and dropping (warm-started from bf_ctnn_vcycle.pt)
- N=6 ω=0.1: +45-55% (from scratch, sampling fix confirmed working — auto sigma_fs active)
- N=12 ω=1.0: +23.5% (epoch 0, ~45s/epoch)
- N=12 ω=0.5: +45% (epoch 0)
- N=20 ω=1.0: +37% (epoch 10, small arch, fits 11GB)

Campaign is running in tmux session `cgsr_camp`. Full results pending.

**Interpretation:** The sigma_fs auto-widening immediately fixed the ω=0.1 initialization (starting to train instead of stuck). The heavier VMC eval should close the probe-eval gap — Phase 1's N=6 ω=1.0 result will be the first definitive test. N=12/N=20 need many more epochs but are successfully training from scratch.

**Caveats:** Campaign still running. N=20 uses reduced architecture (64 hidden, 2 layers vs default 128 hidden, 3 layers) — may hit a lower accuracy ceiling. Low-ω runs need hundreds of epochs to converge from scratch.

**Output reference:** `outputs/2026-03-17_0031_cgsr_campaign/`

**Next question:** Does the heavier final eval eliminate the probe-eval gap? Can N=6 ω=0.1 reach <1% with CG-SR + widened sampling? Does N=20 with small arch get within 5% of DMC?

### [2026-03-15] — Seed-path fix and stability-tail ablation campaign launch

**Motivation:** Resolve long-standing suspicion that seed differences were not being honored, then test whether tail-regularized resampling improves stability and probe-to-final consistency.
**Method:** Changed config seed default to non-reseeding mode (`seed=None`), threaded trainer CLI seed into weak-form `setup/config.update`, validated post-setup RNG divergence for seeds 11 and 22, added two training controls (importance-weight tempering and log-weight clipping) plus optional selection-time mini-heavy VMC tie-break, and launched a 6-job campaign (`base_hi`, `reg_hi`, `base_lo`, `reg_lo`) via `scripts/stability_core_campaign.py` in tmux with structured outputs.
**Results:** Seed-path validation showed distinct post-setup torch/numpy streams for seed 11 vs 22 (where previously both collapsed to the same sequence after setup); campaign launched successfully with orchestrator + worker processes active and plan ETA around 1.73h wall-clock.
**Interpretation:** A concrete reproducibility threat was present and is now removed for weak-form runs; new campaign is positioned to test whether reduced resampling weight spikiness translates to improved final heavy-VMC behavior.
**Caveats:** Campaign outcomes are still pending; conclusions about regularization benefits are not yet available.
**Output reference:** `outputs/2026-03-15_1453_stability_core_campaign/`
**Next question:** Do regularized variants reduce rollback frequency and probe-final gap while preserving or improving final heavy-VMC error, especially at low omega?

### [2026-03-14] — DMC matrix validation + short collocation pipeline smoke

**Motivation:** Ensure collocation runner uses correct DMC energies across supported particle counts/frequencies before launching a long production run.
**Method:** Added DMC lookup tests and runner-level tests; executed a zero-epoch smoke matrix over all configured DMC table entries; executed a short end-to-end matrix run with 5 epochs per stage to validate orchestration and checkpoint flow.
**Results:** Pytest passed (`19 passed`); smoke matrix validated 19/19 configured entries with `rc=0` and correct `DMC reference` lines for N={2,6,12,20} and all configured omegas; short validation run in [outputs/2026-03-14_1130_overnight_auto](outputs/2026-03-14_1130_overnight_auto) completed Phase A jobs with `returncode=0` and expected checkpoint outputs.
**Interpretation:** DMC references are now correctly sourced from config for all supported combinations, and the refactored collocation/orchestrator pipeline is operational.
**Caveats:** Short validation used only 5 epochs and is not quality-indicative for final physics metrics.
**Output reference:** [outputs/2026-03-14_smoke_collocation/smoke_matrix.json](outputs/2026-03-14_smoke_collocation/smoke_matrix.json), [outputs/2026-03-14_1130_overnight_auto](outputs/2026-03-14_1130_overnight_auto), [outputs/2026-03-14_1135_overnight_auto](outputs/2026-03-14_1135_overnight_auto)
**Next question:** Should the production target matrix be expanded beyond N=6/12 to include N=2 and N=20 runs now that DMC coverage is validated?

## Codebase Snapshot

*This section is maintained by `explain-codebase.prompt.md`. Updated when the architecture changes significantly.*

### Structure

*(To be filled in after first codebase explanation session.)*

### Data flow

*(To be filled in.)*

### Key design decisions reflected in code

*(To be filled in.)*

### [2026-07-11b] — Staged recipe reaches thesis accuracy; and the Jastrow/backflow split is a SOFT (path-influenced) direction, not a fixed physical quantity

**Setup:** N=6, omega=1. Five arms trained to full SR polish with the thesis staged curriculum
(Jastrow -> [cusp] -> frozen-Jastrow backflow -> joint), bf_scale_init=0.7, zero_init_last=False:
PINN+CTNN-bf (seeds 0,1), PINN+conv-bf (seeds 0,1), and a no-cusp CTNN arm.

**Result 1 — accuracy (the training fix works).** Final SR energies vs ref 20.15932:
  ctnn s0  +0.012% | ctnn s1  -0.034% | no-cusp ctnn -0.005% | conv s0 +0.028% | conv s1 +0.001%.
All five within |0.034%| — thesis quality (the thesis reports ~0.008-0.08%). The small negatives are
the known MAD-clip bias (report unclipped; the variational principle forbids a true undershoot). This
is the payoff of the two training fixes: the tanh/COM annihilation and the joint-from-scratch starvation.

**Result 2 — the substitution finding (replaces the retracted rank story).** Final |dx|/ell:
  ctnn 0.119, 0.123 | no-cusp ctnn 0.149 | conv 0.068, 0.076.
The no-cusp CTNN arm ENTERED stage 3 at |dx|=0.344 (5x the cusp arm's 0.03) and CONVERGED to 0.149 at
the same energy as the cusp arm (0.119). So the equilibrium displacement is set by the optimisation,
not the initialisation: the correlation split between Jastrow and backflow is a soft, path-influenced
direction of the loss, not a fixed property of the ground state. This is WHY every earlier backflow-rank
/ |dx| claim reversed under a change of training protocol — those quantities are gauge-like. It also
explains the thesis checkpoint's |dx|=0.364 vs my ~0.12 at equal energy: different training path, same
state. A real, measured mechanism result, and a more honest one than the rank story it replaces.

**Architecture signature that DOES survive:** at matched energy the CTNN backflow settles ~1.7x larger
(0.12) than the conventional per-particle backflow (0.07). Measured on live, thesis-quality states, not
a starved or dead one. To be confirmed by the dE_backflow ablation (energy with Delta_x deleted) once
the low-omega tiers finish.

**Still running:** ctnn omega=0.1 (2 chains). Cascade omega=0.01 already done.

### [2026-07-16] — THE INVERSION: it is the CONVENTIONAL backflow that collapses at Wigner, not the message-passing one

**Setup:** N=6, thesis ansatz (PINN Jastrow x backflow), thesis omega grid, 2 seeds, both backflow
architectures trained with an identical staged recipe to thesis accuracy. Analyzer on live states.

| omega | err% ctnn/conv | d_eff ctnn/conv | BFrank ctnn/conv | dT_msg |
|-------|----------------|-----------------|------------------|--------|
| 1.0   | -0.02 / +0.07  |  4.2 /  6.4     |  9.1 / 10.5      | +0.60  |
| 0.1   | +0.02 / +0.29  |  7.8 / 10.2     |  7.6 / 10.1      | +0.07  |
| 0.01  | -0.03 / +0.58  |  9.8 /  1.1     |  9.9 /  1.0      | +0.007 |

**The finding:** at Wigner the CONVENTIONAL per-particle backflow collapses — displacement rank
10.5 -> 1.0 and the whole tangent space d_eff 6.4 -> 1.1 — while the message-passing CTNN backflow
HOLDS rank ~10 and d_eff ~10. Its energy error correspondingly balloons to +0.58% while CTNN stays
at ~0.03%. Both seeds agree at every omega.

**This inverts the retracted 2026-07-02i claim.** I had reported "the message-passing backflow
collapses to rank-1 at Wigner" — measured on a backflow that was actually the CONVENTIONAL one
(backflow_arch silently defaulted to "conv"), and additionally starved by joint-from-scratch training.
The observation of a rank-1 collapse at Wigner was REAL; the attribution was exactly backwards. This
is the rare case where a retraction returns as its own mirror image, now measured on correctly
labelled, thesis-quality states.

**Resolves the kinetic paradox (Aleksander's objection):** "messages buy kinetic energy" and "CTNN
helps most at low omega, where Coulomb dominates" cannot both be the same mechanism. Measured:
dT_msg = 0.60, 0.068, 0.007 at omega = 1, 0.1, 0.01 — i.e. CONSTANT at ~0.6-0.7 in units of omega,
but a SHRINKING share of the total energy (3.0% -> 1.9% -> 1.0%). So they are two different
mechanisms: messages buy KINETIC energy at high omega, and at Wigner they buy EXPRESSIVITY —
they prevent the tangent-space collapse that destroys the conventional backflow. The advantage at
low omega is structural, not kinetic. Aleksander's objection was correct and productive.

**Scale check (not a relative-error artifact):** in absolute energy the conv error is 0.0056 (w=1)
and 0.0038 (w=0.01) — comparable. In units of omega it goes 0.006 -> 0.38 (68x worse) vs CTNN
0.002 -> 0.03 (12x worse). Both degrade toward Wigner; the conventional backflow degrades far faster.

**Caveat, now fixed:** the first ablation pass evaluated the ablated wavefunction on samples drawn
from the UN-ablated |Psi|^2, giving no-backflow energies BELOW the exact ground state (-1.6%), which
is variationally impossible. Ablations now resample from their own distribution, and are decomposed
into kinetic and Coulomb parts. The dT_msg trend above is a fixed-configuration probe and is
unaffected; the dE numbers are being recomputed.

**Next:** multi-day N-scaling campaign (N=6/12/20 x thesis omega grid x both backflows x 2 seeds) to
test whether the collapse persists and how the gap scales with N.

### [2026-07-20] — CORRECTED ablation: what the backflow BUYS, and the kinetic -> Coulomb crossover

All ablations resampled from their own |Psi|^2 (the earlier pass reused un-ablated samples and gave
variationally impossible energies). N=6, thesis omega grid, 2 seeds, both architectures.

**dE = E(ablated) - E(full), i.e. what the backflow is worth. In units of omega:**

| omega | CTNN (s0/s1)   | conventional (s0/s1) |
|-------|----------------|----------------------|
| 1.0   | 0.05w / 0.07w  | 0.07w / 0.09w        |
| 0.1   | 0.30w / 0.40w  | 0.18w / 0.24w        |
| 0.01  | **1.16w/1.10w**| **0.02w / 0.11w**    |

**Finding 1 — the CTNN backflow becomes ~19x more valuable (in natural units) toward Wigner**
(0.06w -> 1.13w) while the conventional one does not grow. In ABSOLUTE energy at omega=0.01 the
conventional backflow buys +0.0002 and +0.0011 — it is worth essentially NOTHING, so that ansatz has
degenerated to a bare PINN Jastrow. That is the mechanism behind its +0.58% error, and it is the same
fact as its rank collapse (BFrank 1.0, d_eff 1.1): a backflow that buys nothing has no structure.

**Finding 2 — the mechanism CROSSES OVER from kinetic to Coulomb** (this is the resolution of
Aleksander's objection, and his physical intuition was correct):
  - omega=1: conv dT = +0.61/+0.67 with dVc NEGATIVE (-0.05/-0.14). The backflow buys KINETIC energy
    and slightly costs Coulomb.
  - omega=0.1: dT and dVc both positive and comparable.
  - omega=0.01: dT = -0.001/+0.0003 (zero), dVc = +0.0167/+0.0156. PURELY Coulomb; the backflow even
    costs a little kinetic energy.
So "message passing buys kinetic energy" is a HIGH-omega statement only. At Wigner the gain is
correlation energy, which is exactly what a per-particle backflow cannot supply.

**Finding 3 — at Wigner the messages ARE the backflow.** ctnn omega=0.01: backflow buys +0.0116,
messages buy +0.0116 (s0) and +0.0127 vs +0.0110 (s1). Zeroing the inter-particle maps is as damaging
as deleting the entire backflow. Without communication a backflow has nothing to offer when
correlation dominates.

**CAVEAT (do not over-read the T/V split at omega=1).** dT and dVc are differences of two independent
MCMC estimates. At omega=1, var(E_L) ~ 0.1 and the seeds disagree sharply for CTNN (dT = -0.277 vs
+0.205) even though dE agrees (+0.051 vs +0.073). So the high-omega T/V decomposition is NOT resolved
at n=1024; only the conventional arm's kinetic dominance (+0.61/+0.67, consistent) is trustworthy
there. At omega=0.01 var(E_L) ~ 1.8e-5 and both seeds agree closely, so the Wigner conclusion is solid.
A larger-n rerun is needed before quoting the omega=1 split.

**Status:** N-scaling campaign (N=6/12/20 x thesis omega grid x both backflows x 2 seeds) running on
all 8 GPUs, 0 failures. Open question it addresses: does the conventional backflow's collapse persist
and does the CTNN advantage keep growing with N?

**[2026-07-20 caveat] N=12 batch mismatch introduced by the OOM retry.** The CTNN backflow builds
B x N x N x 128 edge tensors, so at N=12 it used 10.8 of 11.3 GB and OOM'd; the orchestrator retried
at half size and it now runs healthily at batch 512 (5.6 GB). The conventional arm never OOM'd and
still runs at batch 1024. So the N=12 CTNN-vs-conv comparison is NOT batch-matched.
Direction of the bias: the smaller batch is a HANDICAP for CTNN (noisier gradients), so if CTNN still
wins at N=12 the conclusion is conservative and safe to report. If conv wins or ties, the result is
confounded and a batch-matched conv rerun (batch 512) is REQUIRED before any claim. Both arms get the
same step count and the same SR polish, and the final energy is a variational upper bound either way,
so the energy comparison remains meaningful — but the caveat must travel with the number.
Note also that the memory asymmetry is itself a finding worth stating in the thesis: the
message-passing backflow costs ~2x the memory of the per-particle one at N=12 (10.8 vs 5.6 GB),
which is the price of the N^2 edge construction.

### [2026-07-27] — SCALING CONFIRMED at N=12: CTNN holds full rank at Wigner, conventional collapses to 1

The N=12 CTNN cascade (which OOM'd in the first run) completed after the exact-Laplacian chunking fix.
Both backflows, both seeds, N=12, full omega grid, thesis-quality energies.

**Energies (CTNN, all within |0.04%|):** w=1 +0.007/-0.039%, w=0.5 -0.005/-0.017%, w=0.28 -0.027/-0.015%,
w=0.1 +0.008/-0.014%, w=0.01 +0.002/-0.018%. Versus conventional at w=0.01: +0.557/+0.572%. So the
~30x CTNN accuracy advantage at Wigner PERSISTS at N=12 (it was ~19x at N=6).

**Rank (BFrank = participation ratio of the backflow displacement):**
| omega | N=6 ctnn/conv | N=12 ctnn/conv |
|-------|---------------|----------------|
| 1.0   | 10.4 / 10.4   | 20.6 / 21.7    |
| 0.1   | 10.3 / 10.3   | 18.5 / 21.4    |
| 0.01  | ~10 / **1.0** | 20.5 / **1.0** |

At high omega both backflows use ~N independent displacement modes. At Wigner the CONVENTIONAL backflow
collapses to a single collective mode (BFrank -> 1.0 at BOTH N), while the CTNN backflow HOLDS full rank
(~10 at N=6, ~20 at N=12). The collapse is sharper at larger N (21x drop vs 10x). Tangent d_eff tells the
same story from the other side: toward Wigner CTNN's GROWS (3.9 -> 10.4 at N=12) while conv's collapses
(3.8 -> 1.1). This is the mechanism behind the energy gap, now established on two system sizes x two seeds.

**Interpretation for the thesis:** the per-particle backflow can only express a rank-1 collective shift
once correlations dominate — it has no channel to place electrons relative to each other. The
message-passing backflow keeps a full-rank, per-particle-differentiated displacement because its
inter-particle messages carry exactly that relational information (consistent with the earlier ablation:
at Wigner the messages account for the backflow's ENTIRE energy contribution). Kinetic at high omega,
relational/Coulomb at Wigner.

**Status:** N=6 and N=12 complete (both arch, both seeds). N=20 was OOMing in the SR exact Laplacian even
at reduced batch; fixed by lowering the Laplacian chunk 256->64 (isolated test: N=20 Laplacian peak
4.4 -> 1.1 GB). N=20 restarted; watching the first SR polish to confirm it clears.

### [2026-08-11] — Q2/Q3 campaign live; first N=2 signal (both campaigns running post GPU-restore)

The GPU driver was restored (580.178.04). Two campaigns now run in parallel: N=20 scaling (GPUs 4-7)
and the Q2/Q3 2x2 (GPUs 0-3). Three bugs were caught by smoke-testing before the multi-day launch and
fixed: N=20 CTNN polish-batch OOM (no inflation at N>=20), colloc_sr None-grad crash (zero-fill missing
grads), and the VMC-SR cell wrongly using the legacy CG-SR trainer (swapped to fast_sr.train_sr, which
then hit the EXACT N=2 energies).

**Q2 (SR vs Adam), first read at N=2 (VMC energies, ref w1=3.0 / w0.1=0.44079 / w0.01=0.073839):**
  w1:   adam 3.000078  sr 2.999967     w0.1: adam 0.440815  sr 0.440841
  w0.01: adam 0.073836 sr 0.073837
At N=2 SR and Adam are INDISTINGUISHABLE — both reach the exact ground state (var ~1e-9 at Wigner).
There is no room for an optimizer to separate on a 2-electron problem; Q2's discriminating regime is
larger N. Consistent with the earlier "SR's edge washes into seed noise" observation. The N=6 cells
(running) are where Q2 is actually decided.

**Q3 (colloc vs VMC), N=2:** the E_weak readout is the biased importance-sampled energy (misleading —
judge by overlap, not E_weak). Fixed-proposal collocation ESS stays healthy (~0.32-0.42) across all
omega at N=2, INCLUDING w=0.01 — so the ESS collapse is an N-and-low-omega effect expected at N=6, not
N=2. The same-state overlap^2(vmc,colloc) is computing (the real Q3 test).

**Status:** N=20 6/16 (conv through w0.28 both seeds; ctnn building). Q2/Q3 18/48 cells, all N=2 s0
groups complete. Both healthy on all 8 GPUs.

### [2026-08-11b] — Q3 answered at N=2: VMC and collocation reach the SAME state (overlap^2 > 0.997)

The same-state overlap^2(vmc, colloc) at N=2, both optimizers, both seeds:
  w1:   adam 0.99945 / sr 0.99859    w0.1: adam 0.99899 / sr 0.99834
  w0.01: adam 0.99863 / sr 0.99745   (s1: adam 0.9986-0.9990; s1 sr cells still running)

overlap^2 > 0.997 EVERYWHERE. So at N=2 the training paradigm is EFFICIENCY-ONLY, not state-shaping:
VMC and collocation converge to the same wavefunction; collocation just pays ~5x variance to get there
(and, from the training logs, its E_weak readout is biased low so it must be judged by overlap, not by
E_weak). This extends the earlier single-point N=2 result (0.9992) across the full omega grid and both
optimizers, and kills for good the retracted "paradigm shapes internal structure" claim at N=2.

Subtle signal to watch: overlap^2 drifts DOWN toward Wigner (0.9994 -> 0.9986 at w=0.01). The paradigms
diverge slightly more as correlation strengthens. If that trend grows at N=6 (where collocation's ESS
may actually collapse at low omega), it becomes a real domain-of-validity finding for Q3.

**Progress:** N=20 8/16 (halfway); Q2/Q3 22/48. Both healthy.
