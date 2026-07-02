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
