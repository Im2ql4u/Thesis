# Decisions

Permanent record of architectural and methodological decisions. Append-only — do not delete or rewrite entries. If a decision was reversed, add a new entry explaining why.

Each entry answers: what was decided, what the alternatives were, why this was chosen, and what this constrains going forward.

---

## Format

```
### [YYYY-MM-DD] — <short title>

**Decision:** <what was chosen>
**Alternatives considered:** <what else was on the table>
**Reasoning:** <why this was chosen over the alternatives>
**Constraints introduced:** <what this makes harder or impossible going forward>
**Confidence:** <high / medium / low — how sure we are this was right>
```

---

## Decisions

### [2026-06-22b] — Refine the spine: three co-equal questions (CTNN-vs-FFNN, SR-vs-Adam, Collocation-vs-VMC) with the kernel/dimension picture as shared lens (refines 2026-06-22a)

**Decision:** Reverse the framing of the earlier same-day decision that made *effective dimension* the
thesis spine. The thesis answers **three co-equal questions** — Q1 architecture (CTNN vs FFNN), Q2
optimizer (SR vs Adam), Q3 paradigm (collocation vs VMC) — each pursued to full mechanistic depth. The
tangent-kernel / dimension picture (O, S, K, d_eff, var(E_L), ESS, zero-variance) is the **shared
lens** serving all three, and the low-dimensional cusp-dominated tangent space is the expected
*synthesis* the three converge on, not the organizing axis. Execution is phased by **system size**
(N=2→6→12→20, ω=1→Wigner) with **all three questions run at every phase**, so none is starved. Roadmap
rewritten accordingly: `plans/2026-06-22_dimension-program-and-roadmap.md`.
**Alternatives considered:** Keep dimension as the spine with the optimizer/paradigm questions as a
cross-cutting note and a late phase (the 2026-06-22a draft); serialise the three questions
(would starve two while one is pursued).
**Reasoning:** Dimension was the user's *illustration of the desired depth*, not the intended axis;
collapsing the program onto it subordinated SR-vs-Adam and collocation-vs-VMC. Phasing by system size
with all three per phase preserves depth-before-breadth while keeping the three balanced, and the fair-
measurement protocol + GS-gate apply to all three (not only dimension).
**Constraints introduced:** Every phase's consolidation gate must report progress on all three
questions; no question may be deferred to "later" as a structural matter. The fair protocol and seed/
GS discipline bind all three. The 2026-06-22a common-probe protocol stands (it now serves all three).
**Confidence:** high — directly reflects the stated thesis aims.

### [2026-06-22a] — Adopt effective tangent-space dimension as the thesis spine; require a fair common-probe protocol for cross-architecture dimension claims

**Decision:** Reorganise the analysis program around one object — the effective dimension of the
variational tangent space `d_eff = eff-rank(S)` and the physical identity of its directions — and
make any cross-architecture or cross-(N,ω) `d_eff` comparison conform to a fixed fairness protocol:
(a) evaluate every architecture's `O` on a **common probe set** from a fixed reference measure (not
each net's own |Ψ|²); (b) report `d_eff` vs `n_samples` and accept only the converged plateau (never
report sample-capped `numerical_rank` as the dimension); (c) identical participation-ratio estimator /
rel_tol / centering; (d) report both **matched-params** and **matched-accuracy** axes; (e) ≥3 seeds +
GS-gate; (f) triangulate with latent intrinsic dimension and message rank. Roadmap:
`plans/2026-06-22_dimension-program-and-roadmap.md`.
**Alternatives considered:** Keep the three angles as separate sub-projects; keep measuring κ(S) as
the conditioning/advantage metric; keep the own-density QGT as the comparison measure; spin up a new
plan per result (the recurring re-planning habit).
**Reasoning:** The CTNN-advantage, the SR conditioning story, the "what is learned" question and the
collocation/Laplacian (zero-variance) thread are all faces of the low-dimensional tangent space — one
spine is more defensible and stops the narrative thrashing. κ(S) was found NOT to discriminate cleanly
across DeepSet sizes (eff-rank and var(E_L) do). The own-density QGT confounds "geometry of the
ansatz" with "where it puts mass", so cross-architecture dimension claims need a common probe set.
**Constraints introduced:** No `d_eff` comparison is reportable unless it meets the protocol;
single-seed / own-density / sample-capped numbers are motivating-only. New plans must extend this
spine rather than replace it (finish the consolidation gate before changing the spine).
**Confidence:** medium-high — the spine unifies the evidence; the headline gap still must survive the
fair protocol (Phase 1) before it is load-bearing.

### [2026-04-15] — Refit ShellFlow from weighted raw candidates and skip refits under catastrophic collapse

**Decision:** When using ShellFlow as an adaptive proposal, refit it from the raw candidate cloud with effective resampling weights, not from the already-resampled collocation batch; additionally, skip ShellFlow refits when ESS or resampling-mass diagnostics indicate catastrophic collapse.
**Alternatives considered:** Keep refitting from the resampled batch even when ESS is near 1; disable ShellFlow refits entirely for high-N low-omega work; treat the failure as a shell-template choice problem and retune geometry only.
**Reasoning:** In the N=20 low-omega diagnostics, ESS=1 meant the resampled batch was effectively copies of one survivor. Refitting ShellFlow on that batch poisoned the proposal and drove outward shell-radius drift without improving overlap. Weighted raw-candidate refits preserve the actual importance-sampling signal, and collapse skips prevent the refitter from learning from a degenerate distribution.
**Constraints introduced:** Future ShellFlow diagnostics must report ESS and resampling-mass diagnostics alongside proposal behavior; proposal-refit comparisons across old and new runs are not apples-to-apples because the refit target changed materially.
**Confidence:** high

### [2026-04-06] — Gate N=20 continuation on post-bugfix ESS evidence, proceed N=12 independently

**Decision:** After executing higher-N Phase 1 diagnostics, treat N=20 as conditionally blocked for optimizer-transfer continuation unless ESS is consistently above the gate, while allowing N=12 full-campaign work to proceed.
**Alternatives considered:** Continue N=20 optimizer transfer immediately despite low ESS; halt all higher-N work until N=20 sampling is fixed; lower standards and accept any ESS above 1 as sufficient.
**Reasoning:** The post-bugfix diagnostics still show weak ESS at N=20 (mean 4.80 at omega=1.0, min 1), with very poor VMC outcomes; this indicates a Layer-1 sampling bottleneck remains plausible and would confound optimizer conclusions. N=12 diagnostics ran cleanly and are not blocked by this issue.
**Constraints introduced:** Future N=20 optimization claims must report ESS evidence and distinguish sampling improvements from optimizer improvements; N=20 long runs should not consume major compute budget without a documented ESS gate decision.
**Confidence:** medium-high

### [2026-03-29] — Gate Phase 3 launch on completed Phase 2 artifacts and heavy-VMC evaluation

**Decision:** Queue Phase 3 automatically, but only allow it to launch after all eight Phase 2 summary files exist and a Phase 2 heavy-VMC evaluation summary has been written.
**Alternatives considered:** Launch Phase 3 immediately after the first six Phase 2 jobs; use only in-training traces to choose whether to continue; wait for manual operator intervention before launching Phase 3.
**Reasoning:** Phase 2 is a diagnostic phase, and its value comes from evidence rather than momentum. Using file-backed completion markers and heavy-VMC evaluation preserves the evaluation gate while still keeping the campaign moving automatically.
**Constraints introduced:** Phase 3 can be delayed by unrelated GPU occupancy affecting late Phase 2 jobs; queue logic now depends on summary/eval artifact paths remaining stable.
**Confidence:** high

### [2026-03-28] — Enable adaptive low-omega proposal widening in trainer runtime path

**Decision:** Add and use adaptive `sigma_fs` selection in the active weak-form training loop so low-omega runs widen proposal mixtures automatically by regime.
**Alternatives considered:** Keep static proposal widths and rely on larger oversampling or longer campaigns; bypass runtime adaptation and only tune per-script CLI values.
**Reasoning:** Diagnostics showed the intended adaptive behavior was documented but not actually applied in the executed runtime path, while low-omega transfer remained ESS-starved.
**Constraints introduced:** Results before this fix are not directly comparable to post-fix behavior for low-omega transfer; future evaluations must report whether adaptive sigma was active.
**Confidence:** medium

### [2026-03-26] — Enforce REINFORCE-only policy for low-omega training

**Decision:** For low-omega targeted runs, do not use SR/natural-gradient paths; execute REINFORCE/Adam-only schedules.
**Alternatives considered:** Keep SR enabled with damping/trust-region retunes; mixed-policy runs where some low-omega stages use SR.
**Reasoning:** Recent low-omega SR attempts produced poor outcomes and consumed iteration budget without improving target regimes; enforcing one policy reduces confounding while debugging the remaining failure at `omega=0.001`.
**Constraints introduced:** May slow convergence if SR would have helped in some low-omega sub-regimes; comparisons with older mixed-policy logs become less direct.
**Confidence:** medium

### [2026-03-26] — For low-omega transfer, gate on `omega=0.01` quality before `omega=0.001`

**Decision:** Use a staged rule: first push `omega=0.01` toward the ~0.1% error band at the current N, then transfer to `omega=0.001`; skip intermediate `omega=0.005/0.002` unless explicitly requested.
**Alternatives considered:** Full cascade `0.01 -> 0.005 -> 0.002 -> 0.001`; direct `0.01 -> 0.001` without checking `0.01` quality.
**Reasoning:** Intermediate steps previously added runtime without resolving the core `0.001` failure; explicit stage-gating creates a measurable quality checkpoint before entering the hardest regime.
**Constraints introduced:** If `0.01` is not predictive of `0.001`, this policy can still fail while increasing time spent polishing an easier regime.
**Confidence:** medium-low

### [2026-03-24] — Fix importance sampling weights; re-enable SR at all omega

**Decision:** Use correct Gaussian mixture density (logsumexp over components) for importance weights, not per-component density. Re-enable SR/natural gradient at all omega values. Increase CG iterations to 100 and fisher-subsample to 1024.
**Alternatives considered:** (a) Keep Adam-only for low omega and try other fixes (more oversampling, weight clipping) — rejected because these are band-aids over the root cause; (b) Replace Gaussian mixture with single Gaussian — rejected because the mixture is correct in principle, only the density evaluation was wrong; (c) Switch to MCMC sampling — unnecessary now that importance sampling weights are correct.
**Reasoning:** The component-density bug exponentially corrupted importance weights with increasing dimensionality (N×d) and decreasing omega (larger spread between component widths). This single bug explains the "low-omega catch-22" and the "higher-N degradation" simultaneously. The SR disable was a downstream workaround for gradient noise caused by this bug.
**Constraints introduced:** None — this is a pure bugfix. All existing checkpoints were trained with biased sampling, so their quality may improve on retraining.
**Confidence:** high — the math is unambiguous (mixture density ≠ component density), and the `eval_mixture_logq` function that does it correctly was already written but never wired in.

### [2026-03-24] — Reversal: SR is now allowed at low omega (reverses implicit decision from ~2026-03-17)

**Decision:** Remove the forced SR disable for ω ≤ 0.1. SR is now permitted at all omega values.
**Reasoning:** The instability that motivated the disable was caused by the importance sampling bug, not by SR itself. With correct weights, SR should be stable and is essential for navigating the ill-conditioned landscape at small energy scales.
**Confidence:** high — contingent on the importance sampling fix being correct, which it is.

### [2026-03-19] — Abandon Langevin proposal refinement; stick with Gaussian mixture + importance resampling

**Decision:** Do not use Langevin dynamics to refine proposal samples before importance resampling. Keep the standard Gaussian mixture proposal with ESS-adaptive oversampling.
**Alternatives considered:** (a) Langevin refinement with K=10-20 steps (tested and failed); (b) Longer Langevin chains K=100+ (too expensive — negates collocation advantage); (c) MALA with Metropolis correction (also expensive); (d) Normalizing flow as proposal (not implemented, high complexity).
**Reasoning:** Short Langevin chains produce non-equilibrium samples that bias the gradient signal. At N=20 ω=0.1, this caused +152% VMC error (catastrophic divergence). The bias creates a positive feedback loop where the wavefunction concentrates around the Langevin-biased region. Meanwhile, simple polishing of existing checkpoints with lower LR produces much better results (+1.3% vs +5.2% at N=20 ω=1.0).
**Constraints introduced:** The code for Langevin remains in the codebase (`--langevin-steps`, `--langevin-step-size`) but should not be used in production runs. Future sampling improvements should explore proper MCMC (if willing to accept the cost) or better proposal distributions (wider/adaptive sigma_fs, mixture of checkpoints).
**Confidence:** high

### [2026-03-19] — For N=20, use Jastrow-only architecture (not backflow)

**Decision:** For N=20 collocation training, use the Jastrow-only ansatz instead of CTNNBackflowNet.
**Alternatives considered:** BF with bf-hidden=128 (OOM), BF with bf-hidden=64 (converged to +18%, then stuck), BF with bf-hidden=48 (not tested alone).
**Reasoning:** Jastrow-only reaches +1.3% at ω=1.0 vs backflow's +18%. The BF architecture at N=20 has O(N²×hidden) edge features that consume most of the GPU memory, leaving little room for collocation points and oversampling. The Jastrow network is 3-4× smaller and allows higher n-coll and oversample settings, which directly improve gradient quality. The BF's additional capacity is not utilized because the gradient signal is too noisy at N=20.
**Constraints introduced:** The N=20 results cannot benefit from electron correlation beyond what the Jastrow factor captures. Future work could revisit BF once a strong Jastrow checkpoint provides a warm-start.
**Confidence:** medium-high (BF with proper warm-start from Jastrow might eventually help)

### [2026-03-17] — For hard-regime stabilisation, use non-blocking ESS gating early and rely on adaptive resampling + trust constraints

**Decision:** In early hard-regime runs (low $\omega$, high $N$), keep `min_ess` non-blocking (`0`) and disable strict epoch-level rollback error thresholds, while retaining ESS-floor adaptive oversampling, tempered/clipped resampling weights, and SR trust-region/max-step constraints.
**Alternatives considered:** Hard-gate low-ESS epochs (`min_ess>0`) and aggressive rollback on initial large error percentages from epoch 0.
**Reasoning:** Early hard-regime states naturally begin far from DMC; strict blocking and rollback thresholds caused repeated epoch-0 rollbacks/stalls without meaningful progress. Adaptive resampling and SR trust controls already constrain update pathology while allowing optimisation to move.
**Constraints introduced:** Future hard-regime campaign templates should not use strict `min_ess` gates or tight rollback-err thresholds at run start; if reintroduced, they must be staged/annealed rather than fixed from epoch 0.
**Confidence:** medium-high

### [2026-03-15] — Disable implicit config reseeding and require explicit run seed threading

**Decision:** Set `Config.seed` default to `None` and pass run seed explicitly from trainer CLI into `config.update(...)` call sites that need deterministic behavior.
**Alternatives considered:** Keep `Config.seed=0` default and rely on each script to reseed after every config update; maintain mixed behavior where some runners silently reseed and others do not.
**Reasoning:** Implicit reseeding from config updates can overwrite caller-provided seeds and collapse intended seed diversity across runs, invalidating robustness conclusions. Explicit seed threading makes seeding intent auditable and avoids hidden RNG resets.
**Constraints introduced:** Any runner requiring deterministic seeds must pass `seed=<value>` into `config.update(...)` explicitly; new scripts can no longer assume config updates preserve external RNG state.
**Confidence:** high

### [2026-03-15] — Keep close-pair samples and control influence instead of deleting them

**Decision:** Preserve near-coalescence configurations in collocation training and stabilize them through adaptive sampling/reweighting controls (ESS floor resampling, stratified replay, and BF short-range regularization) rather than removing close-pair samples from the training set.
**Alternatives considered:** Hard-cut close-pair samples with distance thresholds; soften Coulomb permanently in training objective; fully suppress hard-tail regions during optimization.
**Reasoning:** Close-pair regions carry physically important cusp/interactions and dominate difficult tail behavior. Removing them can make optimization appear stable while biasing the learned wavefunction and harming exact-VMC fidelity. Influence control preserves physics while reducing estimator instability.
**Constraints introduced:** Trainer policies must include explicit robustness controls for hard regions and cannot rely on sample deletion as a default stabilization method; any Coulomb softening must be temporary/annealed and clearly labeled as continuation.
**Confidence:** medium-high

### [2026-03-14] — Keep BF architecture fixed in trainer-comparison campaigns

**Decision:** In trainer-generalization experiments, BF jobs that do not resume from an existing BF checkpoint must still initialize backflow from the canonical `bf_ctnn_vcycle.pt` BF state/config rather than falling back to the larger default BF architecture.
**Alternatives considered:** Let BF jobs with only `init_jas` use the default backflow constructor in `src/run_weak_form.py`; tune trainer policies while silently changing BF architecture between target families.
**Reasoning:** The campaign is meant to compare trainers, not networks. Allowing low-ω or N=12 BF jobs to use the 182k-parameter default backflow while N=6 ω=1.0 uses the canonical 23,811-parameter BF makes the comparison scientifically meaningless. Fixing BF initialization preserves the intended “same network family, different trainer policy” setup.
**Constraints introduced:** Future BF trainer campaigns must pass a BF init checkpoint explicitly whenever they are not resuming from an existing BF run; the default BF constructor is no longer acceptable for cross-target trainer benchmarking unless architecture change is the explicit purpose of the experiment.
**Confidence:** high

### [2026-03-14] — Treat heavy exact VMC as the authoritative report metric

**Decision:** In experiment writeups, report final heavy exact VMC re-evaluations as the canonical result metric, and treat online probe VMC values only as checkpoint-selection diagnostics.
**Alternatives considered:** Quote the best online probe number as the result; mix collocation-batch, probe-VMC, and final-VMC numbers in the same summary table without distinction.
**Reasoning:** Historical logs such as `jas_reinf_v2` and `bf_hardfocus_v1b` show that cheap online probes can mis-rank checkpoints, sometimes by enough to change the scientific conclusion. The final 30k-sample exact VMC number is the only consistent cross-run basis for comparison.
**Constraints introduced:** Future reports must clearly label whether a number is a collocation estimate, a sparse online probe, or a final heavy exact VMC result; tables that omit that distinction are no longer acceptable.
**Confidence:** high

### [2026-03-14] — Treat the 20.161 result as a checkpoint-chain method, not a single run

**Decision:** Document the N=6, $\omega=1.0$ `20.161314` result as the outcome of a specific continuation chain `bf_ctnn_vcycle -> bf_joint_reinf_v3 -> bf_resume_lr_v1 -> bf_hardfocus_v1b`, and treat that lineage as part of the canonical reproduction recipe.
**Alternatives considered:** Summarize only the final checkpoint `bf_hardfocus_v1b.pt`; describe the result as a generic “CTNN BF + V-cycle Jastrow” outcome; present a simplified one-command approximation starting from `bf_ctnn_vcycle.pt`.
**Reasoning:** The forensic comparison of historical logs and failed reruns showed that the simplified one-command recipe lands near `20.1896`, not `20.1613`. The optimization path itself materially affects the outcome, so omitting the continuation stages makes the report inaccurate and non-reproducible.
**Constraints introduced:** Future writeups of best results must include upstream checkpoint provenance whenever continuation path is outcome-critical; “final checkpoint only” summaries are no longer acceptable for this N=6 BF result.
**Confidence:** high

### [2026-03-14] — Centralize DMC lookup and weak-form collocation primitives

**Decision:** Use [src/config.py](src/config.py) as the single source of truth for DMC references via shared helper `lookup_dmc_energy` in [src/functions/Neural_Networks.py](src/functions/Neural_Networks.py), and host reusable weak-form collocation primitives in the same shared module instead of keeping them runner-local.
**Alternatives considered:** Keep hardcoded fallback DMC table in [src/run_weak_form.py](src/run_weak_form.py); keep weak-form helper implementations duplicated in runner files.
**Reasoning:** Centralized references avoid silent NaN behavior for supported N/omega combinations; shared collocation helpers reduce duplication and make runner files thinner and easier to maintain.
**Constraints introduced:** Future DMC updates must be made in [src/config.py](src/config.py); collocation helper changes now affect all runners importing from [src/functions/Neural_Networks.py](src/functions/Neural_Networks.py).
**Confidence:** high

### [2026-06-13] — Analyse VMC/SR-trained ground states through the tangent-kernel picture

**Decision:** The new analysis program targets primarily VMC-trained (SR / Adam) wavefunctions that pass an explicit GS-quality gate, and interprets all three research angles (SR/natural-gradient, CTNN-vs-FFNN expressivity, what-is-learned) as properties of the per-sample log-derivative matrix `O` and its Gram matrices `S=OᵀO` (Fisher/QGT) and `K=OOᵀ` (NTK). Execution is phased by system size N=2→6→12→20 (N=2 first at ω=1.0 then full ω span), with a consolidation gate before each scale-up. Plan: `plans/2026-06-13_kernel-analysis-program.md`.
**Alternatives considered:** Analyse the existing collocation-trained checkpoints directly without a GS-quality gate; treat the three angles as independent diagnostic mini-projects; scale to all N immediately to maximise coverage.
**Reasoning:** Internal structure is only meaningful on genuine ground states, so analysis-grade GS verification must precede interpretation. VMC/SR wavefunctions are the most trustworthy GS approximations; collocation becomes a comparison axis (energy degeneracy vs representational degeneracy). The kernel picture gives one theory connecting trainability (spectrum of S/K), expressivity (span of O), and feature learning (eigenvector rotation), rather than a loose collection of plots. N=2 has an exact reference (E=3.0 at ω=1.0) to validate every tool before scaling.
**Constraints introduced:** No wavefunction is interpreted until it passes the §4 GS-quality gate (energy vs reference, var(E_L), ESS/khat, train-route agreement, seed stability); diagnostics must be expressible as statements about O/S/K; we do not advance to N+1 before a written consolidation of the current N.
**Confidence:** high

### [2026-06-14] — Train analysis-target wavefunctions with Adam-VMC; use SR only as diagnostic/polish

**Decision:** In the kernel-analysis program, train the ground-state wavefunctions with Adam-VMC (REINFORCE score-function gradient, one batched backward per step) rather than CG-SR. Stochastic reconfiguration and the per-sample score matrix O are reserved for (a) the kernel diagnostics and (b) an optional final polish. The exact 2-electron reference and all diagnostics live in the generalizable package `src/analysis/`, driven by `scripts/run_phase_analysis.py` (any even N, any omega).
**Alternatives considered:** Train with the repo CG-SR trainer (`train_model_sr_energy`) as the primary optimiser; reconstruct and analyse the legacy `official_models` checkpoints directly.
**Reasoning:** CG-SR's per-step cost is dominated by `_score_rows` building the per-sample score matrix (a Python loop of `total_rows` backprops); for the heavily over-parameterised small-N CTNN (9,842 params at N=2) this is minutes per step and the stable step-size window is narrow (kappa(S) ~ 1e12). Adam-VMC reached overlap^2 = 0.999984 with the exact N=2 ground state in 600 fast steps. The thesis scope explicitly includes Adam-trained wavefunctions; SR vs plain-gradient is a diagnostic comparison computed post-hoc on converged states, so training need not use SR. The legacy `official_models` lack saved hyperparameters, making reconstruction brittle; training our own guarantees verified, reproducible ground states.
**Constraints introduced:** Final reported energies must use the unclipped mean / variance extrapolation (the MAD-clipped training estimator biases E low by removing coalescence spikes); the clipped value is for stability only. Any wavefunction analysed must still pass the GS-quality gate (energy + overlap where exact is available).
**Confidence:** high
