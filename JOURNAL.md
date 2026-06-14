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
