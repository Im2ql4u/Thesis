# Session Log

Rolling log of working sessions. Most recent entry at the top. Each entry is a snapshot of what happened, what changed, and what is pending.

The model reads this at session start. Write it as if you are handing off to yourself after a week away.

---

### [2026-03-19/20] — Langevin sampling attempt, N=20 polish campaign, negative Langevin result

**Goal:** (1) Implement and test Langevin-guided proposal sampling to address the ESS catastrophe at high N + low ω; (2) Push N=20 accuracy at ω=1.0, 0.5, 0.1 by polishing best Jastrow checkpoints; (3) Attempt N=12 at low omega (ω=0.05, ω=0.01).

**What was done:**
- Implemented `langevin_refine_samples()` in [src/functions/Neural_Networks.py](src/functions/Neural_Networks.py) — overdamped Langevin dynamics (x' = x + ε·∇log|Ψ|² + √(2ε)·η) applied to proposal samples before importance resampling.
- Fixed three crash bugs: (a) `@torch.no_grad()` decorator on `importance_resample` blocked autograd — wrapped Langevin call with `torch.enable_grad()`; (b) `eval_mixture_logq` at Langevin-shifted positions gave q≈0, crashing multinomial — switched to flat proposal after Langevin; (c) step size `ε/ω` exploded at low ω (ε=10 for ω=0.001) — removed 1/ω scaling, added per-sample gradient norm clipping and NaN guards.
- Added CLI flags `--langevin-steps` and `--langevin-step-size` to [src/run_weak_form.py](src/run_weak_form.py).
- Ran campaign v3 across 8 GPUs testing both Langevin and non-Langevin polish approaches.

**Key result — Langevin is negative:** Across all tested configurations, Langevin sampling performed worse than standard importance resampling:
- N=20 ω=0.1: Langevin VMC **+152%** vs non-Langevin **+5.4%** (catastrophic — short Langevin chains create biased sampling that destroys the wavefunction)
- N=20 ω=1.0: Langevin VMC +5.2% vs non-Langevin **+1.3%**
- N=6 ω=0.001: Langevin VMC +0.30% vs Adam baseline **+0.22%**
The root issue: K=10-20 Langevin steps from a Gaussian mixture starting point do NOT equilibrate to |Ψ|² in 40 dimensions. The biased sample distribution gives biased gradients, which push the wavefunction toward matching the biased distribution → positive feedback loop.

**Key result — N=20 polish works extremely well:** Simply resuming from best Jastrow checkpoints with lower LR produced dramatic improvements:
- N=20 ω=1.0: **+2.63% → +1.32%** (VMC), final eval +1.45%. New record.
- N=20 ω=0.5: **+7.0% → +2.38%** (VMC). New record.
- N=20 ω=0.1: **+5.9% → +5.44%** (VMC). Modest improvement, possibly at Jastrow capacity limit.

**What was not done:** N=12 at ω=0.05 and ω=0.01 did not converge (fresh start at +28%, cascade at +280%). N=20 at ω≤0.01 not attempted.

**Still running:** Ultra-polish of N=20 ω=1.0 (GPU 0, n-coll=8192, lr=5e-5), four N=20 polish runs at ω=0.5 and ω=0.1, N=12 ω=0.05 fresh start. See `outputs/2026-03-19_1909_campaign_v3/logs/`.

**Implicit assumptions made:** The flat proposal (lq=0) after Langevin is theoretically correct only when the Langevin chain fully equilibrates. With K=10-20 steps, it's a poor approximation.

**Next action:** Monitor ultra-polish; if N=20 ω=1.0 goes below 1%, cascade to ω=0.5 and ω=0.1. For N=12 low omega, may need BF architecture with careful warm-start (not Jastrow-only). Consider whether Langevin is worth pursuing with much longer chains (K=100+) or should be abandoned.

**Open questions:** Why does simple LR reduction + polish work so well for N=20? The checkpoints from jastrow_transfer campaigns (March 14) were undertrained — the network had capacity but hadn't converged. Is the Jastrow architecture hitting its limit at N=20 ω=0.1 (+5.4%), or would even more patient training push it lower?

---

### [2026-03-17] — CG-SR campaign: sampling fix, heavier eval, N=20 fit

**Goal:** Fix three blockers (low-ω sampling failure, probe-eval gap, N=20 OOM), then launch a 12-hour 3-phase campaign across N={6,12,20} and ω={0.1,0.5,1.0}.
**What was done:**
1. Added omega-dependent auto-widening of Gaussian mixture `sigma_fs` in `run_weak_form.py` (lines 1230-1238) — ω≤0.15 gets (0.4,0.7,1.0,1.5,2.5,4.0), ω≤0.05 gets even wider.
2. Increased VMC eval quality: final eval burn_in 400→800, thin 3→5, sampler_steps 80→120; probes burn_in 200→400, thin 2→3, sampler_steps 40→60.
3. Wrote `scripts/cgsr_campaign.py` — 3-phase orchestrator (verify→main→refine), 8 GPUs, 24 total jobs.
4. First launch: N=20 OOMed (CTNN backflow needs 4.69 GiB for edge features at N=20 on 11GB GPU). Killed, patched N=20 to use `--bf-hidden 64 --bf-layers 2 --micro-batch 128 --n-coll 1024`. Relaunched.
5. Confirmed auto sigma_fs working: N=6 ω=0.1 shows `[auto] omega=0.1 < 0.5 → widened sigma_fs to (0.4, 0.7, 1.0, 1.5, 2.5, 4.0)`.
6. Early Phase 1 results: N=6 ω=1.0 at +0.02% (epoch 30), N=6 ω=0.5 at +0.84%.
**What was not done:** Final campaign results pending (12h runtime). DECISIONS.md not updated (no new architectural decisions, only parameter tuning).
**Issues encountered:** N=20 CTNN backflow OOM on 11GB GPU — edge update concatenation requires O(N²×hidden) memory. Fixed with smaller arch.
**Workarounds in place:** N=20 uses bf-hidden=64/2 layers (vs default 128/3) — may limit accuracy ceiling. Proper fix would be chunked/sparse message passing.
**Implicit assumptions made:** Phase 2/3 resume logic assumes Phase 1 checkpoints save to `results/arch_colloc/{tag}.pt`. Duplicate campaign outputs from first aborted launch exist in `outputs/2026-03-17_0018_cgsr_campaign/`.
**Next action:** Monitor campaign completion. After 12h, analyze results table, identify which targets reached <0.1% of DMC, and determine next steps for remaining gaps.
**Open questions:** Will the heavier final eval actually close the probe-eval gap? Can N=20 with reduced architecture get within 5% of DMC?

---

## Format

```
### [YYYY-MM-DD] — <one-line summary of the session>

**Goal:** <what we set out to do>
**What was done:** <concrete summary — files changed, experiments run, decisions made>
**What was not done:** <what we planned but didn't get to, and why>
**Issues encountered:** <anything that went wrong or needed a workaround>
**Workarounds in place:** <any temporary solutions that need a proper fix later>
**Implicit assumptions made:** <anything the model assumed that was not stated>
**Next action:** <the single most important thing to do next session>
**Open questions:** <anything unresolved that might affect the next session>
```

---

## Log

### [2026-03-17] — Added thesis appendix on post-catch-22 campaign history and launched stabilized hard-regime rollout

**Goal:** Write a full thesis appendix that documents all post-catch-22 experimentation (sampling, losses, architectures, training styles), then push a structured stabilisation rollout for low-omega/high-N regimes.
**What was done:** Appended a new chapter to `Thesis/appendix.tex` (`app:postcatch22`) covering philosophy, full experiment-family map, chronology, major outcomes, and current frontier; created `scripts/stabilize_hard_regimes.py` and launched a 3-job hard-regime campaign (N=6 ω=0.1, N=12 ω=0.5, N=20 ω=1.0) with stricter SR trust limits, ESS-adaptive resampling, tempered importance weights, and rollback guards; identified and fixed first-launch stalls/OOM by relaxing overly strict `min_ess`/rollback error thresholds and killing stale GPU workers before relaunch.
**What was not done:** Did not complete the hard-regime campaign to final metrics yet; the new appendix is complete but still awaits integrated discussion in main-results chapters.
**Issues encountered:** OOM on relaunch due to stale worker processes occupying GPU memory; early hard-regime settings (`min_ess` + rollback error threshold) were too strict and caused epoch-0 rollback stalls on some branches.
**Workarounds in place:** Relaunched corrected v2 jobs with stale workers terminated and non-blocking ESS gating (`min_ess=0`) while retaining ESS-adaptive oversampling and trust-region controls.
**Implicit assumptions made:** Current campaign summaries and logs are treated as authoritative for post-catch chronology; where branches are still running, claims were explicitly bounded as provisional.
**Next action:** Monitor stabilized hard-regime jobs through at least first 50-100 epochs and compare against cascade baselines on trend, ESS/top-mass, rollback count, and VMC probe drift.
**Open questions:** Whether N=12 ω=0.5 and N=20 ω=1.0 can improve materially under strict SR with tempered resampling, and whether ω=0.01/0.001 require an intermediate cascade stage beyond current transfer chain.

### [2026-03-15] — Fixed seed-path reset bug and launched stability-first tail-control campaign

**Goal:** Eliminate suspected cross-run seed collapse, add tail-stability controls to weak-form training, and launch a focused campaign to test stability over generality.
**What was done:** Patched `src/config.py` so default `seed=None` (no implicit reseed), threaded CLI seed through `setup(..., seed=...)` in `src/run_weak_form.py`, and verified empirically that post-setup RNG streams differ for seed 11 vs 22; added resampling controls (`--resample-weight-temp`, `--resample-logw-clip-q`) and optional checkpoint tie-break eval (`--vmc-select-n`) in `src/run_weak_form.py` and `src/functions/Neural_Networks.py` with added history diagnostics (`ess_raw`, top-mass metrics, clipping threshold); added orchestrator `scripts/stability_core_campaign.py` and launched it in tmux (`hybrid:stability_core`) with structured `plan/results/summary` outputs and runtime ETA.
**What was not done:** Did not yet complete the new stability campaign or aggregate final conclusions from it.
**Issues encountered:** The earlier seed behavior was indeed being overwritten by config updates when seed was not passed; this explained suspicious cross-seed similarity risk.
**Workarounds in place:** Stability campaign includes explicit resample regularization and probe-selection tie-break for controlled variants while keeping baseline variants unchanged.
**Implicit assumptions made:** N=6 high/low-omega targeted runs are sufficient to validate whether tail regularization improves stability before reintroducing broader N/omega generalization goals.
**Next action:** Monitor `outputs/*_stability_core_campaign/summary.json` and compare baseline vs regularized variants on final heavy-VMC error, probe-final gap, ESS medians, and rollback count.
**Open questions:** Whether reduced weight spikiness translates to better final-heavy ranking consistency and whether low-omega BF remains dominated by near-coalescence gradient noise even after tempering/clipping.

### [2026-03-15] — Implemented and fast-validated four trainer stability controls

**Goal:** Roll out practical stability controls requested after the long-campaign analysis: adaptive ESS sampling, automatic rollback with LR decay, stratified hard-region replay, and BF short-range cusp-preserving regularization.
**What was done:** Extended `src/run_weak_form.py` with new trainer controls and CLI flags: adaptive ESS floor resampling (`--ess-floor-ratio`, `--ess-oversample-max`, `--ess-oversample-step`, `--ess-resample-tries`), instability rollback + LR decay (`--rollback-decay`, `--rollback-err-pct`, `--rollback-jump-sigma`), stratified geometry replay (`--replay-stratified`, `--replay-geo-bins`) using invariant bucketization of `(min_pair, r_mean)`, and BF short-range regularization (`--bf-cusp-reg`, `--bf-cusp-radius-aho`); fixed two implementation bugs discovered during smoke tests (CUDA index assert in stratified selector and replay alignment when `EL_all` length is shorter than `X`); ran a smoke check; launched and completed four short parallel GPU ablations in `outputs/2026-03-15_quick_stability4` plus a matched baseline.
**What was not done:** Did not run an extended multi-omega/multi-N matrix for these new controls yet; this pass was intentionally short-turnaround for same-target calibration.
**Issues encountered:** Stratified replay initially triggered a CUDA index assert due to direct GPU indexing path and later a length mismatch because some collocation samples are filtered before `EL_all`; both were patched in `run_weak_form.py`.
**Workarounds in place:** Stratified top-k selection now executes on CPU-safe indices and maps results back to device; replay now aligns to `n_valid=min(len(X), len(EL_all))` before bucketed selection.
**Implicit assumptions made:** Fast 80-epoch N=6, ω=1.0 BF runs with canonical BF/Jastrow init are sufficient to validate control plumbing and compare first-order stability impact before broader sweeps.
**Next action:** Run a second short batch that combines controls (ESS + rollback + stratified replay, with/without cusp regularizer) and then promote the best policy to a broader N/ω screen.
**Open questions:** Why baseline and rollback runs can still land on numerically identical final values across different seeds in short runs; whether BF cusp regularization needs a smaller coefficient or annealing schedule.

### [2026-03-14] — Built and launched a long trainer-generalization campaign

**Goal:** Move from one-off lucky runs toward a reusable trainer framework by running a long, GPU-aware campaign that compares trainer policies across multiple particle counts and trap frequencies.
**What was done:** Added trainer-side denoising controls to `src/functions/Neural_Networks.py` and `src/run_weak_form.py` (`reward_qtrim`, hard-sample replay), built `scripts/robustness_protocol.py` and confirmed that the stabilized trainer outperforms the baseline on N=6, ω=1.0, then built `scripts/long_trainer_campaign.py` to dispatch a large matrix of trainer-policy runs across N={2,6,12,20} and multiple ω values using only GPUs that are currently free enough by `nvidia-smi` memory threshold; also caught and fixed a campaign bug where low-ω BF jobs were unintentionally using the large default backflow network rather than the canonical `bf_ctnn_vcycle.pt` architecture, then relaunched the corrected 20-hour campaign.
**What was not done:** Did not yet audit the suspicious identical-across-seed robustness outputs to root cause; did not yet summarize the long campaign because it has only just been launched.
**Issues encountered:** The first long-campaign launch mixed trainer policies with the wrong BF architecture for non-resume BF jobs, which would have invalidated any trainer comparison; the bug was found immediately from log inspection and the run was aborted before being allowed to continue.
**Workarounds in place:** The campaign now explicitly passes `--init-bf results/arch_colloc/bf_ctnn_vcycle.pt` for BF transfer jobs that do not resume from an existing BF checkpoint, keeping the BF architecture fixed across trainer-policy comparisons.
**Implicit assumptions made:** Reusing the canonical N=6 BF architecture across ω and across N=12 BF transfer jobs is a better test of trainer generalization than allowing the default larger BF architecture to vary the network family implicitly.
**Next action:** When the user returns, inspect the first tranche of completed jobs from `outputs/*_long_trainer_campaign/summary.json`, especially low-ω BF and N=2/N=20 jastrow transfer results, and then decide whether to keep broad screening or narrow onto the best trainer family.
**Open questions:** Why the earlier robustness protocol produced numerically identical seed outcomes; whether N=12 BF transfer from N=6 BF architecture is actually stable enough to be meaningful; whether ESS gating should be added as a fourth high-priority policy in the long campaign.

### [2026-03-14] — Expanded the experiment report into a full weak-form retrospective

**Goal:** Turn the report from a narrow reproduction note into a broader explanation of the weak-form program: philosophy, losses, sampling, architecture families, representative outcomes, and the specific path to the `20.161314` result.
**What was done:** Audited the shared weak-form utilities in `src/functions/Neural_Networks.py`, the historical training scripts, `results/arch_colloc/summary.json`, and representative logs across Jastrow-only, BF, and Pfaffian branches; extracted final heavy exact VMC values for the main branches; rewrote `EXPERIMENT_REPORT.md` so it now explains the search from scratch, distinguishes FD residual vs hybrid REINFORCE objectives, documents the Gaussian-mixture and screened-collocation sampling story, compares representative families and energies, and then narrows down to the Stage A/B/C BF continuation chain that produced `20.161314 ± 0.002342`.
**What was not done:** Did not run any new training jobs in this pass; the work was interpretive and documentation-focused.
**Issues encountered:** Several historical runs had misleadingly good online VMC probes that were materially worse under final heavy re-evaluation, so the report had to be structured around final exact VMC numbers rather than probe minima.
**Workarounds in place:** The report now treats heavy exact VMC as the authoritative result and uses probe values only to explain checkpoint selection behavior.
**Implicit assumptions made:** The current code paths in `src/functions/Neural_Networks.py` are taken as the cleanest statement of the intended loss philosophy even when some historical scripts used earlier copies of the same ideas.
**Next action:** If the documentation needs one more layer of rigor, rerun the full Stage A/B/C chain under fresh tags and append a direct side-by-side confirmation to the report.
**Open questions:** Whether orbital-backflow experiments should get their own dedicated subsection once a stable, heavy-evaluated archival result is available for that branch.

### [2026-03-14] — Reconstructed and documented the exact 20.161 BF continuation path

**Goal:** Fix the experiment report so it explains, in a reproducible way, how the N=6, $\omega=1.0$ result `20.161314` was actually obtained.
**What was done:** Audited the historical logs [results/arch_colloc/bf_joint_reinf_v3.log](results/arch_colloc/bf_joint_reinf_v3.log), [results/arch_colloc/bf_resume_lr_v1.log](results/arch_colloc/bf_resume_lr_v1.log), [results/arch_colloc/bf_hardfocus_v1b.log](results/arch_colloc/bf_hardfocus_v1b.log), and [results/arch_colloc/jas_reinf_v2.log](results/arch_colloc/jas_reinf_v2.log); compared them against failed modern reruns in [outputs/2026-03-14_1135_overnight_auto/logs/n6_o1p0_bf_repro_s42.log](outputs/2026-03-14_1135_overnight_auto/logs/n6_o1p0_bf_repro_s42.log); rewrote [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md) around the real three-stage checkpoint lineage `bf_ctnn_vcycle -> bf_joint_reinf_v3 -> bf_resume_lr_v1 -> bf_hardfocus_v1b`, including stage-by-stage hyperparameters, explanation of why the heavy final VMC number is authoritative, and runnable current-runner commands for reproduction.
**What was not done:** Did not rerun the full historical chain from scratch as part of this documentation pass; the report now documents how to do that, but the work here was forensic/documentation only.
**Issues encountered:** The old report materially stopped at the earlier `20.173` story and mixed in experimental hypotheses that were no longer the key story for the best published N=6 BF result.
**Workarounds in place:** The report now treats the checkpoint lineage itself as part of the method and explicitly calls out the failed approximate rerun recipe that landed at `20.18958`.
**Implicit assumptions made:** Current runner semantics for `--resume` are treated as the modern equivalent of the historical “load BF+Jastrow weights and continue training” flow.
**Next action:** If needed, run the three documented commands under fresh tags and verify that the rebuilt chain lands in the same `20.16x` basin.
**Open questions:** Whether `bf_ctnn_vcycle.pt` itself should get its own retrospective report section with exact heavy re-evaluation provenance, since its role as the root of the successful chain is now central.

### [2026-03-14] — Refactor collocation core + DMC table integration + fresh production run

**Goal:** Stop faulty run, remove generated clutter, make collocation logic reusable, ensure DMC references come from central config for all supported N/omega, validate with tests/smoke checks, and relaunch a proper run.
**What was done:** Killed active overnight run; moved reusable weak-form collocation primitives into [src/functions/Neural_Networks.py](src/functions/Neural_Networks.py); switched [src/run_weak_form.py](src/run_weak_form.py) to central DMC lookup via config and shared collocation helpers; added thin runner [src/run_collocation.py](src/run_collocation.py); pointed orchestrator [scripts/auto_overnight_matrix.py](scripts/auto_overnight_matrix.py) at the thin runner; removed prior generated outputs/fallback artifacts; added tests [tests/test_dmc_lookup.py](tests/test_dmc_lookup.py) and [tests/test_run_weak_form_dmc.py](tests/test_run_weak_form_dmc.py); ran py_compile + pytest + matrix smoke; launched fresh production run [outputs/2026-03-14_1135_overnight_auto](outputs/2026-03-14_1135_overnight_auto).
**What was not done:** No broad refactor of older legacy collocation scripts in [src/run_colloc_archs.py](src/run_colloc_archs.py), [src/run_colloc_bf_jastrow.py](src/run_colloc_bf_jastrow.py), and [src/run_colloc_orbital_bf.py](src/run_colloc_orbital_bf.py) yet.
**Issues encountered:** Tool output truncation made one short validation summary hard to inspect directly; resolved by reading run artifacts directly from output directory.
**Workarounds in place:** Used direct reads of [outputs/2026-03-14_1130_overnight_auto/results.jsonl](outputs/2026-03-14_1130_overnight_auto/results.jsonl) for short-run verification.
**Implicit assumptions made:** "All omegas for N spanning 2-20" interpreted as all configured DMC table entries and supported N values present in [src/config.py](src/config.py): N={2,6,12,20}.
**Next action:** Monitor [outputs/2026-03-14_1135_overnight_auto/results.jsonl](outputs/2026-03-14_1135_overnight_auto/results.jsonl) through Phase A and Phase B completion.
**Open questions:** Whether to include N=2 and N=20 in the production matrix orchestrator target list for future runs, not only DMC coverage tests.
