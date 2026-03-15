# Session Log

Rolling log of working sessions. Most recent entry at the top. Each entry is a snapshot of what happened, what changed, and what is pending.

The model reads this at session start. Write it as if you are handing off to yourself after a week away.

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
