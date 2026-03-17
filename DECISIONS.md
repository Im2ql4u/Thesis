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
