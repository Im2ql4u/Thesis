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

### [2026-03-14] — Refactor collocation core + DMC table integration + fresh production run

**Goal:** Stop faulty run, remove generated clutter, make collocation logic reusable, ensure DMC references come from central config for all supported N/omega, validate with tests/smoke checks, and relaunch a proper run.
**What was done:** Killed active overnight run; moved reusable weak-form collocation primitives into [src/functions/Neural_Networks.py](src/functions/Neural_Networks.py); switched [src/run_weak_form.py](src/run_weak_form.py) to central DMC lookup via config and shared collocation helpers; added thin runner [src/run_collocation.py](src/run_collocation.py); pointed orchestrator [scripts/auto_overnight_matrix.py](scripts/auto_overnight_matrix.py) at the thin runner; removed prior generated outputs/fallback artifacts; added tests [tests/test_dmc_lookup.py](tests/test_dmc_lookup.py) and [tests/test_run_weak_form_dmc.py](tests/test_run_weak_form_dmc.py); ran py_compile + pytest + matrix smoke; launched fresh production run [outputs/2026-03-14_1135_overnight_auto](outputs/2026-03-14_1135_overnight_auto).
**What was not done:** No broad refactor of older legacy collocation scripts in [src/run_colloc_archs.py](src/run_colloc_archs.py), [src/run_colloc_bf_jastrow.py](src/run_colloc_bf_jastrow.py), and [src/run_colloc_orbital_bf.py](src/run_colloc_orbital_bf.py) yet.
**Issues encountered:** Tool output truncation made one short validation summary hard to inspect directly; resolved by reading run artifacts directly from output directory.
**Workarounds in place:** Used direct reads of [outputs/2026-03-14_1130_overnight_auto/results.jsonl](outputs/2026-03-14_1130_overnight_auto/results.jsonl) for short-run verification.
**Implicit assumptions made:** "All omegas for N spanning 2-20" interpreted as all configured DMC table entries and supported N values present in [src/config.py](src/config.py): N={2,6,12,20}.
**Next action:** Monitor [outputs/2026-03-14_1135_overnight_auto/results.jsonl](outputs/2026-03-14_1135_overnight_auto/results.jsonl) through Phase A and Phase B completion.
**Open questions:** Whether to include N=2 and N=20 in the production matrix orchestrator target list for future runs, not only DMC coverage tests.
