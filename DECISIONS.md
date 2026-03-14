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

### [2026-03-14] — Centralize DMC lookup and weak-form collocation primitives

**Decision:** Use [src/config.py](src/config.py) as the single source of truth for DMC references via shared helper `lookup_dmc_energy` in [src/functions/Neural_Networks.py](src/functions/Neural_Networks.py), and host reusable weak-form collocation primitives in the same shared module instead of keeping them runner-local.
**Alternatives considered:** Keep hardcoded fallback DMC table in [src/run_weak_form.py](src/run_weak_form.py); keep weak-form helper implementations duplicated in runner files.
**Reasoning:** Centralized references avoid silent NaN behavior for supported N/omega combinations; shared collocation helpers reduce duplication and make runner files thinner and easier to maintain.
**Constraints introduced:** Future DMC updates must be made in [src/config.py](src/config.py); collocation helper changes now affect all runners importing from [src/functions/Neural_Networks.py](src/functions/Neural_Networks.py).
**Confidence:** high
