# Phase 0 — Consolidation & Gate 0

**Date:** 2026-06-22
**Plan:** [plans/2026-06-22_dimension-program-and-roadmap.md](../../../plans/2026-06-22_dimension-program-and-roadmap.md)
**Scope:** Collate the un-journalled `2026-06-15_{2x2,eq}_N6w1_*` + `N2_w1_ctnn_acc` runs (no training)
into a per-question picture, and recover the alignment-trajectory data.
**Data:** `consolidation_table.csv` (this dir); sources in `results/analysis/2026-06-15_*`.

---

## What these runs are

All are **VMC-trained** (Adam or SR), cusp-on, N=6 ω=1 (+ one N=2 ω=1 anchor), with full kernel
diagnostics (`build_O`/`kernel_spectrum`) and a per-step alignment trajectory. They form:
- a **2×2 arch×optimizer** grid (Q1×Q2 / X2): {CTNN, DeepSet} × {Adam, SR};
- a **DeepSet equivalence ladder** (Q1/B3): sizes s(20k), m/big(48k), match(89k), xl(164k);
- **CTNN seeds** (seed1, seed2) + the 2×2 CTNN runs ⇒ 4 CTNN replicates.

## Consolidated numbers (N=6, ω=1, own-|Ψ|² sampling, 768 pts)

| family | runs | eff-rank(S) | var(E_L) | κ(S) | error% |
|---|---|---|---|---|---|
| **CTNN** (80k) | 4 (2 seeds × adam/sr) | **1.49** [1.39–1.60] | **0.026** [0.023–0.029] | 5e7–1.2e8 | −0.01…+0.04 |
| **DeepSet** (20k–164k) | 7 | **3.78** [3.24–4.76] | 0.030–0.097 | 1.4e7–3e10 | −0.05…+0.10 |
| CTNN anchor (N=2, 9.8k) | 1 | 1.12 | ~0 (exact) | 9.8e11 | −0.03 |

## Per-question status after Phase 0

### Q1 — CTNN vs FFNN: **strongly supported (pending fair re-measure)**
- **CTNN's tangent space is ~1.5-dimensional; DeepSet's is 3.2–4.8.** The gap is robust across seeds
  and optimizer, and **does not shrink as DeepSet grows** (ds_s 3.56 → ds_xl 4.76) — it is an
  **architectural/inductive-bias** property, not capacity. Confirmed by the trajectories: the gap is
  present at initialization (CTNN ≈1.1, DeepSet ≈3.9 at step 25).
- var(E_L) is lower for CTNN (~0.026); the best-converged DeepSet (xl, 2× params) reaches 0.033 but
  does not match. **Caveat:** DeepSet var spans 0.030–0.097 ⇒ uneven convergence; the var claim needs
  a matched-accuracy control (Phase 1).

### Q2 — SR vs Adam: **null at ω=1 extends to N=6; the decisive test is still unrun**
- In the 2×2, **SR ≈ Adam for both architectures** (CTNN err 0.029% both; DeepSet 0.071 vs 0.078; var
  tied). This extends the Jun-20 N=2 cusp-on null to N=6 ω=1 — the easy regime where the cusp prior +
  |Ψ|² already pre-condition.
- **But a real optimizer-independent dynamic appears:** κ(S) *decreases* over training for CTNN
  (2e9→8e7) and *increases* for DeepSet (1.4e9→2e10). cos_plain stays small (~0.06–0.17) throughout —
  the plain gradient is misaligned, yet Adam (diagonal preconditioning) reaches SR's endpoint here.
  So the *descriptive* claim (SR = NTK whitening, plain grad misaligned) holds; the *prescriptive*
  claim (SR beats Adam) does **not** at ω=1. The where-SR-wins test (**cusp-OFF, low-ω**) is the open
  decisive experiment (Phase 1/2).
- **X2 (does SR help CTNN more):** inconclusive at ω=1 (SR vs Adam var: CTNN 0.026 vs 0.029, DeepSet
  0.097 vs 0.093 — within noise). Needs low-ω.

### Q3 — Collocation vs VMC: **not advanced in Phase 0**
- These runs are all VMC. Q3 stands at the Jun-21 state (ESS collapse 3.0%→0.11%; measure is the
  conditioning lever). The N=6 `*_bf_casc` (collocation-trained) checkpoints exist for a dual-track
  comparison but that is Phase 1+.

### D4 — lazy vs rich (recovered, was thought missing)
- The low dimensionality is **baked in at initialization**, not discovered: CTNN stays ~1.1→1.6,
  DeepSet ~3.9→3.4 across training. Both are "lazy" in *dimension*; the quantity that actually evolves
  is **conditioning** (κ), oppositely for the two architectures.

## Caveats (carried into Phase 1 as the things to fix)
1. eff-rank measured on each net's **own |Ψ|²**, 768 samples ⇒ apply the common-probe-set + sample-
   convergence protocol before the number is load-bearing.
2. DeepSet runs are **not matched-accuracy**; var(E_L) comparison needs that control.
3. eff-rank trajectory mixes `num_rank` 383/767 (sample-count change mid-run) — within-run trends are
   indicative only.
4. zero-variance extrapolation is unreliable for DeepSet (high var, few points) — ignore `error_pct_zv`
   there.
5. N=6 ω=1 only; CTNN has 4 replicates, DeepSet sizes have 1 each (no per-size seeds).

## Gate 0 verdict
- **Q1 has a real, seeded, architectural signal** (eff-rank 1.5 vs 3.8) ready to be hardened.
- **Q2's ω=1 null is consolidated**; the cusp-OFF / low-ω test is correctly identified as the next
  decisive move.
- **Q3 is untouched** and must not be silently dropped — it is a first-class Phase-1/2 target.
- → Proceed to **Phase 1 (N=2 ω=1 anchor, all three questions)**: fair-protocol d_eff (Q1),
  cusp-OFF 2×2 (Q2), var(strong)-vs-var(weak) + dual-track (Q3).
