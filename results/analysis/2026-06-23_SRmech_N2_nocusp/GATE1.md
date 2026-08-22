# Gate 1 — N=2/N=6 anchor, all three questions

**Date:** 2026-06-23
**Plan:** [plans/2026-06-22_dimension-program-and-roadmap.md](../../../plans/2026-06-22_dimension-program-and-roadmap.md)
**Scope:** Consolidate Phase 1 (the anchor) across Q1/Q2/Q3 before the Phase-2 ω-sweep.

---

## Q1 — CTNN vs FFNN: **gap survives fair measurement** ✅
Fair common-probe eff-rank(S) (f_net tangent space, same points for all models, measure-robust):
**CTNN 1.20–1.66 (3 seeds) vs DeepSet 2.94–3.65 (20k–164k).** Architectural (present at init,
independent of DeepSet capacity), and within ±0.04 across probe measures. Outputs:
`2026-06-22_fair_dimension_N6w1/`. → CTNN compresses the correlator into ~2× fewer collective
tangent directions. **Gate-1: PASS** (ready to scale in ω and N).

## Q2 — SR vs Adam: **the 2×2 is complete; cusp prior is a stiff-mode preconditioner; SR's edge is real-but-not-significant at ω=1** ◐
The {cusp ON/OFF} × {Adam/SR} grid at N=2 ω=1, 4 seeds (mean ± seed-std):

| | stiff-mode err (Adam) | stiff-mode err (SR) | energy err (Adam) | energy err (SR) |
|---|---|---|---|---|
| **cusp ON** (2026-06-20) | 3.4e-3 | 5.0e-3 | +0.047% | +0.056% (tie; Adam ≥ SR on stiff) |
| **cusp OFF** (2026-06-23) | 1.13e-2 [1.05,1.21] | 1.06e-2 [0.94,1.19] | −0.277% [−0.35,−0.20] | −0.216% [−0.24,−0.19] |

Findings:
1. **Removing the cusp prior inflates the stiff-mode error ~2–3×** (3–5e-3 → ~1.1e-2) for both
   optimizers — direct confirmation of D2's core claim that the fixed cusp removes the hardest
   (stiffest) direction from the *learning* problem.
2. **The optimizer ranking flips:** cusp-ON Adam ≥ SR on stiff modes; cusp-OFF SR ≤ Adam on *every*
   metric (stiff, dist-to-exact, energy) **and** SR is markedly more reproducible (tighter seed
   bands: energy std ≈0.025% vs Adam ≈0.074%). Direction matches D2.
3. **But the SR-vs-Adam gap is within seed noise** at N=2 ω=1 (bands overlap on every metric). So SR
   is *consistently but not significantly* better cusp-off here. The decisive "SR collapses the stiff
   mode while Adam stalls" is **not** demonstrated in this easy regime.
→ **Honest verdict:** the cusp prior is a function-space preconditioner doing part of SR's job; the
optimizer choice is largely immaterial *in the easy regime even cusp-off*. The decisive SR test moves
to **low ω / high var(E_L)** (Phase 2), where κ(S) and var(E_L) explode and whitening should bite.

## Q3 — Collocation vs VMC (Laplacian/zero-variance): **weak form sacrifices zero-variance** ✅
var(weak)/var(strong) = 2071× (N=2 exact), 10957× (N=6 CTNN), 403× (N=6 DeepSet); means agree within
the (large) weak-form sampling error → integration-by-parts holds, weak form is unbiased-but-
catastrophic. At the exact N=2 GS var(E_L)=1.4e-3 (≈0) vs var(e_w)=2.9. Outputs:
`2026-06-22_estimator_variance/`. The Laplacian's gift is zero-variance; dropping it is why the
winning recipe keeps the forward-only E_L reward and β small. (Caveat: var(e_w) heavy-tailed; only
the order-of-magnitude gap is robust.)

---

## Gate-1 verdict
All three questions advanced at the anchor with honest verdicts. Common thread: **the easy regime
(N=2/N=6, ω=1) under-resolves the effects** — Q1's gap is clear, but Q2's SR advantage and Q3's
estimator divergence both point to **low ω (the Wigner approach)** as the regime where the
architecture/optimizer/paradigm differences should sharpen. → Proceed to **Phase 2: ω-sweep
{1, 0.5, 0.28, 0.1, 0.01} for all three questions** (does d_eff gap grow; does SR finally beat Adam;
does var(e_w) and the collocation-vs-VMC gap track ω).
