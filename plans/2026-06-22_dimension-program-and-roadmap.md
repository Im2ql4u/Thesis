# Three Questions, One Lens — NQS Analysis Roadmap

**Created:** 2026-06-22 (rebalanced same day: the first draft over-indexed on effective dimension;
this version treats the three questions as co-equal, with the kernel/dimension picture as the shared
lens, not the spine).
**Status:** Active spine document. Subsumes the open items of
[2026-06-13_kernel-analysis-program.md](2026-06-13_kernel-analysis-program.md) and
[2026-06-21_collocation-conditioning.md](2026-06-21_collocation-conditioning.md).
**Companion:** [STATUS_REPORT_2026-06-22.md](../STATUS_REPORT_2026-06-22.md)

---

## 0. The three questions (co-equal) and the one lens

The thesis answers **three first-class questions**, each pursued to full mechanistic depth:

- **Q1 — Architecture: CTNN vs FFNN.** Why and when does message-passing beat a pairwise/separable
  (DeepSet) correlator — in energy, in wavefunction quality, in what it represents?
- **Q2 — Optimizer: SR vs Adam.** What does natural gradient actually *do* to the update, and *where*
  does it earn its keep over Adam (and where is it immaterial)?
- **Q3 — Paradigm: Collocation vs VMC.** What changes between weak/strong-form, importance-sampled
  collocation and |Ψ|²-sampled VMC — in conditioning, variance, bias, and the learned state — and
  which wins where?

**The shared lens (toolkit, not goal).** Every question is asked and answered through the
tangent-kernel picture: the per-sample score `O_{k,i}=∂ log|Ψ(x_k)|/∂θ_i`, the QGT `S=OᵀO/B`, the NTK
`K=OOᵀ`, their spectrum and **effective dimension `d_eff`**, plus `var(E_L)`, ESS, and the
zero-variance principle. Effective dimension is **one cross-cutting diagnostic** that happens to show
up in all three (it is the example that motivated this program's depth) — it is not the spine.

**The synthesis we expect to reach (destination, not organizing axis).** Pursued honestly, the three
converge on one object — the **low-dimensional, cusp/kinetic-dominated tangent space**: CTNN reaches a
lower-dimensional, better-conditioned, lower-variance representation (Q1); SR whitens that space and
matters exactly where the ansatz/measure leave it ill-conditioned (Q2); the sampling measure and the
Laplacian set whether that space is resolvable with low variance (Q3). We let the three questions lead
us there; we do not assume it.

---

## 1. Shared infrastructure, lens, and discipline (build/standardise once; serve all three)

- **Kernel kit (built):** `build_O`, `kernel_spectrum` (d_eff = participation ratio, κ, supported
  rank), `sr_vs_plain_alignment`, `gradient_snr`, `gs_quality`, `zero_variance_extrapolation`,
  `residual_jacobian` (strong/weak), `bodyorder.py`, `collectivity.py`, `representation.py`,
  trajectory checkpointing (`train.py::ckpt_every`).
- **The fair-measurement protocol (applies to ALL three, not just dimensions):**
  1. **Common probe set** — evaluate every comparison on the *same* configurations from a fixed
     reference measure (|Ψ_exact|² at N=2; pooled / best-checkpoint |Ψ|² at N≥6), not each model's own
     density. Report own-measure too and check agreement.
  2. **Convergence** — report the metric vs `n_samples`; accept only the plateau; scale samples with N.
     Never report sample-capped `numerical_rank` as a dimension.
  3. **Identical estimator** — same formula / `rel_tol` / centering across arms; report sensitivity.
  4. **Two axes, both reported** — matched-params *and* matched-accuracy (they answer different
     questions); for Q2/Q3 also matched-compute / matched-checkpoint where relevant.
  5. **Seeds + GS-gate** — ≥3 seeds; analyse only checkpoints passing the §4 gate (energy, var(E_L),
     ESS/khat, seed stability).
- **Exact anchor:** N=2 (Taut E=3.0 at ω=1; finite-volume reference solver at any ω) validates every
  claim before it scales.
- **Depth before breadth:** finish a system size (all three questions) before scaling N; ω=1 before
  the Wigner sweep. One run = one dated folder; save every spectrum `.npz`.

---

## 2. The three tracks (equal depth)

### Q1 — Architecture: CTNN vs FFNN

- **Sharp question:** message-passing vs separable pairwise — energy, quality (var(E_L)),
  representation, and the body-order/dimension it can carry at fixed cost.
- **Status:** var(E_L) discriminator (2–7×, grows toward Wigner) — single-seed; d_eff ≈1.5 (CTNN) vs
  3.4–4.8 (DeepSet) at ω=1 — unconsolidated/2-seed; message decodes local physics, collapses at
  Wigner; lazy (CTNN CKA 0.93) vs rich (DeepSet 0.54). Energy ties at ω≥0.1 (expected).
- **Deep sub-questions:** (a) **body-order/ANOVA** — does CTNN carry genuine 3-/4-body the pairwise
  net cannot (`bodyorder.py`)? (b) **message identity** — is the message a local density/field;
  **where-it-helps** ΔE_L(x) map; (c) **effective dimension** (the worked example) under the fair
  protocol; (d) **lazy vs rich** — does CTNN discover features (NTK rotation) DeepSet can't; (e)
  **FFNN-equivalence curve** — params to match CTNN in energy *and* var; (f) **over-smoothing /
  non-separability** — Dirichlet energy across V-cycle stages, cross-particle sensitivity
  (`collectivity.py`).
- **Scaling:** var ratio, d_eff, body-order vs **ω (→Wigner)** and **N**. Predict the advantage grows
  toward Wigner and with N.
- **Map to physics:** body-order ↔ true correlation order; message ↔ local density functional;
  non-separability ↔ the Wigner collective/consensus state.
- **Decisive figure:** {var(E_L), d_eff, body-order} for CTNN vs FFNN over (N, ω), fair + seeded.

### Q2 — Optimizer: SR vs Adam

- **Sharp question:** what natural gradient does to the update direction, and the regime map of where
  it beats Adam vs ties it.
- **Status:** SR = NTK whitening, exact at N=2 (cos_sr 0.997 vs cos_plain 0.04); plain gradient
  misaligned at N=6 too (cos_plain 0.07–0.16); **Jun-20 NULL** — but only the cusp-ON quadrant ran,
  the quadrant where SR is *predicted* not to help. `gradient_snr` exists.
- **Deep sub-questions:** (a) **alignment with imaginary-time/H-flow** vs ω, N (done at N=2 ω=1); (b)
  **the cusp-OFF 2×2** {cusp ON/OFF}×{Adam/SR} — the unrun decisive test: does SR collapse the stiff
  cusp-mode error while Adam stalls, and does a stiff NTK mode appear/disappear with the cusp? (c)
  **order of learning / spectral bias** — does SR change *which physics is learned first* and collapse
  the two-timescale signature (needs trajectories); (d) **the SR-advantage regime map** vs (cusp,
  ω, N, var(E_L)) — when does whitening matter; (e) **does SR help CTNN more than FFNN** (X2 — bridges
  Q1: lower-rank S ⇒ more to gain or less?); (f) **why Adam floors** — gradient-SNR / var(E_L)
  conflation mechanism.
- **Scaling:** SR-advantage vs **ω (→Wigner)** and **N** — predict it grows where var(E_L)/κ grow.
- **Map to physics:** SR places the update on the cusp/correlation-hole (high-frequency physics); the
  "where it matters" boundary = where the ansatz (cusp prior, backflow) and measure (|Ψ|²) leave
  ill-conditioning behind.
- **Decisive figure:** the SR-advantage map over (cusp on/off, ω, N) with the ‖δ‖-on-stiff-modes
  decomposition; cos(SR/plain, H-flow) vs ω.

### Q3 — Paradigm: Collocation vs VMC

- **Sharp question:** weak/strong-form importance-sampled collocation vs |Ψ|²-VMC — conditioning,
  variance, bias, learned state, and the energy reckoning.
- **Status:** ESS collapse 3.0%→0.11% across ω; the sampling **measure** is the conditioning lever;
  |Ψ|² is itself the preconditioner; matched Gaussian beats the "smart" mixture; the
  integration-by-parts (Laplacian) identity. **VMC+SR (−0.028%) may have overtaken collocation
  (+0.01%) — unchecked.** `run_weak_form.py` header still claims weak-form "eliminates the
  conditioning catastrophe" (contradicted).
- **Deep sub-questions:** (a) **the two estimators (T6)** — same estimand, different variance *and*
  direction through parameter space; (b) **the Laplacian anatomy** — var(strong)→0 vs var(weak) finite
  (zero-variance cost), why the forward-only/detached Laplacian trains (Hermiticity), detach-bias vs
  ESS; (c) **dual-track §7** — same ansatz, both routes: same Ψ or just same E (CKA, 1-RDM, d_eff,
  u(r)); (d) **is collocation still competitive** (honest energy reckoning vs VMC+SR); (e) the
  **measure/ESS conditioning** story consolidated honestly (correct the doc claim); (f) **what the
  ω-cascade transfers** (invariant vs regime-specific representation, C7).
- **Scaling:** collocation-vs-VMC gap (energy, var(E_L), ESS, bias-vs-exact, d_eff) vs **ω** and **N**;
  where each paradigm wins.
- **Map to physics:** |Ψ|² = the natural preconditioner; the Laplacian = zero-variance; collocation's
  coverage vs |Ψ|²'s conditioning = the exploration/exploitation + zero-variance trade-off.
- **Decisive figure:** collocation vs VMC on {energy error, var(E_L), bias-vs-exact, d_eff} vs ω, plus
  var(strong) vs var(weak) vs ω at exact N=2.

### Cross-links (so the three reinforce rather than fragment)

- **X2** (Q1×Q2): does SR help CTNN more than FFNN — low-rank S.
- **d_eff / var(E_L)** appear in all three (shared diagnostics).
- **Kinetic/zero-variance** thread links Q1 (var discriminator) ↔ Q3 (Laplacian).
- **Synthesis:** the low-dimensional cusp-dominated tangent space (the destination of §0).

---

## 3. Phased execution — phase by system size, run all three questions at each

Each phase ends with a **consolidation gate covering all three questions** (and an explicit
low-hanging-fruit / surprise list) before scaling N.

### Phase 0 — Consolidate what exists (days; no training) ★ start here
- Collate `results/analysis/2026-06-15_{2x2,eq}_*` + `AB_modes_alignment`: the Q1 table (eff-rank,
  var, equivalence ladder) and any Q2 content (the 2x2 is arch×optimizer), into the journal.
- Recover `data_alignment_trajectory.csv` → first lazy-vs-rich (Q1/Q2) drift curves, free.
- **Gate 0:** what is already answered for each of Q1/Q2/Q3, written with caveats explicit.

### Phase 1 — N=2 ω=1 anchor, all three questions (days; exact-anchored)
- **Q1:** fair d_eff + var(E_L) CTNN vs DeepSet (common probe set, seeds); body-order at N=2 (sanity:
  ~pure 2-body). **Q2:** the **cusp-OFF 2×2** + stiff-mode/NTK decomposition (turn the null into the
  real test). **Q3:** var(strong) vs var(weak) at the exact GS; dual-track VMC-vs-collocation overlap.
- **Gate 1:** does each question's N=2 claim hold against exact truth? Is the d_eff gap real under the
  fair protocol?

### Phase 2 — N=2 + N=6 across ω (→ Wigner), all three (≈week)
- Re-train low-ω to analysis-grade first. Track per question vs ω: Q1 {var ratio, d_eff, body-order,
  message decode}; Q2 {SR-advantage, cos(SR/plain), order-of-learning via trajectories}; Q3
  {collocation-vs-VMC gap, ESS, var(strong)/var(weak)}.
- **Gate 2:** write the ω-story end-to-end for each question; are the lens metrics predictive?

### Phase 3 — Scale to N=12, N=20, all three (≈weeks)
- Scaling laws: Q1 d_eff/var growth (control the N=20 reduced-arch confound); Q2 SR-advantage vs N;
  Q3 collocation-vs-VMC vs N. **X5 failure forensics** at N=20: expressivity (NTK span) vs
  trainability (κ) vs sampling (ESS).
- **Gate 3:** do the N=2/6 findings survive scale for each question?

### Phase 4 — Map to physics + synthesis (≈weeks)
- Name the dimensions/modes (response projection ∂_ωΨ\*/∂_λΨ\*; excitation overlap; network-internal
  Wigner order parameter C2). Fold Q1/Q2/Q3 into the unified low-dim-tangent-space synthesis and the
  thesis chapter mapping (theory / what-is-learned / architecture / collocation).

### Cross-cutting (run against the relevant phase, not serialized)
- Time-resolved trajectories (Q2 order-of-learning, Q1 lazy-vs-rich) — re-train with `ckpt_every`.
- Body-order ANOVA (Q1) — `bodyorder.py`, at N=6/N=12.
- Correct the `run_weak_form.py` conditioning claim once Q3 Phase-1/2 settles it.

---

## 4. Pre-registered outcomes (per question)

- **Q1:** CTNN advantage = lower d_eff + lower var(E_L) + higher body-order, growing toward Wigner/large
  N (spine holds); or the d_eff gap collapses under the fair protocol (artifact) → fall back to var/
  body-order; or advantage is energy-neutral everywhere (expressivity story only).
- **Q2:** SR collapses the stiff cusp-mode error cusp-OFF and at low-ω/large-N (earns its keep where
  predicted); or SR ≈ Adam even cusp-OFF → "with a physics-informed ansatz the optimizer is largely
  immaterial" (honest, publishable null).
- **Q3:** |Ψ|²/VMC dominates on conditioning+variance while collocation wins only on coverage/low-ω
  robustness, and VMC+SR has overtaken collocation on energy → reframe collocation as the bridge; or
  collocation remains competitive at low-ω/large-N → it stays a contribution. Either way the
  measure/Laplacian mechanism is the result.

## 5. Risks & discipline

- Sample-limited d_eff at large N (enforce convergence); common-probe choice at N≥6 (report
  sensitivity); N=20 reduced-arch confound (control). **Anti-thrash rule:** if a phase returns a null,
  finish its consolidation gate before changing the spine — do not spin up a new plan. Journal +
  decisions after each gate; reuse existing tools; the only new code is the common-probe wrapper and
  the Phase-4 response-projection.
