# Collocation-Conditioning Program

**Created:** 2026-06-21
**Status:** active
**Predecessor:** [2026-06-13_kernel-analysis-program.md](2026-06-13_kernel-analysis-program.md) (VMC/SR kernel picture)

---

## Why this program exists

The decisive VMC experiment ([JOURNAL 2026-06-20]) returned a **null**: SR ≈ Adam at
N=2, cusp-on. The reason is structural, not a measurement failure — **VMC with a cusp prior
is already well-conditioned.** The VMC loss `E[E_L]` under `|Ψ|²` plus a fixed cusp removes the
one stiff direction, so SR has nothing left to whiten. We have been measuring **κ(S)** (the
Fisher / QGT), but the thesis's central theory — `κ(A) ~ (k_max/k_min)⁴`, De Ryck conditioning,
"conditioning not depth" — is about the **strong-form least-squares residual operator**
`A = J*J`, `J = ∂_θ[(H−E)Ψ]`. **That object only exists in the collocation picture.**

`run_weak_form.py:8` already asserts weak-form "eliminates the conditioning catastrophe of the
strong-form approach" — a pivot made by intuition and **never measured**. This program measures
it, and moves the optimizer question to the picture where it has teeth (and should resolve
*positive*, unlike the VMC null).

**Central thesis claim to validate or break:** strong-form `κ(A) ~ k⁴`, weak-form `κ(A) ~ k²`,
and the regime where natural-gradient earns its keep is exactly where that conditioning is bad
*and* the Fisher estimate is stable enough to invert.

---

## What already exists (reuse, do not rebuild)

- `src/run_weak_form.py` (weak-form / Rayleigh trainer; ESS controllers, importance mixture,
  shellflow proposals, diagonal natural-gradient preconditioner).
- `src/run_colloc_archs.py`, `run_colloc_bf_jastrow.py`, `run_colloc_orbital_bf.py` (strong-form
  collocation variants).
- `src/analysis/diagnostics.py`: `build_O`, `kernel_spectrum`, `local_energy` (exact Laplacian),
  `gs_quality`. **Needs one addition:** `residual_jacobian` / `conditioning_A` (Phase 0).
- `src/analysis/system.py::load_system` (rebuild any checkpoint).
- Converged checkpoints: N=2 ω=1 (exact-validated), N=6 ω∈{1,0.5,0.1,0.01} cascade.
- Reference energies: `COLLOCATION_BEST_RESULTS.md`, DMC table in config.

Prior collocation results to explain: N=6 reaches **+0.002%**; **N=20 ω=1 = +33%, N=20 ω=0.1 =
+5.5%** (broken). SR wins high-ω, **Adam+ESS wins low-ω** (the tension to resolve).

---

## Phases — quickest first

### Phase 0 — Measure κ(A): strong vs weak (LOAD-TIME, minutes) ★ start here

κ(A) is a property of the ansatz at a point in parameter space, *independent of how it was
trained* — so we measure it on the **existing** good checkpoints. No training.

**Build (small):** `diagnostics.conditioning_A(system, x, form)` —
- strong form: `J[k,i] = ∂_θi R_s(x_k)`, `R_s = (HΨ−EΨ)/Ψ = E_L(x) − E` (uses the exact
  Laplacian already in `local_energy`); `A = JᵀJ / B`.
- weak form: `J[k,i] = ∂_θi r_w(x_k)`, `r_w` = first-derivative (Rayleigh) residual from
  `run_weak_form.weak_form_local_energy` (no Laplacian).
- return spectrum, κ, eff-rank — same shape as `kernel_spectrum`.

**Runs (one GPU, minutes each):**
1. N=2 ω=1 and N=6 ω=1: κ(A_strong) vs κ(A_weak) vs κ(S). **Test: `k⁴` vs `k²`?**
2. ω-sweep on the N=6 cascade {1, 0.5, 0.1, 0.01}: κ(A) vs ω (predict: blows up as ω↓).
3. Cusp on/off: build A with the analytic cusp subtracted vs not — **does a high-k mode
   appear at coalescence when cusp is off?** (the Kato-spike prediction).

**Decisive output:** a single figure, κ vs k (mode index) for strong/weak/Fisher, plus the
power-law fit. Validates or breaks the thesis's core theory. **Everything below hangs on this.**

### Phase 1 — SR-vs-Adam-vs-ω in collocation (fast training, hours; the positive counterpart)

Reuse `run_weak_form.py`. N=6, ω ∈ {1.0, 0.1}, optimizer ∈ {Adam+ESS, natural-gradient}, 3 seeds.
Log along training: ESS, Fisher-noise proxy, κ(A) at checkpoints, final error.
**Mechanism test:** natural-gradient wins where ESS high / Fisher stable (ω=1); Adam+ESS wins
where ESS low (ω=0.1). Resolves the high-ω/low-ω tension as a *conditioning × estimator-noise*
trade-off — the live, positive version of the VMC null.

**GPU map:** 4 independent runs = {ω}×{opt}, one per GPU, ×3 seeds staged. Log to
`/tmp/collo/*.log`, grep markers.

### Phase 2 — The N-scaling wall: conditioning or sampling? (training, ~day)

N ∈ {6, 12, 20}, ω=1, collocation. At fixed wall-clock, measure κ(A) and ESS vs N.
**Question:** does κ(A) (→ natural-gradient is the fix) or ESS collapse (→ point selection is
the fix) predict the +33% N=20 failure? Then test the implied fix on N=20.
**GPU map:** one N per GPU; N=20 gets the most wall-clock.

### Phase 3 — Dual-track representation + synthesis

Same ansatz, VMC/SR vs collocation: CKA, 1-RDM overlap, NTK alignment — same Ψ or just same E?
Fold Phases 0–2 into the "conditioning not depth" chapter: the theory is a *collocation*
statement; VMC hides it; weak-form `k²` vs strong-form `k⁴` is the measured backbone.

---

## Pre-registered outcomes

- **(theory holds)** strong `k⁴` / weak `k²` confirmed; natural-gradient wins iff κ(A) bad AND
  Fisher stable; N-wall is a conditioning wall → natural-gradient scales it. → clean thesis spine.
- **(theory partial)** scaling exponents differ from 4/2 but strong ≫ weak, and the
  ESS-vs-conditioning trade-off explains the optimizer tension. → still a real, honest result.
- **(null)** κ(A_strong) ≈ κ(A_weak) and optimizer choice immaterial in collocation too → the
  conditioning narrative is wrong and we say so; pivot to pure expressivity/scaling.

## Discipline

One run = one dated folder under `results/`. Save all spectra (`.npz`) behind every figure.
Reuse existing trainers/diagnostics; the only new code is the Phase-0 `conditioning_A` probe.
Journal + decisions after each phase.
