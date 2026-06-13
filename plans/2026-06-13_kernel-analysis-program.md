# Kernel-Picture Analysis Program for Neural Quantum States

**Created:** 2026-06-13
**Status:** Active vision document. Supersedes ad-hoc diagnostics for the analysis chapter.
**Scope:** Mechanistic + theoretical analysis of trained NQS wavefunctions for 2D parabolic
quantum dots, organised around the tangent-kernel picture.

---

## 0. Vision

We have an ansatz — Slater determinant × CTNN Jastrow × CTNN coordinate backflow — that reaches
near-exact ground-state (GS) energies across `N ∈ {2,6,12,20}` and `ω ∈ {1e-3 … 1.0}`. The thesis
question is no longer *"can we hit the energy"* but **"what did the network actually learn, why does
the optimiser get there, and what does that teach us about the physics."**

Three questions, one object. Define the per-sample log-derivative matrix

```
O_{k,i} = ∂ log|Ψ(x_k)| / ∂θ_i           (N_samples × N_params)
```

Everything we care about is a property of `O`:

| Angle | Object | Property of O | What it controls |
|---|---|---|---|
| 1. SR / natural gradient | `S = (1/N) OᵀO` (Fisher / QGT) and its dual `K = OOᵀ` (NTK) | **spectrum** | trainability |
| 2. CTNN vs FFNN | the column span of `O` | **rank / richness** | expressivity |
| 3. What is learned | eigenvectors of `K` over training | **rotation** | feature learning |

The unifying claim we want to make precise and demonstrate:

> Stochastic reconfiguration is *whitening of the neural tangent kernel*. It replaces the
> parametrisation-dependent Euclidean gradient with the intrinsic Fubini–Study gradient, so that the
> wavefunction moves along the projected imaginary-time flow `δΨ = −P_T (H−E)Ψ` — i.e. along the
> action of the **Hamiltonian**, not merely down the loss. Architecture (CTNN) sets *what directions
> `Ψ` can move in*; SR sets *how efficiently it moves*; the learned representation is the record of
> *which directions it chose*.

---

## 1. Guiding principles

1. **Analyse only genuine ground states.** Every wavefunction we interpret must pass the GS-quality
   gate (§4). Pretty internal structure on a mediocre wavefunction is an artefact, not physics. We
   will re-train where needed — time is not the constraint, correctness is.
2. **Primary targets are VMC-trained (SR / Adam) wavefunctions.** These sample from `|Ψ|²` and are
   the most trustworthy approximations to the true GS. Collocation / weak-form results are a
   *comparison axis*, not the primary object.
3. **Dual-track comparison wherever it is cheap.** For matched `(N, ω)` we already hold SR-trained
   (`f_netSR.pt`, `backflowSR.pt`) and collocation-trained (`f_netCTNN.pt`, `backflowCTNN.pt`) models
   in `results/official_models/`. Ask at every step: *do the two training routes reach the same
   wavefunction, or just the same energy?* (energy degeneracy vs representational degeneracy.)
4. **Kernel picture is the lens, not an afterthought.** Each diagnostic should be expressible as a
   statement about `O`, `S`, or `K`. If it cannot, ask whether it belongs.
5. **Depth before breadth.** Fully understand `N=2` before `N=6`, etc. Each size ends with a
   consolidation gate (§5) that explicitly hunts for missing pieces and low-hanging fruit before we
   scale up.
6. **Truth anchors first.** `N=2, ω=1.0` has the exact analytic energy `E = 3.0` Ha (Taut). Use
   exact / FCI references to validate *every* diagnostic at small `N`, then carry the validated tool
   to larger `N`.
7. **One run = one folder, never overwrite.** Results in `results/` or `outputs/` with dated
   subdirectories (repo rule). Re-trained "analysis-grade" GS checkpoints get a clearly versioned
   home so we never confuse them with legacy `official_models`.

---

## 2. Theoretical backbone (what we derive and write)

This program has a genuine theory component. The following should be written up cleanly (likely a
theory subsection + an appendix), independent of any single experiment.

- **T1 — SR as Fubini–Study natural gradient.** Derive `δθ = S⁻¹ f`, `f_i = Cov(E_L, O_i)`, from the
  imaginary-time projection of `(H−E)Ψ` onto the tangent space `T = span{∂_i Ψ}`. State the
  parametrisation-invariance and the equivalence to projected power iteration `e^{−τH}`.
- **T2 — The S ↔ K duality.** `S = (1/N)OᵀO` (P×P, parameter space) and `K = OOᵀ` (N×N, sample space)
  share their nonzero spectrum. Woodbury SR (`src/sr_preconditioner.py::WoodburySR`) already inverts
  `OOᵀ + λI`: our optimiser *is* the NTK. MinSR is the explicit sample-space form. Write when each is
  the right representation (`N_samples ≷ N_params`).
- **T3 — SR = NTK whitening.** Function-space flow under plain GD is `∂_t ψ = −K r` (r = residual);
  under SR it is `∂_t ψ = −P_T r`. Make precise that SR equalises the NTK spectrum on its support and
  amplifies the small-eigenvalue (high-frequency: cusp, correlation-hole) directions.
- **T4 — Spectral bias & convergence rate.** Relate per-mode convergence to the NTK/Fisher
  eigenvalues; predict the *order* in which physics is learned (mean-field → exchange → cusp →
  higher-body). This is the theoretical scaffolding for the time-resolved experiments (X3).
- **T5 — Expressivity vs body-order.** State the relationship between message-passing rounds `K` and
  attainable correlation order `(K+1)`-body at `O(N²)` cost, contrasted with explicit `k`-body
  DeepSet at `O(Nᵏ)`. This is the formal backbone for the FFNN-equivalence experiment (B3).
- **T6 — Two gradient estimators.** Collocation REINFORCE gradient `2E[(E_L−R)O]` under
  importance-sampled mixture vs VMC SR gradient under `|Ψ|²`. Same estimand in the limit; compare
  variance and, after preconditioning, the *direction* each takes through parameter space. This
  formalises "what changes between collocation and VMC, especially with SR and conditioning."

---

## 3. Cross-cutting infrastructure (build once, reuse every phase)

These are tools, not experiments. Build and unit-test them against `N=2` ground truth in Phase A,
then reuse unchanged.

- **I1 — `O` / `S` / `K` builder.** Given a checkpoint and a set of `|Ψ|²` samples, assemble the
  per-sample parameter-gradient matrix `O` (chunked over params for memory), and expose `S`, `K`,
  their spectra, effective rank, and condition number. Reuse the per-sample machinery already in
  `Stochastic_Reconfiguration.py::_score_rows` and `sr_preconditioner.py`.
- **I2 — GS-quality evaluator** (§4): heavy VMC energy + error vs reference, `var(E_L)`, ESS, Pareto
  `khat`, and zero-variance / variance-extrapolation check. One function, one verdict.
- **I3 — Imaginary-time direction probe.** For small `N`, build `(H−E)Ψ`, project onto `T`, and
  return `cos(δθ_method, δθ_imag-time)` for method ∈ {plain grad, SR}. Ground-truthable at `N=2`.
- **I4 — Body-order (HDMR/ANOVA) decomposer.** Decompose `log|Ψ|` (and the Jastrow alone) into
  1/2/3/4-body variance contributions. Reused for CTNN vs DeepSet vs Triadic.
- **I5 — Representation probe kit.** Activation hooks (node/edge embeddings, bottleneck), effective
  rank, intrinsic dimension (TwoNN/MLE), linear probes to physical targets, CKA between
  representations. Generalise the existing `diagnose_*.py` scripts into a reusable module.
- **I6 — Physics read-out kit.** Effective `u(r)` extraction, Kato-cusp coefficient, one-body density
  matrix → natural-orbital occupations, `⟨L_z⟩` and rotational-invariance check.

> Repo discipline: these live in `src/` (reusable) with thin `scripts/` drivers. Do not duplicate the
> existing diagnostic scripts — refactor them into the kit.

---

## 4. The GS-quality gate (acceptance criteria)

A checkpoint is **analysis-grade** only if it passes all of:

1. **Energy.** Heavy VMC energy within a stated tolerance of the reference (`config.DMC_ENERGIES`;
   exact `3.0` at `N=2, ω=1.0`). Target: ≤ 0.1% where a DMC/exact ref exists; record the number
   regardless.
2. **Local-energy variance.** `var(E_L)` small and, ideally, a **variance extrapolation**
   `E(σ²→0)` consistent with the reference — the cleanest GS-quality signal independent of any table.
3. **Sampling health.** ESS and Pareto-`khat` in the healthy range (`khat < 0.7`) for the evaluation
   sampler; no near-node / near-coalescence blow-ups dominating the estimate.
4. **Train-route agreement.** Where both exist, SR-trained and collocation-trained energies agree to
   within their error bars. Disagreement is itself a finding — flag, don't hide.
5. **Stability.** Re-evaluation across seeds / sampler restarts reproduces the energy.

Wavefunctions that fail get re-trained (preferably VMC/SR from a good warm start) before analysis.
The gate verdict is stored beside each analysis-grade checkpoint.

---

## 5. Phased execution

Each phase: (a) secure analysis-grade GS checkpoints, (b) run the analysis menu (§6) through the
kernel lens, (c) hold a **consolidation gate** before scaling up.

### Phase A — `N=2, ω=1.0` (the anchor)
Goal: establish and *validate against exact truth* every tool and every claim on the simplest system.
- Secure analysis-grade `N=2, ω=1.0` GS (SR-trained primary; collocation for comparison). Energy must
  land on `3.0` Ha.
- Stand up infrastructure I1–I6; unit-test against the exact solution.
- Kernel: full `S`/`K` spectrum (small enough to diagonalise exactly), condition number, effective
  rank. Verify T1–T3 numerically: `cos(δθ_SR, imag-time) ≈ 1` vs plain gradient (I3, exp A2).
- Representation: latent dimensionality, what the single pair "message" encodes (B1) — at `N=2` the
  Jastrow *is* a pure two-body object, so this is the cleanest possible decode.
- Physics read-out: does it learn the exact Kato cusp (C1)? Natural-orbital occupations vs exact (C4).
- Dual-track: SR vs collocation wavefunction — same `Ψ` or just same `E`? (CKA, density-matrix
  overlap, `u(r)` overlap).

**Consolidation gate A:** Are all six tools validated against ground truth? Does the kernel story hold
where we can check it exactly? List anything that surprised us before moving on.

### Phase B — `N=2`, full ω span (`1e-3, 1e-2, 0.1, 0.5, 1.0`)
Goal: the quantum→classical (Wigner) crossover in the *tractable* case.
- Secure analysis-grade GS at every ω (re-train low-ω as needed; these are the hard ones even at N=2).
- Kernel: how `κ(S)` and the NTK spectrum evolve with ω (A1) — first quantitative "why low ω is hard."
- NTK drift / feature-learning vs lazy across ω (A4); spectral-bias ordering (A5).
- Representation: the φ↔ψ axis already reported for `N=2` in `results.tex` (ψ-dominated at low ω,
  g-dominated at ω=1.0) — re-derive it as an NTK-eigenvector rotation.
- Physics: a network-internal Wigner order parameter (C2) tested against the exact `N=2` crossover.
- Theory: validate T4/T6 at `N=2`.

**Consolidation gate B:** Write the `N=2` story end-to-end. Is the kernel picture predictive (not just
descriptive)? Identify low-hanging fruit (e.g. ω=0.28 fills the crossover) before adding particles.

### Phase C — `N=6` (first genuine many-body; closed shell `1+5`? — confirm shell structure)
Goal: where pair → many-body correlation first matters; richest existing diagnostics live here.
- Secure analysis-grade GS across ω (SR primary). Reconcile with the existing
  `architecture_diagnostics` campaign — re-run any diagnostic whose checkpoint fails the §4 gate.
- Angle 2 in force: body-order spectroscopy (B4), CTNN vs DeepSet vs Triadic FFNN-equivalence curve
  (B3), where-it-helps map (B5), V-cycle bottleneck probe (B7), backflow-vs-Jastrow labour split (B8).
- Kernel: does SR help *more* for CTNN than for DeepSet/FFNN (X2)? Compare `κ(S)` across architectures.
- Re-examine the published `N=6` claims (100% kinetic, random=zero messages) with seed sweeps before
  they anchor a chapter.

**Consolidation gate C.**

### Phase D — `N=12`
- Analysis-grade GS across ω. Kernel methods now require chunked/stochastic estimators (lean on
  CG-SR / MinSR machinery). Confirm the `N=2`/`N=6` findings survive scale; watch for the `N=12,
  ω=0.1` effective-rank peak already seen in `results.tex`.

**Consolidation gate D.**

### Phase E — `N=20` (frontier)
- Analysis-grade GS where attainable; **failure forensics (X5)** where not: decompose failure into
  expressivity (NTK span) vs trainability (`κ(S)`) vs sampling (ESS) using the kernel tools.
- Reduced-architecture caveat (64 hidden vs 128) must be controlled for in any cross-N claim.

---

## 6. Analysis menu (run each phase; IDs carry from the 2026-06-13 brainstorm)

**Angle 1 — SR / NTK / natural gradient**
- A1 spectrum & condition number of `S`/`K` across (N, ω)
- A2 `cos(δθ_SR | δθ_plain, imag-time)` — alignment with the Hamiltonian flow (ground-truthed at N=2)
- A3 representable-residual fraction `‖P_T(H−E)Ψ‖ / ‖(H−E)Ψ‖` (convergence + expressivity-ceiling)
- A4 NTK drift / kernel alignment over training (lazy vs feature-learning)
- A5 spectral bias: error projected on NTK eigenmodes → which physics is learned first
- A6 parameter-space vs sample-space SR (Woodbury/CG vs MinSR) crossover

**Angle 2 — CTNN vs FFNN**
- B1 decode the edge "message" (rank≈1 scalar → screened `u_eff(r)`?)
- B3 FFNN-equivalence curve: energy vs param count, CTNN as horizontal line; body-order baselines
- B4 body-order spectroscopy (HDMR/ANOVA) for CTNN / DeepSet / Triadic
- B5 where-it-helps map: `ΔE_L(x)` over configuration / triplet geometry
- B7 V-cycle bottleneck probe: does it encode a global collective coordinate? ablate vs flat CTNN
- B8 backflow (nodes/sign) vs Jastrow (cusp/hole) labour split; node motion vs exact nodes at N=2
- B9 expressivity = NTK richness at matched param count (bridges to Angle 1)

**Angle 3 — what is learned**
- C1 learned vs Kato cusp coefficient
- C2 network-internal Wigner order parameter; network phase diagram vs Γ / Lindemann
- C4 natural-orbital occupations from the one-body density matrix vs HF/FCI
- C5 true intrinsic dimension (TwoNN/MLE) vs # physical collective modes
- C6 causal probing: clamp spin / messages / bottleneck → which observable degrades
- C7 what the ω-cascade transfers: invariant (universal) vs changing (regime-specific) representation
- C9 invert the Hamiltonian: reconstruct `V(x)` from `Ψ`; spatial map of ansatz error

**Cross-cutting**
- X1 N=2 exact anchor (Phase A/B)
- X2 SR helps CTNN more than FFNN (low-rank latent ⇒ well-conditioned `S`)
- X3 time-resolved: log rank / φ↔ψ / NTK spectrum / body-order / force-alignment every K epochs
- X5 N=20 failure forensics (expressivity vs trainability vs sampling)

---

## 7. Dual-track comparison (recurring theme)

At each `(N, ω)` where both exist, compare **VMC/SR-trained vs collocation-trained**, and within VMC,
**SR vs Adam**:
- Energy & GS-quality (§4) — do they agree?
- Representation — CKA, latent dimensionality, `u(r)`, natural orbitals: same wavefunction or just
  same energy?
- Kernel — does SR-trained sit in a better-conditioned region of parameter space? Does collocation's
  importance-sampled gradient (T6) leave a different fingerprint on `S`?
- Conditioning narrative — quantify the cascade-vs-single-run difference via `κ(S)` trajectories.

This is where "what changes between PINN/collocation and VMC, especially with SR and conditioning"
gets answered concretely.

---

## 8. Mapping to the thesis

- **Theory chapter:** T1–T6 (SR as Fubini–Study NG; S↔K duality; NTK whitening; spectral bias;
  expressivity↔body-order; two estimators).
- **"What the networks learn" (§ results):** C-series + X3 time-resolved narrative; recast the existing
  φ↔ψ / backflow-rank / coupling tables as kernel statements.
- **Collocation chapter:** A-series + dual-track (§7) as the mechanistic backing for the SR-vs-Adam
  and collocation-vs-VMC claims.
- **Architecture / methods:** B-series (message decode, FFNN-equivalence, V-cycle, body-order).
- **Discussion / future work:** X5 failure forensics; network-internal order parameter (C2).

---

## 9. Open theoretical questions & risks

- Does "SR aligns with H-flow" survive finite sampling and damping, or is it only exact in the
  `λ→0`, infinite-sample limit? (Quantify the degradation.)
- Is the low latent rank a property of the *physics* or of the *architecture's inductive bias*?
  (X2 + cross-architecture comparison should separate these.)
- Energy degeneracy vs representational degeneracy: if SR and collocation reach different internal
  representations at the same energy, *which is "more physical"*, and by what criterion?
- Reduced architecture at `N=20` confounds any clean cross-N scaling claim — control explicitly.
- Several published single-seed results (100% kinetic, random=zero messages) need seed sweeps before
  they carry a chapter.

---

## Appendix: current assets

- **Checkpoints:** `results/official_models/{2p,6p,12p}/w_*/` hold matched SR (`*SR.pt`) and
  collocation (`*CTNN.pt`) variants. Treat as *candidates* — each must pass §4 before analysis.
- **Architectures:** `src/jastrow_architectures.py` — `CTNNJastrowVCycle` (primary), `CTNNJastrow`
  (flat), `DeepSetJastrow` (pairwise), `TriadicDeepSetJastrow` (3-body), `CTNNShellAwareJastrow`.
- **SR:** `src/sr_preconditioner.py` (Woodbury/CG/MinSR); `src/functions/Stochastic_Reconfiguration.py`
  (`train_model_sr_energy`, `sr_step_energy_mb`, `_score_rows`).
- **Collocation:** `src/run_weak_form.py`, `src/run_colloc_*.py`; Fisher preconditioner in
  `src/fisher_preconditioner.py`.
- **References:** `src/config.py::DMC_ENERGIES` (N=2, ω=1.0 = 3.0 exact/Taut).
- **Prior diagnostics:** `scripts/diagnose_{input_attribution,ctnn_pairwise,deeper}.py`,
  `results/figures/architecture_diagnostics/DIAGNOSTIC_SUMMARY.md`.

**Immediate next step:** Phase A — secure an analysis-grade `N=2, ω=1.0` GS and stand up I1–I3, then
run A1/A2/A3 and check the kernel story against the exact solution.
