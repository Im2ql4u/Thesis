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

## 2.5 The mechanistic depth layer (the questions we actually care about)

**Self-critique (2026-06-14).** Phase A established the tools and confirmed, against exact truth,
that the ansatz is the ground state and that SR whitens the NTK (κ(S)≈10¹², cos(plain)→0.04). But
global scalars of `O` (κ, eff-rank, a cosine) *describe* the geometry; they do not *explain the
mechanism* or *open the network*. Three honest gaps:

1. We characterised the **output** (J(r)=log(1+r)), never the **internal mapping**.
2. **CTNN-vs-FFNN is structurally unanswerable at N=2** (one pair, zero many-body). That question
   only has signal at N≥6.
3. We showed *that* the plain gradient is misaligned, not *what function* it wastes itself on, *where
   in space*, or *why*.

This section defines the deeper questions and the minimal system each needs. The cusp is a **fixed
analytic prior** (`u_cusp=Σγr e^{-r/ℓ}`, not learned), so the network learns the *smooth residual*
`J − u_cusp`; this reframes both "what the cusp does" and "what the network learns".

### Deep questions

- **D1 — What "gradient ⊥ Hamiltonian flow" means, and where it matters.** Plain step ∝ `K·r`, SR
  step ∝ `P_T·r`, target = `r=E_L−E`. We measured cos(plain,r)=0.04 *with* rep_fraction(SR)=0.997:
  the residual is spread democratically over ~150 stiff NTK modes while `K` (anisotropic over 12
  decades) collapses the plain step onto the single softest mode, so
  `cos(plain,r) ≈ 1/√(#modes the physics occupies)`. Investigate by (i) mapping NTK eigenvectors to
  **real-space functions of r** (soft top mode = global Jastrow amplitude? stiff modes =
  cusp/short-range + far tail?); (ii) plotting **δψ_SR(x) vs δψ_plain(x) in real space** — *where*
  the natural-gradient update lands (predict: coalescence cusp + density tail); (iii) treating
  **cos(plain) at convergence as a correlation-non-triviality meter** that should shrink as Γ grows
  (low ω, larger N).

- **D2 — The cusp prior as a function-space preconditioner; cusp×SR 2×2.** The fixed cusp removes the
  hardest (lowest-NTK-eigenvalue) direction from the *learning* problem — the same stiff direction SR
  fixes in parameter space. Centrepiece experiment: {cusp ON/OFF} × {Adam/SR}. Predict cusp-OFF+Adam
  worst (E_L spikes at coalescence, extra stiff mode in the NTK, energy plateau); cusp-ON+Adam works
  *because the prior did SR's job*; SR rescues cusp-OFF. Measure the NTK spectrum ON vs OFF (a stiff
  mode should appear/disappear). Also decompose `J = u_cusp(fixed) + f_net(learned)` and study the
  learned correction vs ω (ℓ=1/√ω stretches the prior).

- **D3 — What the network learns internally.** (i) **Effective variational coordinate**: eff-rank(S)
  ≈1 ⇒ a ~10⁴-parameter net uses ~1 functional DOF — identify it (real-space shape of the top NTK
  eigenfunction; perturb the wavefunction along it and watch the density). A 1-D effective theory
  embedded in parameter space. (ii) **Mechanistic circuit decoding**: hook edge/node/message/
  bottleneck/readout; plot activations vs r (N=2) → which units are independent, is there a
  cusp-region unit vs a tail unit.

- **D4 — Lazy vs rich (feature learning).** Is the net a kernel machine in a *fixed* feature space
  (lazy: NTK constant → learns only coefficients) or does it *discover* features (rich: NTK rotates)?
  Measure kernel alignment `K(t)·K(t')` and top-eigenvector rotation over training. This is the
  rigorous form of "what are the learned hidden mappings": lazy ⇒ mappings fixed at init; rich ⇒
  discovered, and they *are* the answer. Hypothesis: smooth N=2/ω=1 target ⇒ fairly lazy; low ω /
  many-body ⇒ rich; **CTNN richer than FFNN may be *why* it wins**. One measurement, all three angles.

- **D5 — Training dynamics.** Order of learning via the error projected on NTK modes (predict:
  smooth/mean-field first, cusp/tail last); the **two-timescale** signature of ill-conditioning and
  whether SR collapses it; NTK evolution (D4 time-resolved); cusp/correlation emergence in J(r) over
  epochs. Requires clean single-optimizer trajectories with mid-training checkpoints.

- **D6 — Why CTNN ≫ FFNN (needs N≥6).** Decode the **message** `m_v` (local field / effective
  density / weighted neighbour count → did message-passing rediscover a local density functional an
  FFNN-of-pairs cannot build?); **body-order (ANOVA)** decomposition (CTNN nonzero 3-/4-body, pairwise
  FFNN exactly zero beyond 2-body); expressivity-at-fixed-cost (K rounds → (K+1)-body at O(N²));
  lazy-vs-rich across architectures.

### Minimal system for each question

| Question | Minimal system | Why |
|---|---|---|
| SR mechanism, δψ maps, soft/stiff eigenfunctions (D1) | N=2 | visible at 2-body; checkable vs exact |
| Cusp×SR 2×2, cusp role (D2) | N=2 | cusp is 2-body; exact target known |
| Effective coordinate, circuit decoding (D3) | N=2 → N≥6 | 1-D latent at N=2; many-body features at N≥6 |
| Lazy vs rich (D4) | N=2 (+ ω, + arch) | measurable anywhere; contrast across regimes |
| Training dynamics (D5) | N=2 | clean, exact-anchored |
| CTNN ≫ FFNN, message decoding, body-order (D6) | **N≥6** | needs ≥3 bodies; vacuous at N=2 |

**Sequencing consequence:** do the mechanism (D1–D5) *in depth at N=2 ω=1.0 first*, then add ω as a
second axis, then go to **N=6** for D6 (CTNN-vs-FFNN). The ω-sweep is reframed from "does SR help
everywhere" (known) to "does the SR-advantage / richness / effective-dim track Γ" (a physics claim).

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

**Phase A-depth (current focus, 2026-06-14):** the mechanism layer (§2.5), all at N=2 ω=1.0 against
exact truth: NTK eigenfunctions → real space and δψ_SR vs δψ_plain maps (D1); cusp×SR 2×2 + the
`J = u_cusp + f_net` decomposition (D2); effective variational coordinate + circuit decoding (D3);
lazy-vs-rich kernel alignment (D4); training dynamics from mid-training checkpoints (D5).

**Consolidation gate A:** Are all tools validated against ground truth? Does the kernel story hold
where we can check it exactly? Is the network lazy or rich here? List surprises before moving on.

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

### Phase C — `N=6` (first genuine many-body; the home of CTNN-vs-FFNN)
Goal: where pair → many-body correlation first matters — the minimal system where **D6** has signal.
- Secure analysis-grade GS across ω (SR primary). Reconcile with the existing
  `architecture_diagnostics` campaign — re-run any diagnostic whose checkpoint fails the §4 gate.
- **D6 (CTNN vs FFNN):** train matched CTNN (message-passing) and FFNN/DeepSet (pairwise) Jastrows;
  decode the message `m_v` (does it build a local density/field?); body-order ANOVA (CTNN nonzero
  3-/4-body, pairwise zero); FFNN-equivalence curve (energy vs params); lazy-vs-rich across arch.
- Carry the depth tools from Phase A-depth: δψ maps, effective coordinate, circuit decoding now on
  the many-body message; does SR help *more* for CTNN than FFNN (better-conditioned S)?
- Re-examine the published `N=6` claims (100% kinetic, random=zero messages) with seed sweeps before
  they anchor a chapter.
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
