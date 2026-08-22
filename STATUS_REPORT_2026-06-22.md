# Status Report — Three Questions (CTNN-vs-FFNN, SR-vs-Adam, Collocation-vs-VMC), One Lens

> **Framing note (rebalanced 2026-06-22):** an earlier draft of this report and its companion plan
> treated *effective dimension* as the thesis spine. That over-indexed on one diagnostic. The thesis
> answers **three co-equal questions** — Q1 architecture (CTNN vs FFNN), Q2 optimizer (SR vs Adam),
> Q3 paradigm (collocation vs VMC) — and the tangent-kernel / dimension picture is the **shared lens**
> that serves all three. The dimension result below is the worked example that set the depth bar, not
> the organizing axis. See [plans/2026-06-22_dimension-program-and-roadmap.md](plans/2026-06-22_dimension-program-and-roadmap.md).

**Date:** 2026-06-22
**Author:** consolidation + audit session
**Companion plan:** [plans/2026-06-22_dimension-program-and-roadmap.md](plans/2026-06-22_dimension-program-and-roadmap.md)

---

## 0. One-paragraph state of the work

We have a strong, exact-anchored **anchor** (N=2 ω=1: SR = NTK whitening, overlap²=1, κ(S)≈10¹²,
eff-rank(S)≈1.2) and a scatter of **single-seed N=6 fragments**, but the connective tissue is
missing and several finished runs were never written up. The most important thing this session
surfaced: a clean, **2-seed, kernel-diagnosed CTNN-vs-DeepSet comparison at N=6 ω=1 has been sitting
unconsolidated in untracked result dirs**, and it already answers the central Q1 question —
*why does CTNN beat a pairwise FFNN* — mechanistically. The deficit in this project is **not tooling
and not ideas** (both are abundant: `bodyorder.py`, `collectivity.py`, trajectory checkpointing, the
full `O/S/K` kit all exist). It is **consolidation + disciplined sweeps**: we run, fail to write up,
hit a null, and re-plan. This report consolidates, then organises the program around **three co-equal
questions** — Q1 (CTNN vs FFNN), Q2 (SR vs Adam), Q3 (collocation vs VMC) — each pursued to full
mechanistic depth through one shared lens (the tangent-kernel / dimension picture, var(E_L), ESS,
zero-variance), and each scaled to Wigner and to higher N.

---

## 1. Ledger: what is proven / partial / glossed / missed

| Item | Status | Note |
|---|---|---|
| SR = NTK whitening (T1–T3) | **proven** | N=2 ω=1, exact (cos_sr 0.997, cos_plain 0.04); holds at N=6 (cos_plain 0.07–0.16). |
| Why CTNN ≫ FFNN (D6/B3/B9/X2) | **answered, unconsolidated** | §2 below — eff-rank + var(E_L) + equivalence ladder, ω=1, 2 seeds. |
| Backflow fixes N≥6 nodes (B8) | **proven** | Jastrow-only plateaus at fixed-node error; backflow reaches/below DMC. |
| Learned Kato cusp (C1) | **proven** | N=2, cusp recovered. |
| Effective coordinate (D3) | **partial** | 1-D "correlation-strength" knob identified at N=2; causal validation not done. |
| Lazy vs rich (D4) | **partial** | Endpoint CKA only; trajectory CSVs exist but uncollated. |
| Cusp × SR 2×2 (D2) | **¼ done / retracted** | Only cusp-ON ran → NULL. The cusp-OFF arm (where SR is *predicted* to win) is unrun. |
| Training dynamics / spectral bias (T4/D5/A5/X3) | **missed** | ckpt infra exists, never exercised systematically. |
| Body-order ANOVA (T5/B4) | **built, unrun** | `src/analysis/bodyorder.py` exists; no result reported. |
| Two estimators / collocation conditioning (T6) | **glossed** | Reframed into "ESS collapse"; the var(strong) vs var(weak) + direction comparison never done. |
| κ(S) / eff-rank vs ω (Phase B, A1) | **missed** | The N=2 ω-sweep — biggest depth-before-breadth gap; blocks the physics claims. |
| Network-internal Wigner order param (C2) | **missed** | Physical Lindemann exists in notebooks; network-internal version not built. |
| Dual-track VMC-vs-collocation (§7) | **missed** | The program's centerpiece; *and* VMC+SR (−0.028%) may have overtaken collocation (+0.01%) — unchecked. |
| GS-quality gate (§4) | **worked around** | Defined; seed-stability + train-route agreement routinely skipped. |
| O/S/K + representation + bodyorder + collectivity tooling | **~80% built** | The bottleneck is execution + consolidation, not infrastructure. |

**Reframe of the "nulls":** the Jun-20 SR null tested the one quadrant (N=2, cusp-ON, ω=1) where the
thesis *predicts* SR cannot help; nothing about SR was disproved — a single-seed claim we should not
have promoted was retracted. The k⁴/k² conditioning law is in limbo (underpowered, not disproved).

---

## 2. The surfaced result: CTNN's low-dimensional tangent space (N=6, ω=1, SR-trained)

From `results/analysis/2026-06-15_{2x2,eq}_N6w1_*` (≥2 seeds for CTNN; never in the journal):

| arch | params | var(E_L) | **eff-rank(S)** | κ(S) | cos_plain | E error |
|---|---|---|---|---|---|---|
| CTNN-vcycle (seed 0) | 80k | 0.026 | **1.60** | 8.3e7 | 0.079 | +0.029% |
| CTNN-vcycle (seed 1) | 80k | 0.026 | **1.46** | 6.9e7 | 0.097 | +0.040% |
| DeepSet-big | 48k | 0.097 | 3.35 | 1.9e10 | 0.068 | +0.071% |
| DeepSet-match | 89k | 0.092 | 3.60 | 2.1e10 | 0.070 | +0.098% |
| DeepSet-xl | 164k | 0.033 | 4.76 | 1.4e7 | 0.156 | −0.052% |

**What it says.** CTNN represents the same ground state in a **~1.5-dimensional** effective tangent
space; DeepSet needs **3.4–4.8**, and even at **2× the parameters** (xl) its var(E_L) does not reach
CTNN's. This is one fact wearing three hats:
- **X2 (trainability):** lower eff-rank ⇒ fewer modes for SR to whiten ⇒ the conditioning advantage.
- **D6 (expressivity):** the message-passing inductive bias compresses the correlated state into
  fewer collective directions.
- **B3 (equivalence):** "how many FFNN params to match CTNN" → **>2× and still short on var(E_L)**.

**Robust vs fragile in this table:** eff-rank and var(E_L) are the trustworthy signals; **κ(S) is
NOT clean** across DeepSet sizes (xl has *lower* κ than match despite more params — the supported-tail
floor moves), so we report eff-rank, not κ, as the dimension.

### Fairness audit of the dimension measurement

- **Identical (fair):** participation-ratio formula, `rel_tol`, centering (connected QGT), and sample
  count (768) are the same for every arch ([diagnostics.py:47](src/analysis/diagnostics.py#L47)).
  eff-rank (1.5–4.8) ≪ 768, so it is **resolved, not sample-capped** (`numerical_rank`=767 *is*
  capped and must not be reported as the dimension).
- **Confound to fix before scaling:** each net's `O` is built on samples from **its own |Ψ|²**
  ([run_depth_analysis.py:135](scripts/run_depth_analysis.py#L135)). Defensible (the QGT is
  intrinsically own-measure) but it mixes "geometry of the ansatz" with "where it puts mass." A clean
  cross-architecture comparison must evaluate every `O` on a **common probe set**.
- **Scaling risk:** 768 samples resolve eff-rank≈few; at higher N / lower ω the true dimension may be
  larger and the rank becomes sample-limited. Must report **eff-rank vs n_samples** convergence and
  scale samples with N.

The hardened protocol is specified in the plan (common probe set; sample-convergence; report at both
**matched params** and **matched accuracy** — they answer different questions).

---

## 3. What *are* these dimensions, and how do they map to the Hamiltonian?

eff-rank(S) is the **effective dimension of the variational manifold at the solution**: a ~10⁴-param
net excites only ~1.5 independent collective combinations. The NTK eigenvectors are functions
δψ_a(x) over configuration space — *collective deformations* of the wavefunction. "What are they"
has three concrete, falsifiable readings, in increasing depth:

1. **Response / susceptibility (cleanest).** The Hamiltonian has a few physical knobs:
   the trap ω (one-body), the Coulomb strength λ (two-body), L_z. The exact GS Ψ\*(ω, λ) traces a
   low-dimensional manifold as those knobs vary; its tangents are physical responses — ∂Ψ\*/∂ω
   (**breathing**), ∂Ψ\*/∂λ (**correlation-hole deepening**). **Hypothesis: the net's useful tangent
   space ≈ span of these physical responses.** *Test:* project the top NTK eigenfunctions onto
   {∂_ω logΨ\*, ∂_λ logΨ\*, …} (exact at N=2; finite-difference in (ω,λ) via the reference solver at
   N=6). If eff-rank≈2 and it is spanned by {breathing, correlation}, the dimensions are *named*.

2. **Excitation / collective-mode.** Imaginary-time flow −(H−E)Ψ removes excited-state contamination
   along directions built from low-lying excitations |n⟩ weighted by 1/(E_n−E). The dimensions
   should ≈ the **count of low collective modes** the GS optimization must resolve (monopole/breathing,
   quadrupole, relative/correlation; the dipole/Kohn mode decouples by Kohn's theorem). *Test:* project
   NTK eigenfunctions onto exact excited states (N=2 exact; few-state Lanczos at N=6).

3. **Operator decomposition.** Expand each δψ_a(x) in physical one- and two-body operators (density
   moments ⟨r^k⟩, pair operators ⟨f(r_ij)⟩, ⟨L_z⟩) — read each dimension out as an observable's
   generator (the "invert the Hamiltonian" spirit, C9).

**The Wigner prediction (why we scale to low ω).** Toward the Wigner crystal the low excitations
become the **normal modes (phonons) of the rigid crystal** — a small, structured set (breathing +
shear + rotation + relative vibrations). So eff-rank should stay small / **drop** toward Wigner
(consistent with the measured intrinsic-dim drop 8.3→5.0), and the CTNN-vs-DeepSet *gap* should
**grow** (the consensus/collective state is exactly what message-passing represents and a separable
DeepSet cannot). **The scaling claim (why we go to higher N):** eff-rank(CTNN) should grow only as
fast as the **number of physical collective modes** (≈O(N) phonons, or fewer), while eff-rank(DeepSet)
grows with its separable/pairwise structure. The *ratio* vs (N, ω) is the thesis-grade scaling law.

---

## 4. The unification (uniting the kinetic-energy and low-rank pictures)

> The trained wavefunction lives in a **low-dimensional, cusp/kinetic-dominated tangent space.** This
> single object is all three angles plus the collocation thread:
> **Angle 1** — low-rank S with one stiff (cusp) mode ⇒ huge κ ⇒ plain gradient collapses onto the
> soft mode (cos_plain≈0.07), SR whitens it. **Angle 2** — CTNN reaches a *lower-rank* tangent space
> than DeepSet ⇒ better conditioning *and* lower var(E_L) at fewer params: that is why it wins.
> **Angle 3** — the few effective coordinates *are* the physical collective modes of H (breathing /
> correlation / phonons). **Collocation/Laplacian (T6)** — var(E_L)→0 (what makes the discriminator
> meaningful) is a Laplacian/zero-variance property; the sampling *measure* sets whether the low-rank
> geometry is resolvable (ESS).

This is the **expected synthesis** the three questions converge on (the destination, not the
organizing axis): **the physics compresses into a low-dimensional tangent space of physical collective
modes, and architecture (Q1) / optimizer (Q2) / paradigm (Q3) are each about how faithfully they
resolve it.** We reach it by answering the three questions equally; every null is a boundary condition
of *this*.

---

## 5. Immediate priorities (detail in the plan)

1. **Consolidate** the §2 result into the journal as a proven Angle-2 finding (done this session).
2. **Harden** the dimension measurement (common probe set, sample-convergence, matched params +
   matched accuracy) at N=6 ω=1.
3. **Scale** eff-dim across ω (toward Wigner) and N — the new spine experiment.
4. **Map** the dimensions to H (response projection / excitation overlap) — the "what are they" answer.
5. Re-open **cusp-OFF 2×2**, **dual-track**, and **var(strong)-vs-var(weak)** as they fall out of the
   above.
