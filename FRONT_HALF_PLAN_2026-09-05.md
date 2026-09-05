# Introduction, Theory, Methods: review against the revised second half

Read after the Ch 6–9 restructure, asking one question of every section: *if this
disappeared, would a reader understand Chapters 3, 6 or 7 less well?*

---

## The page budget, which frames everything

| | pp | | | pp |
|---|---|---|---|---|
| §1.1 Mathematical foundations | 1 | | §2.1 Statistical learning | 1 |
| §1.2 Quantum mechanics | 1 | | §2.2 Optimisation | 2 |
| §1.3 Second quantization | 1 | | §2.3 Feed-forward networks | **3** |
| §1.4 Hartree–Fock | 1 | | §2.4 **Architectures for many-body wavefunctions** | **1** |
| §1.5 FCI | <1 | | §2.5 Expressivity & generalisation | 2 |
| §1.6 Quantum dots | 2 | | §2.6 PINNs | 2 |
| §1.7 Wigner markers | <1 | | §2.7 Summary | 2 |
| §1.8 Variational methods | 3 | | | |

**Ch 1 = 9 pp, Ch 2 = 13 pp, Ch 3 = 6, Ch 4 = 5, Ch 5 = 6.** Front half 39 pp against
43 pp of results, discussion and conclusion.

---

## A. What I found

### A1. The proportions are inverted relative to what the second half uses

**§2.4 — the section on which the thesis's central architectural result rests — is one
page.** It sets up permutation symmetry, DeepSets, message passing, the CTNN, and the
relevance of all four to the ansatz. Chapter 7 §7.1 spends five pages on the comparison
that section makes possible.

**§2.3 (activations and initialisation) is three pages**, and the thesis draws exactly one
decision from it: the local energy contains second derivatives, so the activation must be
smooth. Sigmoid, tanh, ReLU, SiLU, Swish, GELU, Mish, Xavier, Kaiming and orthogonal
initialisation are surveyed in succession.

**§2.5 (expressivity and generalisation) is two pages** and is used for nothing.

§2.4 is not deficient — it is precise, well-scoped, and careful in exactly the way the
later chapters are ("we write copresheaf-*style* deliberately: the name reflects the
node/edge transport structure of the updates, not a claim to a formal sheaf-theoretic
construction"). It is compressed to denseness while §2.3 sprawls. **The fix is to cut
around it, not to expand it.**

### A2. Chapter 2 frames the work in a paradigm the thesis does not operate in

This is sharper than "generic ML background". **There is no generalisation problem here in
the statistical-learning sense.** There is no dataset, no train/test split, no held-out
set. The variational energy is evaluated on fresh samples from the model's own
distribution at every step.

Yet §2.1 opens on empirical risk minimisation and the bias–variance tradeoff; §2.5.4
discusses overparameterised generalisation, implicit regularisation and double descent;
and §2.7 then propagates the framing forward:

> Framing learning as empirical risk minimisation makes *generalisation*—not fitting—the
> real problem…

For this thesis the real problem is **trainability and conditioning**, which is what the
Introduction says and what Chapters 6–9 demonstrate. The ERM framing is not merely unused;
it points the reader at the wrong difficulty and the chapter summary then hands that wrong
difficulty forward into the Methods.

### A3. The front half under-supplies four things the second half leans on

This is the half the work is missing — everything else here is subtractive, and cutting
alone would widen these gaps.

| Needed by | Currently |
|---|---|
| **Fixed-node bias.** §8.1's central argument is that agreement with fixed-node DMC is agreement with *another approximation*, and that sitting below a published DMC number can be legitimate. | One subordinate clause, in a subsection titled "Conclusion" at the end of §1.8. The reader is never told why fermionic DMC needs a fixed node (the sign problem), or what the constraint costs. **This is the most important addition.** |
| **The zero-variance principle.** $\Var(E_L)$ is now the discriminator that separates the architectures where energy and $d_{\rm eff}$ cannot, with its own table in §7.1. | An aside about the hydrogen atom in §1.8.3. |
| **The Pareto $\hat k$ theory** — $\hat k<0.7$ gives a finite-variance importance estimator, $\hat k>1$ an infinite-variance one. The whole gradient-quality chain in §7.3.4 turns on it. | Methods §4.5 defines the diagnostic and cites Vehtari; the theory that makes it meaningful appears thirty pages later, inline in Chapter 7. |
| **The symmetric overlap $\mathcal S^2$.** Listed as an invariant in four places; the claim "the two paradigms reach the same *state*" depends on it. | Its estimator is never defined in the front half. |

### A4. §1.6.3 calls the reference an "RHF reference"

> This is the *restricted* HF (RHF) reference and a natural starting point for correlated
> ansätze.

The orbitals are Cartesian harmonic-oscillator eigenstates, not self-consistent HF
orbitals — as §1.5.3 and §3.1 both say. This is a closed-shell **non-interacting** HO
reference. An examiner from electronic structure will catch this in the first ten minutes.

### A5. §1.8.3 describes a different implementation

"Efficient determinant updates: for a single-particle move, the Sherman–Morrison formula
updates the inverse Slater matrix in $\mathcal O(N^2)$…"; "recursive methods and efficient
numerical routines for evaluating $\nabla\Psi_T/\Psi_T$ and $\nabla^2\Psi_T/\Psi_T$".

The thesis uses automatic differentiation throughout and never performs a rank-1
determinant update. This is generic QMC boilerplate that actively misdescribes the work.

### A6. Stochastic reconfiguration is introduced three times, consolidated nowhere

§1.8.1 mentions it before it is defined; §2.2.4 defines the natural gradient; §2.2.5
identifies the QGT with the Fisher metric and SR with natural-gradient descent under it;
§4.4 gives the algorithm. The §2.2.5 treatment is the good one and should own the concept.

### A7. Three stale statements in the Introduction

- **Line 303**, the contributions list, still gives the invariants as "energy, state
  overlap, and the energetic price of ablating a component". $\Var(E_L)$ is missing; I
  updated that list in §7.5, §8.2 and Ch 9 but not here.
- **The Optimiser research question** states two regimes — "negligible for
  well-conditioned VMC, decisive for the ill-conditioned collocation objective" — where
  the thesis now argues three, the third being that SR does active harm once the metric is
  unestimable. The Paradigm question likewise omits the second wall.
- **"The first three we read through a single lens"** — Chapter 7's title was changed
  precisely because §7.4 is physics-informed sampling rather than tangent-kernel analysis.

Also, the diagnostics list omits $\Var(E_L)$ and the ESS/$\hat k$ pair, which between them
carry all of §7.3.

---

## B. Where the external read is right

Verified against source, all of these hold:

- The funnel principle, and its editing test (*if this paragraph disappeared, would a
  reader understand Ch 3, 6 or 7 less well?*). My page table above is the quantitative
  form of the same complaint.
- Chapter 1 should become "the physics problem this thesis solves".
- The RHF language (A4 above — same finding).
- Chapter 2 needs the largest cut.
- The representability/trainability inconsistency. §2.5.6 literally says *"These
  theoretical guarantees justify the use of neural networks"*, against the Introduction's
  *"Representability is cheap; trainability is the difficulty."*
- The generic ML claims. All four verified: SGD noise escaping poor minima (§2.2.2), flat
  minima generalising better (§2.2.6), depth mitigating the curse of dimensionality
  (§2.5.3), modern architectures producing "broad, connected valleys" (§2.5.5).
- The activation section should explain the choice made, not survey the field. And the
  Figure 2.1 caption — "Smooth activations (GELU, Mish) yield stable derivatives, while
  Tanh and Sigmoid have less stable higher derivatives" — is vague in a way nothing in
  Chapters 6–9 is. All four are $C^\infty$. The property meant is presumably the decay and
  dynamic range of the second and third derivatives; say that.
- Chapter endings should hand the reader forward.

---

## C. Where I differ

**C1. Its advice is entirely subtractive.** Twelve points, all cut/compress/reframe. But
the front half's larger defect is that it *omits* four things the revised second half now
leans on (A3). Cutting without adding those would make Chapters 7 and 8 harder to follow,
not easier — particularly §8.1, whose whole argument presumes the reader understands
fixed-node bias.

**C2. "Reduce second quantisation drastically" needs care.** This is a UiO computational
physics thesis and the examiner may well come from that tradition, where second
quantisation, HF and FCI are the shared vocabulary. §1.5.3 already says honestly that they
set the vocabulary rather than supply the numbers. I would compress rather than gut, and I
would keep HF specifically, because both *closed shell* and *correlation energy* — terms
the thesis uses constantly — are defined there.

**C3. It misses the §2.4 imbalance**, which I think is the single most consequential
proportion error in the front half: one page for the architectural distinction that five
pages of Chapter 7 analyse.

**C4. It misses that Chapter 2 is in the wrong paradigm** (A2), not merely too general.
Cutting the generic passages without fixing §2.1 and §2.7 would leave the framing intact.

**C5. Its trajectory is right and its mechanism is half.** "Methods should feel almost
forced" — yes, and §3.2's "Design principles" already achieves exactly that. That
paragraph is the model to propagate *backwards*: Chapter 1 should end on the constraint it
imposes, Chapter 2 on the constraints it imposes, and Chapter 3 should open by naming
them.

---

## D. Proposed plan

Target: Ch 1 9→7 pp, Ch 2 13→9 pp, with roughly 1.5 pp of that redirected into material
the second half needs. Net about −5 pp, and a front half that funnels.

**1. Chapter 1 → "the physics problem this thesis solves."**
Compress §1.1–1.2 (Hilbert space completeness, bra/ket mechanics, operator matrix
elements) from 2 pp to well under one — a competent examiner needs none of it. Compress
§1.3–1.5 to the vocabulary actually used. Fix the RHF language (A4). Rewrite the chapter
opening, which is currently a table of contents in prose. End the chapter on the
constraint: *a compact antisymmetric variational state that can represent correlation
without destroying local-energy conditioning.*

**2. Rebuild §1.8 around what the second half asks of it.**
Delete §1.8.3's inapplicable numerics (A5). Promote the fixed-node approximation from a
subordinate clause to its own short subsection: the sign problem, why a nodal constraint
is imposed, what it costs, and therefore in what sense a DMC number is a benchmark. Give
the zero-variance principle a paragraph of its own, since $\Var(E_L)$ becomes a headline
discriminator. Remove the forward reference to SR.

**3. Chapter 2: cut §2.1, §2.3 and §2.5 hard; leave §2.2.5 and §2.4 alone.**
Replace the ERM/bias–variance opening with a short statement of what learning means *here*
— minimising an expectation under a distribution the model itself defines, with no
held-out set and therefore no generalisation problem in the usual sense, but a severe
conditioning and sampling problem instead. Reduce §2.3 to one page: the local energy needs
two clean derivatives, here is the activation and why, here is the initialisation and why.
Fix the Figure 2.1 caption. Delete §2.5.4 and §2.5.5 outright and compress §2.5.1–2.5.3 to
a short statement that approximation theorems establish possibility and nothing else.

**4. Add the missing estimator theory** (A3): $\hat k$'s meaning next to its definition in
Methods §4.5, and the $\mathcal S^2$ estimator where the invariants are first named.

**5. Rewrite §2.7 as a handoff, not a summary.** Three constraints, stated as constraints:
a second-order operator in the objective; a Coulomb-singular, node-carrying state; and a
sampled measure that must remain estimable. Then Chapter 3 opens by answering them, which
is what "Design principles" already does.

**6. Fix the three stale Introduction statements** (A7).

**Order:** 6 first (small, isolated), then 2 and 4 (additive, and they make the cuts
safe), then 1, 3 and 5.
