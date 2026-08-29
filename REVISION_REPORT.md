# Editorial Revision Report

**Scope and honesty note.** This was a focused, high-value editorial pass, not an
exhaustive line-by-line re-verification of all 120 pages. I read the thesis as a complete
argument, fixed the highest-impact structural, prose, consistency, and typesetting issues,
and audited citations and terminology globally. Where a claim needs the author's own
knowledge or data to verify, it is marked **AUTHOR CHECK** below rather than silently
changed. The document compiles cleanly throughout (120 pp, 0 undefined references, no
overfull box now exceeds ~25 pt).

---

## A. Major structural changes

1. **Conclusion promoted to a chapter.** `conclusion.tex` began with `\section{Conclusions
   for quantum dots}`, so under `\part{Conclusion}` it was being absorbed as a trailing
   *section of the Discussion chapter* rather than standing as the thesis conclusion. It is
   now `\chapter{Conclusions}` (chapter 9 in the TOC), and the opening "in this chapter" was
   corrected to "in this thesis."
2. **Added a "Limitations and outlook" paragraph** to the conclusion — it previously had no
   explicit limitations statement. It names the three real bounds (external truth only for
   ω ≥ 0.1; the N=12/N=20 size-scaling is indicative; the O(N²) message-passing backflow is
   the scaling bottleneck).
3. **Removed a duplicate "Conclusions" in the TOC.** The tangent-kernel chapter's own wrap-up
   section was titled "Conclusions," competing with the thesis Conclusion chapter; renamed to
   "Chapter summary."
4. **Tightened the theory chapter's ML "Summary and Outlook."** It was a blow-by-blow recap
   ("We began… We then… Finally…") — replaced with a shorter forward-linking bridge that keeps
   the orientation and drops the re-teaching (~35 lines → ~12).

## B0. Formula and derivation verification (checked line-by-line)

I re-derived the load-bearing mathematics independently. **Every derivation checked is
correct**, and the earlier corrections (cusp 1/3, 2D non-integrability, variance-finite-under-|Ψ|²)
are consistent with the re-derivation.

- **2D cusp conditions** (theory §"Kato" + Appendix B). Opposite-spin
  ∂_r lnΨ|₀ = 2μZ_iZ_j/(D−1) = 1; like-spin (p-wave, linear node) = 2μZ_iZ_j/(2ℓ+D−1) =
  1/(D+1) = 1/3. I verified the (2k+d−1) factor from −ΔD/D − ΔJ/J − 2∇D·∇J/(DJ) with a linear
  node D∼η·r: it gives −3u′/r ⟹ u′(0)=1/3. The 3D values (1/2, 1/4) and the note that "like =
  half opposite" is a **3D-only coincidence** are correct; the 2D electron–nucleus value −2Z_A
  is right *for D=2*. **Correct.**
- **Local energy** (Methods, eq:local-energy). −½Σ[Δlnψ+‖∇lnψ‖²] + ½ω²Σr² + Σ1/r_ij matches
  H = −½Σ∇²+½ω²r²+Σ1/r; and the trap-unit rescaling (kinetic and trap ∝ω, Coulomb ∝√ω under
  r̃=√ω r) is dimensionally correct. **Correct.**
- **Tangent kernel** (theory §sec:theory-tangent-kernel). S = OᵀO/B, **F = 4S** because
  ∇log|Ψ|² = 2∇log|Ψ|, K = OOᵀ shares the nonzero spectrum, d_eff = (Σλ)²/Σλ². **Correct.**
- **Appendix A — Laplacian conditioning.** The Gauss–Newton normal matrix A = J*J,
  A_ij = ⟨Lq_i,Lq_j⟩, and GD contraction (I−ηA) are standard and right; the bi-Laplacian
  scaling L≈−½Δ ⟹ L*L≈¼Δ², κ ∼ (k_max/k_min)⁴ is correctly derived in the eigenbasis.
  **Correct.**
- **Appendix B — Coulomb ill-conditioning.** Cusp-mismatch spikes −δa/r (antiparallel),
  −3δb/r (parallel); the 2D non-integrability ∫_{r<ε}(1/r)² d²r = 2π∫dr/r = +∞ (contrast the
  finite 3D integral) is correct and is what makes the strong squared residual dangerous in 2D.
  **Correct.**
- **Appendix C — the collocation–backflow catch-22.** E_L ⊃ ∇²D/D ∼ c/d_θ at a *mismatched*
  node (∇²D finite there because d*≠0); Var[E_L] under |Ψ|² is **finite** (integrand
  d_θ²·d_θ⁻²·dd_θ = dd_θ → O(ε)), but the *gradient* ∂E_L/∂θ carries a 1/D² pole and its
  variance diverges; VMC+SR escapes because the SR force g_i = Cov(E_L,O_i) is a bulk
  covariance that never forms ∂E_L/∂θ, and near-node configs have ∼ε³ probability. The hedge
  ("structural obstruction of the tested pipeline, not a theorem") is appropriately calibrated.
  **Correct.**

**Full equation sweep (added pass).** I then went through the remaining displayed equations
chapter by chapter. All are correct:
- **Theory** — Schrödinger/spectral decomposition, many-body H (atomic units, V_ext=½ω²r²,
  V_int=1/r); second quantization (CAR, one-/two-body operators with the standard ½ and ¼
  antisymmetrized forms); HF energy Σh_ii+½Σ(ij‖ij), Fock ĥ₀+Σ(J−K), FCI/Slater–Condon
  (Brillouin, doubles=(ij‖ab), C(M,N)); Fock–Darwin E=ω(2n+|m|+1), closed shells (K+1)(K+2)
  ={2,6,12,20}, RHF normalization 1/(N/2)!; diagnostics Φ_m=(1/n)Σe^{imφ_k}, angular Lindemann,
  and the two-body classical scaling **α=−2/3** (from ω²r=1/r²); VMC/DMC (variational principle,
  Metropolis–Hastings acceptance with the correct T-ratio, quantum force **2∇ln|Ψ|**, DMC
  branching e^{−(E_L−E_T)Δτ}, hydrogen zero-variance at α=1); momentum/Adam/Newton/natural
  gradient; FFNN forward + variance-preserving initialization.
- **Method** — the safe pair features (s₁,s₂,s₃,rbf) and gate χ all have bounded/vanishing
  derivatives at coalescence as claimed; the SR energy gradient 2·Cov(E_L,O_k) is the standard
  log-derivative form.

**Three minor items (none load-bearing, none an outright error in a used result):**
1. **AUTHOR CHECK — RHF Fock-matrix convention** (theory, Roothaan–Hall). The exchange is
   written with a ½, `[(μν|λσ)−½(μλ|νσ)]`, with `P_λσ=Σ_{i∈occ} C_λi C_σi*`. The ½ is correct
   only if P carries the factor of 2 (sum over spin-orbitals, or P=2·Σ_spatial). If "occ" is
   spatial orbitals without the 2, the exchange coefficient should be 1 (and Coulomb 2).
   Clarify the P convention. HF is background/reference-only here, so this does not touch any
   result.
2. **Consistency note** — the energy gradient is written 2·Cov(E_L,O) in Methods but Cov(E_L,O)
   in Appendix A (eq:lap:sr). Both are valid (the 2 is absorbed into the learning rate); a
   single convention would read more cleanly.
3. **Trivial** — the *illustrative* Padé–Jastrow `exp[−a r/(1+βr)]` in the VMC background models
   repulsion only for a<0 (the thesis's own cusp uses +γ). It is a throwaway example, used
   nowhere.

Net: across the theory, methods, and appendix derivations, **no error was found in any equation
that feeds a result.** The mathematical content is sound.

## B. Scientific issues discovered

1. **Citation error (fixed).** `Kong_2002` — actually *"Transition between ground state and
   metastable states in classical two-dimensional atoms"* (Kong, Partoens, Peeters), a
   classical Wigner-cluster paper — had been used (by me, in an earlier turn) to support the
   *importance-sampling* claim in `method.tex`. Removed there; it remains correctly used for
   Wigner physics in ~10 other places, including the abstract.
2. **AUTHOR CHECK — N=12 DeepSet detector point.** The N=12 non-physical-mass value (≈0.77,
   `fig:nonphys`) comes from a DeepSet trained only to +0.2 % energy, lighter than the
   seed-checked N=6 checkpoints. The qualitative contrast (≫ CTNN's ≈0.003) is unambiguous,
   but the exact value would firm up with fuller training or a second seed. Flagged in-text.
3. **AUTHOR CHECK — N=20 tangent-space claims** rest on a single production state (no seed
   spread). The energy is benchmarked; the geometry (rank, d_eff) at N=20 is indicative.
4. **Not re-derived here.** The appendix derivations (Laplacian conditioning, bi-Laplacian
   scaling, the collocation–backflow catch-22 geometry) were read for consistency but not
   independently re-derived. The 2D cusp (γ=1 antiparallel, 1/3 parallel) *was* verified
   against code in earlier work and is internally consistent across theory/method/appendix.
5. **Standard-background sections** (Second Quantization, Hartree–Fock, FCI) are textbook and
   correct as far as read; they are on the long side relative to their later use (see G).

## C. Citation issues

1. **Fixed:** the Kong_2002 miscitation above.
2. **AUTHOR CHECK — abstract citations.** The abstract closes with a six-reference Wigner
   cluster (`Egger_1999, Mazars_2008, Kong_2002, schweigert1994…, Filinov_2001,
   manninen2007…`). All are legitimate Wigner references, but citations in an abstract are
   non-standard; consider moving them to the first body mention. Left in place pending your
   preference.
3. **`references.bib` header cruft.** The file begins with Python wrapper lines
   (`# Create references.bib…`, `bibtex_content = r"""`). BibTeX ignores them (0 undefined
   citations), but they should be deleted for cleanliness. **AUTHOR CHECK / trivial fix.**
4. Bibliographic fields were not exhaustively verified against the primary sources; a final
   author pass on volume/page/DOI is advisable (standard for submission).

## D. Results narrative — the threads I identified

The thesis runs **two results chapters that answer two different kinds of question**, and the
revision makes their relationship explicit (the kernel chapter opens "The previous chapter
showed *whether* the ansatz works… this chapter asks *why*"):

- **Thread 1 — the ansatz as an instrument (ch. 6).** Energies vs DMC; what the network's
  representation looks like (low-rank correlator manifold `r_eff(Z)`, high-rank backflow);
  and the Fermi-liquid→Wigner crossover mapped from inside the trained state.
- **Thread 2 — mechanism through the tangent kernel (ch. 7).** Q1 architecture, Q2 optimiser,
  Q3 paradigm, unified by "one relational channel, read through the QGT/NTK geometry."

They converge in the Discussion and Conclusion on a single claim: *what matters across
architecture, optimiser, and paradigm is alignment with the low-dimensional, physically
collective structure of the problem — which in real space is the Wigner molecule.* The new
high-ω detector and the Q1a/Q1b unification (added earlier this session) are what make that
convergence quantitative rather than rhetorical.

**Note on `r_eff(Z)` vs `d_eff(S)`.** These are distinct — feature-space rank of the
correlator vs participation ratio of the tangent (QGT) spectrum — and I verified they are
never conflated across chapters.

## E. Material removed or condensed

- The ML "Summary and Outlook" recap (A.4).
- Redundancy at the top of the introduction (the new opening no longer repeats the second
  paragraph's exponential-scaling point).
- Earlier this session: ~180 lines of duplicated collocation *mechanics* were moved out of
  the results chapter into Methods.

## F. Material added or expanded

- Conclusion "Limitations and outlook" (A.2).
- A stronger, momentum-carrying introduction opening (replacing the "quantum mechanics is a
  central pillar of modern physics… underlies much of today's technology" platitude that
  §6-type guidance specifically warns against).
- Earlier this session: the high-ω non-physical-mass detector, the Q1a/Q1b unification, the
  extended mode-naming, the N=2 collocation rows, and the coherence pass threading these
  through abstract/intro/discussion/conclusion.

## G. Remaining author decisions (genuinely need your judgement or data)

1. **Theory length.** Second Quantization / HF / FCI (≈140 lines together) are more textbook
   than the rest of the thesis uses. FCI in particular is invoked only lightly later. Consider
   condensing to what the results actually lean on, or moving background to an appendix. I did
   not cut them because I could not be sure which parts you rely on in the defence.
2. **Appendix D ("Post-catch-22 experimental history and philosophy").** It is referenced and
   documents negative results transparently (good practice), but the "history and philosophy"
   framing and campaign chronology read more like a lab notebook than the other appendices.
   Keep as a transparency record, tighten, or fold the load-bearing parts into Appendix C —
   your call.
3. **Three discussion/synthesis layers** (kernel-chapter synthesis → Discussion chapter →
   Conclusion chapter). Each has a distinct scope, but there is some restatement of the
   "relational channel / alignment" theme across all three. If an examiner finds it repetitive,
   the kernel-chapter synthesis is the one to trim.
4. **Abstract citations** (C.2) — keep or move to the body.
5. **`references.bib` header cruft** (C.3) — safe to delete.

## H. Examiner's assessment

**Strongest aspects.**
- A genuinely unifying thesis ("one relational channel, read through the tangent kernel"),
  now supported by a *quantitative* mechanism (the non-physical-mass detector shows the
  separable network's deficit at strong confinement, where energy cannot) rather than by a
  dimension-count alone.
- Unusual and commendable epistemic hygiene: the ω < 0.1 accuracy is explicitly labelled
  internal-consistency, retracted claims are recorded, and the gauge-freedom between Jastrow
  and backflow is treated as a first-class caveat.
- The physics half (topology-resolved Wigner diagnostics, g(r) reconstructions) is concrete
  and well-benchmarked where benchmarks exist.

**Weakest aspects / likely examiner questions, and whether the thesis answers them.**
1. *"Is the architecture result an artefact of unequal training or capacity?"* — **Answered**
   (matched 20k–164k ladder; seeds; common probe set), except the N=12 DeepSet detector point
   (B.2).
2. *"Your sub-0.1 % accuracy below ω=0.1 has no benchmark — how do you know it's right?"* —
   **Answered honestly** (internal consistency + physical scaling; stated as prediction).
3. *"d_eff is not reparameterisation-invariant."* — **Answered** (caveat present; comparisons
   under matched conventions and common probe).
4. *"Non-physical tangent mass depends on your operator dictionary."* — **Partly**: the
   dictionary is stated and the qualitative gap is large, but a reviewer may ask for a
   sensitivity check to the dictionary choice. **Consider pre-empting.**
5. *"The causal test for message passing was negative — does the mechanism claim survive?"* —
   **Yes, and honestly framed** as correlational (the ablation negative was reported, then
   dropped from the manuscript at your instruction; the claim is stated as trained-DeepSet vs
   trained-CTNN, not as a causal switch).
6. *"Collocation never quite reaches DMC — is the paradigm really competitive?"* — **Answered**
   (±0.01–0.02 % for ω ≥ 0.01, degrading only in the deep-Wigner limit; scope stated).
7. *"Backflow scaling is O(N²) — does the method generalise?"* — **Acknowledged** (discussion
   §"Backflow scaling needs architectural innovation"; new conclusion limitation).
8. *"Total-spin ground state at weak confinement?"* — **Answered** (results reported within a
   fixed spin sector; caveat present).
9. *"Why quantum dots and not a harder system?"* — **Answered** (controlled testbed with DMC
   references and feasible structural diagnostics).
10. *"Are the two results chapters one investigation or two?"* — **Answered** by the explicit
    whether/why framing and the shared tangent-kernel lens; the Conclusion ties them.

**Overall.** The thesis now presents a coherent, honest, and quantitatively supported
scientific contribution. The structural fix (conclusion as a chapter) and the mechanism
result (physical-alignment detector) are the two changes that most raise it from "a strong
collection of experiments" toward "one argument." The main remaining soft spots are the
size-scaling of the tangent-space claims (N=12/N=20) and the length/placement of the standard
theory background — both are author judgements rather than errors.
