# Grand Mechanism Program — What the CTNN Computes, and Why It Wins

**Created:** 2026-07-02. **Status:** Active spine. Supersedes the dimension-centric framing of
`2026-06-22_dimension-program-and-roadmap.md` (that program's tangent-kernel d_eff is now *one probe
among many*, folded into Thread 1). **Companion results file:** `Thesis/results_kernel.tex`.

---

## 0. The reset (read this first)

The three-questions program answered a narrow diagnostic (effective tangent dimension) to death and
walked past the physics. The existing thesis (`results.tex` §"What the networks learn",
`architecture_diagnostics/DIAGNOSTIC_SUMMARY.md`) already contains a **much richer, half-finished
mechanism**: message passing buys +22–30% energy that is **100% kinetic**; random messages equal zero
messages; the edge embedding is rank ≈1.4 (one scalar/pair), the node ≈2.3; the CTNN carries genuine
2–3× beyond-pairwise (3-body) correlation; the backflow **switches physical role** at the Wigner
crossover; and the ansatz spontaneously splits into a **low-rank collective correlator** (r_eff ≤ 3)
and a **high-rank local backflow** (~2N) held *orthogonal* (CKA > 0.98). This program's goal is to
**finish and explain that mechanism**, and to answer every open thread — architecture internals,
optimizer×paradigm, low-ω physics, and scaling.

**Three lessons carried in (hard-won this cycle):**
- **Seed before promoting.** The single-seed d_eff "inversion" was retracted under seeding. ≥3 seeds
  (or, at init, ≥3 inits) before any claim enters the thesis.
- **Untangle confounds.** The "CTNN ≈ DeepSet energy" result had a hidden confound: both arms shared
  the *same message-passing backflow*, so message passing was never actually removed. Every
  CTNN-vs-FFNN claim must state exactly what is and isn't ablated.
- **Connect symptom to mechanism.** var(E_L) (my discriminator) ∝ curvature of logΨ = the kinetic
  energy Fig I measures. Report the mechanism, not just the symptom.

## The one question

> **What does the message-passing graph structure compute that a separable network cannot, why does it
> lower the kinetic energy, how does that change across the quantum→Wigner crossover and with N, and
> which optimizer/paradigm resolves it best?**

Everything below is a face of this.

---

## Thread 1 — Architecture: what the graph computes and why it wins

### T1.1 Backflow-confound-free CTNN-vs-FFNN (the honest comparison)
- **Sharp Q:** how much is the *Jastrow's* message passing worth, separately from the backflow's?
- **Known:** ablation (zeroing trained-CTNN Jastrow messages) = +22–30% kinetic; but my from-scratch
  DeepSet tied CTNN *while both kept the CTNN backflow*.
- **Do:** the full 2×2 {CTNN-Jastrow, DeepSet-Jastrow} × {CTNN-backflow, no/DeepSet-backflow}, trained
  from scratch to matched quality at N=6 ω∈{1,0.1,0.01}, seeded. Plus the *trained-net* message
  ablation on each. Decompose ΔE into kinetic vs potential (reproduce Fig I) for each arm.
- **Predict:** message passing (Jastrow or backflow) is worth a large *kinetic* gain vs a fully
  separable ansatz; the backflow can substitute for Jastrow messages (they overlap), so the "energy
  tie" was backflow doing the work. The clean statement: *no message passing anywhere* ≫ any MP.
- **Output:** the 2×2 energy+kinetic table; the "what MP is worth" figure.

### T1.2 The two feature spaces — do they exist, and why kept apart?
- **Sharp Q:** is the correlator manifold (r_eff ≤ 3) the *same* low-dimensionality as the tangent
  d_eff (~1–4)? Why is the backflow (rank ~2N) orthogonal to it (CKA > 0.98) — forced or learned?
- **Do:** (a) on the same checkpoints, measure correlator feature-rank, tangent d_eff, and their
  principal-angle overlap — are they the same subspace? (b) turn the backflow on/off *during a
  gradient step* and measure whether the correlator tangent space rotates (learned separation) or is
  structurally blind to it (forced by the ×-factorization). (c) intrinsic dimension (TwoNN) of both.
- **Predict:** feature-rank and tangent-d_eff track each other (one collective object); the
  orthogonality is *mostly forced* by Slater×Jastrow×backflow but sharpened by training.
- **Output:** the unified low-dimensionality figure (feature rank ↔ tangent d_eff ↔ intrinsic dim).

### T1.3 What the messages carry (decode the edge scalar + node 2D)
- **Sharp Q:** the edge embedding is ~1 scalar/pair — *what* scalar? The node ~2D — which two?
- **Do:** regress the edge scalar (top PC of edge_embed) against {r, spin-pair-type, local density,
  #neighbors within cutoff}; regress node PCs against {r_i², spin, local energy}. Decode as a learned
  pair pseudopotential u_edge(r; spin, environment) and a node [density×spin] code. Sweep ω.
- **Predict:** edge scalar ≈ a monotone, spin- and environment-modulated pair interaction strength
  (a learned pseudopotential); node ≈ [energy/density] × [spin], matching DIAGNOSTIC_SUMMARY.
- **Output:** the decoded message figure (edge scalar vs r by spin/environment; node code).

### T1.4 Where does the 3-body come from? (localize in the message passing)
- **Sharp Q:** the 2–3× environment-sensitivity is genuine 3-body — which MP step / V-cycle stage
  produces it?
- **Do:** re-run the environment-conditional variance (DIAGNOSTIC_SUMMARY Fig E) after truncating to
  1 MP step, 2 steps, down-only, down+bottleneck, full V-cycle. Also a clean fixed-pair-distance 2D
  construction (particle at the center of a fixed square) for an unconfounded 3-body test.
- **Predict:** 3-body appears at the *second* message exchange (and is amplified by the bottleneck);
  one MP step is ~pairwise.
- **Output:** 3-body-signal vs MP-depth curve.

### T1.5 What is the V-cycle *for*? (multiscale = scale separation)
- **Sharp Q:** does the coarse bottleneck carry the *global collective* coordinate while the fine
  passes carry local structure — i.e., is the multigrid separating scales the way the physics does?
- **Do:** read out bottleneck (16-dim) vs fine node/edge activations; correlate each with global
  observables (total ⟨r²⟩/breathing, shell occupations, ⟨L_z²⟩) and local ones (pair distances). CKA
  between bottleneck code and the correlator's leading PC. Ablate the bottleneck (identity skip) and
  measure the energy/kinetic and d_eff cost. Compare V-cycle vs flat CTNN (same params).
- **Predict:** bottleneck ≈ the global collective coordinate(s) (breathing, shell reorganization); its
  removal costs the collective part of the correlation and raises d_eff/kinetic. This *is* "why the
  graph structure helps."
- **Output:** the scale-separation figure (bottleneck↔global, fine↔local); V-cycle ablation table.

### T1.6 The kinetic-smoothing mechanism (why MP lowers kinetic)
- **Sharp Q:** *why* does coordinated messaging lower ½|∇logΨ|²?
- **Do:** decompose ∇logΨ into pairwise-additive vs message (coordinated) parts; show the message part
  cancels the roughness of the pairwise sum (lower |∇logΨ|² at fixed positions). Connect to var(E_L).
- **Predict:** the coordinated field is smoother (lower curvature) → lower kinetic → lower var(E_L);
  the pairwise sum over-counts near-coalescence gradients that messages regularize.
- **Output:** the kinetic-decomposition figure tying Fig I to var(E_L).

### T1.7 N-scaling of the mechanism (do the transitions move?)
- **Sharp Q:** do the φ↔ψ transition, the ω≈0.1 PINN–CTNN coupling switch, the backflow
  force-reversal, and the 3-body signal shift in ω with N (as Γ scales)?
- **Do:** extend the DIAGNOSTIC_SUMMARY probes (attribution, force-alignment, 3-body, coupling) to
  N=12, N=20 on existing checkpoints (no training). Overlay the transition-ω vs N.
- **Predict:** transitions shift to *higher* ω as N grows (Wigner sets in earlier in ω for more
  electrons); the mechanism is N-robust in form.
- **Output:** transition-ω vs N phase-diagram.

---

## Thread 2 — Optimizer × Paradigm: SR where it actually matters

### T2.1 SR vs Adam **on collocation** (the untested quadrant) ★
- **Sharp Q:** does SR beat Adam for *collocation* (κ~10⁸, measure not preconditioning) far more than
  for VMC (|Ψ|² self-preconditions)?
- **Do:** the 2×2 {VMC, collocation} × {Adam, SR} on the same ansatz, N=6 ω∈{1,0.1,0.01}, seeded;
  measure energy, var, ESS, and the SR-vs-plain alignment under each measure.
- **Predict:** SR's advantage is small for VMC (confirmed) but *large* for collocation, because the
  ill-conditioned collocation operator is exactly what whitening fixes. This is the Q2×Q3 bridge and
  may overturn the "optimizer is immaterial" verdict *in the regime the thesis uses*.
- **Output:** the SR-advantage-by-paradigm table — the missing Q2 result.

### T2.2 SR at genuine low-ω (hardened cascade)
- **Do:** rebuild the SR-mechanism test at ω=0.01 (N=2 and N=6) with warm-start + annealed lr +
  clipping (the lightweight recipe diverged). Does the modest ω=0.1 edge grow at the true crystal?

### T2.3 Order-of-learning / spectral bias
- **Do:** with `ckpt_every` trajectories, does SR change *which physics is learned first* (cusp vs
  long-range vs collective) and collapse the two-timescale signature? (Q2 × T1.6 spectral bias.)

---

## Thread 3 — Physics at low ω: what the network *becomes* at the crystal

### T3.1 Network-internal Wigner order parameter (C2, unbuilt)
- **Do:** build a latent observable from node/message features that tracks crystallization (Lindemann,
  bond-orientational order); does it show a sharper transition than the physical order parameter?

### T3.2 Do the tangent directions become phonon modes?
- **Do:** at low ω, project the top NTK eigenfunctions onto the crystal's normal modes (breathing +
  shear + rotation + relative vibrations, from a small-oscillation model of the classical Wigner
  molecule). Name the d_eff≈5–6 directions as phonons. (The physical closure of "what are the modes".)

### T3.3 Lattice-assignment computation
- **Do:** test whether at Wigner the message passing becomes a spin→site assignment (attribution is
  72% spin; correlator φ/g-loaded). Decode whether node features predict lattice-site occupancy.

### T3.4 Backflow force-alignment across N and during training
- **Do:** extend Fig J (force alignment) to N=12/20; log it *during* training — does the regime-switch
  emerge or is it immediate?

---

## Thread 4 — Scaling infrastructure (unblock trained-state N≥12)

### T4.1 Fix N≥12 training
- **Blockers found:** from-scratch N=12 diverges; exact Laplacian OOMs at batch 512.
- **Do:** chunk/stochastic (Hutchinson) Laplacian; warm-started cascade recipe for N=12; validate
  against the existing N=12 collocation energies. Then trained-state d_eff, mechanism, and the T1/T3
  probes at N=12/20 (currently only init-scaling exists).

---

## Phased execution (depth-before-breadth, gate after each)

- **Phase M0 — Reproduce & unify the existing mechanism (no training, days).** Re-run the
  DIAGNOSTIC_SUMMARY probes with the analysis package on current checkpoints (they used older
  checkpoints); T1.2 (feature-rank ↔ tangent-d_eff unification), T1.3 (message decode), T1.5
  (bottleneck read-out), T1.7 (N=12/20 extension of attribution/force/3-body). **Gate M0:** the
  mechanism is reproduced, unified with the tangent-kernel picture, and extended in N.
- **Phase M1 — The clean ablations (training, ~week).** T1.1 (backflow-confound-free 2×2), T1.4
  (3-body vs MP-depth), T1.5 ablation (bottleneck removal), T1.6 (kinetic decomposition). **Gate M1:**
  "what MP is worth, and why (kinetic)" is answered honestly and seeded.
- **Phase O — Optimizer×paradigm (training, ~week).** T2.1 (SR on collocation — the headline), T2.2
  (low-ω SR), T2.3 (order-of-learning). **Gate O:** the SR-advantage-by-paradigm map.
- **Phase P — Wigner internals (mixed, ~week).** T3.1–T3.4. **Gate P:** the low-ω "what the network
  becomes" story (order parameter, phonons, lattice assignment, force-switch).
- **Phase S — Scaling (compute, weeks).** T4.1 then trained-state N=12/20 for the load-bearing T1/T3
  results. **Gate S:** the mechanism and its transitions confirmed at scale.

## Discipline
≥3 seeds/inits before any thesis claim; state every ablation explicitly (no hidden shared MP);
one run = one dated folder; save every spectrum/probe .npz; reuse `src/analysis` + the diagnostic
scripts (don't duplicate); journal + decisions after each gate. Pre-register the prediction for every
experiment above and record hits/misses honestly. Reproduce old single-seed diagnostics with seeds
before building on them.

## Decisive thesis figures (the deliverables)
1. **"What MP is worth"** — the confound-free 2×2 energy/kinetic (T1.1) + kinetic decomposition (T1.6).
2. **Scale separation** — bottleneck↔global vs fine↔local (T1.5), the mechanistic "why the graph".
3. **Decoded messages** — edge pseudopotential + node code + 3-body-vs-depth (T1.3/T1.4).
4. **SR-advantage-by-paradigm** — the missing optimizer result (T2.1).
5. **Wigner internals** — order parameter + phonon-named tangent modes + force-switch vs N (T3).
6. **Transition phase-diagram in (N, ω)** — where every mechanism transition sits (T1.7/T3.4).
