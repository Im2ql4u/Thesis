# Over-smoothing, non-separability, and why message-passing is more trainable

**Created:** 2026-06-15
**Status:** Active hypothesis + measurement program. Extends the kernel-analysis program.

---

## The hypothesis

Switching the correlator from a pairwise FFNN/DeepSet to a message-passing CTNN gave a large,
real improvement (historically: broke the ~0.1% wall). The mechanism, grounded in the thesis's own
appendix theory (De Ryck conditioning, the `k^4` law, Barron/spectral) and GNN over-smoothing theory:

1. **Spectral conditioning.** Training speed ~ `1/kappa(A)`, `A=J*J`, and in the Laplacian-dominant
   regime `kappa(A) ~ (k_max/k_min)^4` (thesis appendix). Conditioning is set by the highest spatial
   frequency the tangent functions `q_j = d_theta logPsi` must carry.

2. **Message-passing is a low-pass filter in configuration space.** Each round aggregates
   (sum/mean) over the permutation-symmetric neighbour set -> suppresses high `|k|` (the GNN
   "over-smoothing" property). The same many-body correlation is represented with **lower `k_max`**
   than a sum-of-raw-pairs -> by the `k^4` law, dramatically smaller `kappa(A)`, and (zero-variance
   principle) lower `var(E_L)`. We measured a 66-139x `var(E_L)` explosion when ablating messages.

3. **Non-separability (the refined version of "stripped of individuality").** Indistinguishability
   (permutation symmetry) is shared by DeepSet. What is *unique* to message-passing is that nodes
   *exchange information and update each other*, so each per-particle representation becomes a
   function of **all** the others -- **non-separable / mutually entangled**. A DeepSet computes each
   `phi(x_i)` independently (separable, mean-field-like). The true correlated ground state is, by
   definition, non-separable. So message-passing's representation is non-separable exactly when the
   wavefunction must be. **Over-smoothing is the extreme of this consensus = maximal non-separability.**
   Controlled, not total: full consensus would give a constant (position-independent) wavefunction;
   the V-cycle is literally controlled coarse-graining (smooth to bottleneck, restore detail via skips).

4. **Why stronger toward Wigner.** The true state is *most non-separable / most collective* at strong
   correlation (the Wigner molecule is set by global lattice order). A consensus/non-separable
   representation matches best there; a separable (mean-field/DeepSet) one fails most. This unifies
   three otherwise-puzzling measured trends toward low omega: many-body fraction -> >200%,
   local-geometry decode -> collapses, intrinsic dimension -> drops (8.3->5.0). All three = the
   network increasingly representing a low-dimensional collective consensus state.

5. **Conditioning at weak coupling, expressivity at strong coupling.** At weak/intermediate omega
   both ansatze can reach the variational floor with a strong optimiser (SR); CTNN just gets there
   easier (better conditioning substitutes for optimiser strength -- explains "CTNN broke the wall,
   FFNN+SR catches up"). At Wigner the messages are genuinely needed (expressivity).

**One-line theory to write up:** message-passing is a configuration-space low-pass filter that lowers
`k_max(logPsi)`, collapsing `kappa(A)` and `var(E_L)` (conditioning), while its non-separable
fixed-point representation is a self-consistent collective state matching the correlated ground state
(physics) -- benefit shifting conditioning -> expressivity as non-separability of the true state grows.

---

## The 7 measurements

1. **Over-smoothing / Dirichlet energy** of node features across V-cycle stages (embed -> down ->
   bottleneck): does it converge toward consensus, and by how much. Metric: Dirichlet energy
   `D(H)=mean_{i,j}||h_i-h_j||^2` (normalised) per stage; node-feature variance at the bottleneck.
2. **Non-separability** (mutual dependence): cross/self sensitivity
   `mean_{i!=j}||d h_i/d x_j|| / mean_i ||d h_i/d x_i||`, full CTNN vs message-ablated (should -> 0).
3. **Effective `k_max` / spectral content** of `logPsi`: 1D coordinate scan -> FFT -> spectral
   centroid & `k95`; CTNN vs ablated vs DeepSet (matched params). Lower = smoother.
4. **`kappa(S)` / Fisher conditioning** at matched parameters: CTNN vs DeepSet.
5. **Collective-consensus order parameter vs omega**: bottleneck node-feature variance / intrinsic
   dim of the consensus state; predict more consensus toward Wigner.
6. **Training-speed curves** {CTNN, DeepSet} x {Adam, SR}, matched params: energy/var per epoch,
   epochs-to-threshold ("how much faster").
7. **Self-consistency / mean-field alignment**: decode node features to the instantaneous local field
   vs the configuration-mean (collective) field; does the consensus resemble a self-consistent field
   more toward Wigner. (Reference: HF one-body field.)

**Existing theory to use:** GNN over-smoothing as low-pass graph filtering (Oono-Suzuki 2020,
Cai-Wang, Di Giovanni et al.); De Ryck PINN conditioning + the `k^4` law (thesis appendix);
Barron/spectral bias (Rahaman et al.); QGT/natural gradient (Park-Kastoryano, Sorella); mean-field/RPA
self-consistency.

**Code:** `src/analysis/collectivity.py` (1,2,3,5,7), `diagnostics.kernel_spectrum` (4),
matched-param 2x2 training with convergence logging (6). Driver: `scripts/run_collectivity.py`.
