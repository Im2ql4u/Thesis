# Phase analysis: N=2, omega=1.0, arch=ctnn_vcycle

- params: 9,842
- **E = 2.995859 +/- 0.000088 Ha**  (ref 3.0, err -0.138%, -47.3 sigma)
- var(E_L) = 2.6761e-05
- QGT/NTK: eff_rank = 1.16, kappa(S) = 9.477e+11, numerical rank = 175/9842
- alignment (final): cos(SR)=1.000, cos(plain)=0.131, NTK kappa=9.533e+09

## Exact-truth checks (N=2)
- exact energy = 3.000000 Ha; exact Jastrow cusp dJ/dr|0 = 0.9991
- **|<Psi_net|Psi_exact>|^2 = 0.999984**

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening. Arrays: diagnostics.npz. Checkpoint: checkpoint.pt.
