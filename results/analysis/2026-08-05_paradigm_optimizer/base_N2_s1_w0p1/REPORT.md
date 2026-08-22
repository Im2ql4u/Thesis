# Phase analysis: N=2, omega=0.1, arch=pinn

- params: 236,336
- **E (unclipped) = 0.440785 +/- 0.000002 Ha**  (exact 0.44079, err -0.001%)
- E (zero-variance extrap, 6 pts) = 0.440784 Ha  (err -0.001%)
- E (clipped est.) = 0.440811 Ha; var(E_L) = 3.8182e-08
- QGT/NTK: eff_rank = 2.14, kappa(S) = 9.903e+11, numerical rank = 1152/236336 (alignment on 2048 samples)
- alignment (final): cos(SR)=0.969, cos(plain)=0.037, NTK kappa=9.952e+09

## Exact-truth checks (N=2)
- exact energy = 0.440792 Ha; exact Jastrow cusp dJ/dr|0 = 0.9990
- **|<Psi_net|Psi_exact>|^2 = 0.999999**

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
