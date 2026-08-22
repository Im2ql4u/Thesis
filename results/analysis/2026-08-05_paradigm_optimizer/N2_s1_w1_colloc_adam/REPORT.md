# Phase analysis: N=2, omega=1.0, arch=pinn

- params: 236,336
- **E (unclipped) = 3.001072 +/- 0.000715 Ha**  (exact 3.0, err +0.036%)
- E (zero-variance extrap, 5 pts) = 2.997926 Ha  (err -0.069%)
- E (clipped est.) = 2.996173 Ha; var(E_L) = 3.5124e-03
- QGT/NTK: eff_rank = 3.62, kappa(S) = 9.904e+11, numerical rank = 1145/236336 (alignment on 2048 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.303, NTK kappa=9.920e+09

## Exact-truth checks (N=2)
- exact energy = 3.000000 Ha; exact Jastrow cusp dJ/dr|0 = 0.9991
- **|<Psi_net|Psi_exact>|^2 = 0.999650**

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
