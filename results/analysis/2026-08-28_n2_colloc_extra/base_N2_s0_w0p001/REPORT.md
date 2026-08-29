# Phase analysis: N=2, omega=0.001, arch=pinn

- params: 236,336
- **E (unclipped) = 0.013837 +/- 0.000004 Ha**  (exact 0.013778, err +0.425%)
- E (zero-variance extrap, 6 pts) = 0.013908 Ha  (err +0.946%)
- E (clipped est.) = 0.013791 Ha; var(E_L) = 5.6466e-08
- QGT/NTK: eff_rank = 1.83, kappa(S) = 9.820e+11, numerical rank = 2002/236336 (alignment on 2048 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.263, NTK kappa=9.997e+09

## Exact-truth checks (N=2)
- exact energy = 0.013768 Ha; exact Jastrow cusp dJ/dr|0 = 0.9929
- **|<Psi_net|Psi_exact>|^2 = 0.989715**

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
