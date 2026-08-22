# Phase analysis: N=2, omega=1.0, arch=pinn

- params: 236,336
- **E (unclipped) = 2.999997 +/- 0.000038 Ha**  (exact 3.0, err -0.000%)
- E (zero-variance extrap, 6 pts) = 2.999991 Ha  (err -0.000%)
- E (clipped est.) = 3.000085 Ha; var(E_L) = 7.7034e-06
- QGT/NTK: eff_rank = 4.29, kappa(S) = 9.960e+11, numerical rank = 832/236336 (alignment on 2048 samples)
- alignment (final): cos(SR)=0.996, cos(plain)=0.050, NTK kappa=9.958e+09

## Exact-truth checks (N=2)
- exact energy = 3.000000 Ha; exact Jastrow cusp dJ/dr|0 = 0.9991
- **|<Psi_net|Psi_exact>|^2 = 1.000000**

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
