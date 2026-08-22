# Phase analysis: N=2, omega=1.0, arch=pinn

- params: 236,336
- **E (unclipped) = 2.999948 +/- 0.000042 Ha**  (exact 3.0, err -0.002%)
- E (zero-variance extrap, 5 pts) = 2.999915 Ha  (err -0.003%)
- E (clipped est.) = 3.000084 Ha; var(E_L) = 1.3162e-05
- QGT/NTK: eff_rank = 3.71, kappa(S) = 9.869e+11, numerical rank = 991/236336 (alignment on 2048 samples)
- alignment (final): cos(SR)=0.996, cos(plain)=0.339, NTK kappa=9.990e+09

## Exact-truth checks (N=2)
- exact energy = 3.000000 Ha; exact Jastrow cusp dJ/dr|0 = 0.9991
- **|<Psi_net|Psi_exact>|^2 = 0.999999**

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
