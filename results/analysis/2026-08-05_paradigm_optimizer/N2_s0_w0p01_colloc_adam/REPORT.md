# Phase analysis: N=2, omega=0.01, arch=pinn

- params: 236,336
- **E (unclipped) = 0.073815 +/- 0.000010 Ha**  (exact 0.073839, err -0.033%)
- E (zero-variance extrap, 5 pts) = 0.073860 Ha  (err +0.028%)
- E (clipped est.) = 0.073830 Ha; var(E_L) = 8.1032e-07
- QGT/NTK: eff_rank = 1.09, kappa(S) = 9.963e+11, numerical rank = 653/236336 (alignment on 2048 samples)
- alignment (final): cos(SR)=0.999, cos(plain)=0.361, NTK kappa=9.827e+09

## Exact-truth checks (N=2)
- exact energy = 0.073835 Ha; exact Jastrow cusp dJ/dr|0 = 0.9976
- **|<Psi_net|Psi_exact>|^2 = 0.999333**

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
