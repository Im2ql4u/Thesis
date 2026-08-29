# Phase analysis: N=2, omega=0.5, arch=pinn

- params: 236,336
- **E (unclipped) = 1.659739 +/- 0.000013 Ha**  (exact 1.65977, err -0.002%)
- E (zero-variance extrap, 6 pts) = 1.659726 Ha  (err -0.003%)
- E (clipped est.) = 1.659763 Ha; var(E_L) = 1.3315e-06
- QGT/NTK: eff_rank = 3.13, kappa(S) = 9.934e+11, numerical rank = 1032/236336 (alignment on 2048 samples)
- alignment (final): cos(SR)=0.993, cos(plain)=0.173, NTK kappa=9.888e+09

## Exact-truth checks (N=2)
- exact energy = 1.659772 Ha; exact Jastrow cusp dJ/dr|0 = 0.9993
- **|<Psi_net|Psi_exact>|^2 = 1.000000**

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
