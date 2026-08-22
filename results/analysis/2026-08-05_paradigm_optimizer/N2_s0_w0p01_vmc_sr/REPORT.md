# Phase analysis: N=2, omega=0.01, arch=pinn

- params: 236,336
- **E (unclipped) = 0.073838 +/- 0.000001 Ha**  (exact 0.073839, err -0.001%)
- E (zero-variance extrap, 5 pts) = 0.073740 Ha  (err -0.135%)
- E (clipped est.) = 0.073836 Ha; var(E_L) = 2.6045e-09
- QGT/NTK: eff_rank = 1.23, kappa(S) = 9.924e+11, numerical rank = 1574/236336 (alignment on 2048 samples)
- alignment (final): cos(SR)=0.993, cos(plain)=0.075, NTK kappa=9.990e+09

## Exact-truth checks (N=2)
- exact energy = 0.073835 Ha; exact Jastrow cusp dJ/dr|0 = 0.9976
- **|<Psi_net|Psi_exact>|^2 = 0.999998**

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
