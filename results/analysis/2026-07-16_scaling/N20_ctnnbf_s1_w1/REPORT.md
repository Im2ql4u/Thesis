# Phase analysis: N=20, omega=1.0, arch=pinn

- params: 236,336
- **E (unclipped) = 155.934506 +/- 0.022627 Ha**  (exact 155.8822, err +0.034%)
- E (zero-variance extrap, 1 pts) = 155.934506 Ha  (err +0.034%)
- E (clipped est.) = 155.934506 Ha; var(E_L) = 8.9212e-01
- QGT/NTK: eff_rank = 3.56, kappa(S) = 4.026e+06, numerical rank = 511/236336 (alignment on 512 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.074, NTK kappa=4.026e+06

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
