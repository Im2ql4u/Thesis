# Phase analysis: N=6, omega=0.01, arch=pinn

- params: 236,336
- **E (unclipped) = 0.691471 +/- 0.000071 Ha**  (exact 0.69036, err +0.161%)
- E (zero-variance extrap, 5 pts) = 0.690439 Ha  (err +0.011%)
- E (clipped est.) = 0.691531 Ha; var(E_L) = 5.0183e-05
- QGT/NTK: eff_rank = 6.29, kappa(S) = 4.562e+06, numerical rank = 2047/236336 (alignment on 2048 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.145, NTK kappa=4.562e+06

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
