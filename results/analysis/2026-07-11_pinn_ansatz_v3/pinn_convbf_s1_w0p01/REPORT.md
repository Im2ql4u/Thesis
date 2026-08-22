# Phase analysis: N=6, omega=0.01, arch=pinn

- params: 105,136
- **E (unclipped) = 0.694396 +/- 0.000128 Ha**  (exact 0.69036, err +0.585%)
- E (zero-variance extrap, 5 pts) = 0.693389 Ha  (err +0.439%)
- E (clipped est.) = 0.694471 Ha; var(E_L) = 1.1335e-04
- QGT/NTK: eff_rank = 1.13, kappa(S) = 9.886e+11, numerical rank = 518/105136 (alignment on 2048 samples)
- alignment (final): cos(SR)=0.527, cos(plain)=0.017, NTK kappa=9.845e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
