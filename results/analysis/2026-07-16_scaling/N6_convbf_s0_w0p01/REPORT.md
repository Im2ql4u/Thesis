# Phase analysis: N=6, omega=0.01, arch=pinn

- params: 105,136
- **E (unclipped) = 0.694305 +/- 0.000098 Ha**  (exact 0.69036, err +0.571%)
- E (zero-variance extrap, 5 pts) = 0.693377 Ha  (err +0.437%)
- E (clipped est.) = 0.694358 Ha; var(E_L) = 1.1353e-04
- QGT/NTK: eff_rank = 1.11, kappa(S) = 9.911e+11, numerical rank = 358/105136 (alignment on 2048 samples)
- alignment (final): cos(SR)=0.492, cos(plain)=0.035, NTK kappa=9.980e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
