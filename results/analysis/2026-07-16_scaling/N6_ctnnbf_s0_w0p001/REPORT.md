# Phase analysis: N=6, omega=0.001, arch=pinn

- params: 236,336
- **E (unclipped) = 0.182983 +/- 0.000115 Ha**  (exact 0.140832, err +29.930%)
- E (zero-variance extrap, 5 pts) = 0.199408 Ha  (err +41.593%)
- E (clipped est.) = 0.183040 Ha; var(E_L) = 1.0589e-04
- QGT/NTK: eff_rank = 2.42, kappa(S) = 9.738e+11, numerical rank = 419/236336 (alignment on 2048 samples)
- alignment (final): cos(SR)=0.987, cos(plain)=0.473, NTK kappa=9.893e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
