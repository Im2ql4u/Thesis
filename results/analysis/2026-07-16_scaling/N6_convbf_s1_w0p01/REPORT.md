# Phase analysis: N=6, omega=0.01, arch=pinn

- params: 105,136
- **E (unclipped) = 0.694374 +/- 0.000100 Ha**  (exact 0.69036, err +0.581%)
- E (zero-variance extrap, 5 pts) = 0.694640 Ha  (err +0.620%)
- E (clipped est.) = 0.694453 Ha; var(E_L) = 1.0768e-04
- QGT/NTK: eff_rank = 1.14, kappa(S) = 9.970e+11, numerical rank = 474/105136 (alignment on 2048 samples)
- alignment (final): cos(SR)=0.491, cos(plain)=0.050, NTK kappa=9.894e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
