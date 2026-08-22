# Phase analysis: N=12, omega=0.01, arch=pinn

- params: 105,136
- **E (unclipped) = 2.486678 +/- 0.000360 Ha**  (exact 2.47363, err +0.527%)
- E (zero-variance extrap, 5 pts) = 2.505102 Ha  (err +1.272%)
- E (clipped est.) = 2.486695 Ha; var(E_L) = 4.2524e-04
- QGT/NTK: eff_rank = 1.07, kappa(S) = 9.834e+11, numerical rank = 377/105136 (alignment on 1024 samples)
- alignment (final): cos(SR)=0.531, cos(plain)=0.041, NTK kappa=9.579e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
