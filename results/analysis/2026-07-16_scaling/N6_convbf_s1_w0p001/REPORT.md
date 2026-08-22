# Phase analysis: N=6, omega=0.001, arch=pinn

- params: 105,136
- **E (unclipped) = 0.182660 +/- 0.000150 Ha**  (exact 0.140832, err +29.700%)
- E (zero-variance extrap, 5 pts) = 0.213813 Ha  (err +51.822%)
- E (clipped est.) = 0.182682 Ha; var(E_L) = 1.7880e-04
- QGT/NTK: eff_rank = 2.47, kappa(S) = 9.910e+11, numerical rank = 565/105136 (alignment on 2048 samples)
- alignment (final): cos(SR)=0.977, cos(plain)=0.262, NTK kappa=9.577e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
