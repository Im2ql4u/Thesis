# Phase analysis: N=20, omega=1.0, arch=pinn

- params: 105,136
- **E (unclipped) = 156.023405 +/- 0.025934 Ha**  (exact 155.8822, err +0.091%)
- E (zero-variance extrap, 1 pts) = 156.023405 Ha  (err +0.091%)
- E (clipped est.) = 156.023424 Ha; var(E_L) = 1.1405e+00
- QGT/NTK: eff_rank = 3.22, kappa(S) = 9.970e+07, numerical rank = 511/105136 (alignment on 512 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.092, NTK kappa=9.970e+07

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
