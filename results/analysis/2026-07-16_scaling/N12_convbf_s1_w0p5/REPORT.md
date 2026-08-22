# Phase analysis: N=12, omega=0.5, arch=pinn

- params: 105,136
- **E (unclipped) = 39.185837 +/- 0.004896 Ha**  (exact 39.1596, err +0.067%)
- E (zero-variance extrap, 5 pts) = 39.283753 Ha  (err +0.317%)
- E (clipped est.) = 39.185200 Ha; var(E_L) = 1.2426e-01
- QGT/NTK: eff_rank = 5.35, kappa(S) = 9.676e+06, numerical rank = 1023/105136 (alignment on 1024 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.075, NTK kappa=9.676e+06

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
