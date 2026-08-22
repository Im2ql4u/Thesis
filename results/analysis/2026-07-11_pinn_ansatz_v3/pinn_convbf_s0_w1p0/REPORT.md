# Phase analysis: N=6, omega=1.0, arch=pinn

- params: 105,136
- **E (unclipped) = 20.171929 +/- 0.002662 Ha**  (exact 20.15932, err +0.063%)
- E (zero-variance extrap, 1 pts) = 20.171929 Ha  (err +0.063%)
- E (clipped est.) = 20.172256 Ha; var(E_L) = 1.0372e-01
- QGT/NTK: eff_rank = 6.51, kappa(S) = 1.638e+09, numerical rank = 2047/105136 (alignment on 2048 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.073, NTK kappa=1.638e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
