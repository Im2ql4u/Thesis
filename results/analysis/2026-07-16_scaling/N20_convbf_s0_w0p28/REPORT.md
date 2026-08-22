# Phase analysis: N=20, omega=0.28, arch=pinn

- params: 105,136
- **E (unclipped) = 61.996084 +/- 0.007717 Ha**  (exact 61.9268, err +0.112%)
- E (zero-variance extrap, 5 pts) = 61.956333 Ha  (err +0.048%)
- E (clipped est.) = 61.996200 Ha; var(E_L) = 1.2762e-01
- QGT/NTK: eff_rank = 3.64, kappa(S) = 2.646e+07, numerical rank = 511/105136 (alignment on 512 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.161, NTK kappa=2.646e+07

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
