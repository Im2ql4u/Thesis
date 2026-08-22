# Phase analysis: N=12, omega=0.1, arch=pinn

- params: 236,336
- **E (unclipped) = 12.268350 +/- 0.000763 Ha**  (exact 12.26984, err -0.012%)
- E (zero-variance extrap, 5 pts) = 12.262399 Ha  (err -0.061%)
- E (clipped est.) = 12.268525 Ha; var(E_L) = 3.0245e-03
- QGT/NTK: eff_rank = 4.82, kappa(S) = 2.727e+05, numerical rank = 1023/236336 (alignment on 1024 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.082, NTK kappa=2.727e+05

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
