# Phase analysis: N=6, omega=0.01, arch=pinn

- params: 236,336
- **E (unclipped) = 0.691907 +/- 0.000171 Ha**  (exact 0.69036, err +0.224%)
- E (zero-variance extrap, 5 pts) = 0.692282 Ha  (err +0.278%)
- E (clipped est.) = 0.692047 Ha; var(E_L) = 6.4500e-05
- QGT/NTK: eff_rank = 8.76, kappa(S) = 7.681e+04, numerical rank = 511/236336 (alignment on 512 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.166, NTK kappa=7.681e+04

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
