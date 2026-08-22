# Phase analysis: N=6, omega=0.01, arch=pinn

- params: 236,336
- **E (unclipped) = 0.848344 +/- 0.000591 Ha**  (exact 0.69036, err +22.884%)
- E (zero-variance extrap, 5 pts) = 0.752615 Ha  (err +9.018%)
- E (clipped est.) = 0.825151 Ha; var(E_L) = 3.4002e-03
- QGT/NTK: eff_rank = 1.57, kappa(S) = 9.956e+11, numerical rank = 472/236336 (alignment on 2048 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.245, NTK kappa=9.841e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
