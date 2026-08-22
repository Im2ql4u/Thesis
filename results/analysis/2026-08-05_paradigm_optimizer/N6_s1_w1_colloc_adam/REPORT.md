# Phase analysis: N=6, omega=1.0, arch=pinn

- params: 236,336
- **E (unclipped) = 21.620731 +/- 0.018687 Ha**  (exact 20.15932, err +7.249%)
- E (zero-variance extrap, 5 pts) = 19.828025 Ha  (err -1.643%)
- E (clipped est.) = 20.287044 Ha; var(E_L) = 2.8374e+00
- QGT/NTK: eff_rank = 8.92, kappa(S) = 3.502e+06, numerical rank = 2047/236336 (alignment on 2048 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.168, NTK kappa=3.502e+06

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
