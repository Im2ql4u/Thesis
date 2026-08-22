# Phase analysis: N=6, omega=0.01, arch=pinn

- params: 67,248
- **E (unclipped) = 0.693731 +/- 0.000107 Ha**  (exact 0.69036, err +0.488%)
- E (zero-variance extrap, 5 pts) = 0.691195 Ha  (err +0.121%)
- E (clipped est.) = 0.693845 Ha; var(E_L) = 8.7705e-05
- QGT/NTK: eff_rank = 2.55, kappa(S) = 9.938e+11, numerical rank = 1253/67248 (alignment on 2048 samples)
- alignment (final): cos(SR)=0.724, cos(plain)=0.043, NTK kappa=9.920e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
