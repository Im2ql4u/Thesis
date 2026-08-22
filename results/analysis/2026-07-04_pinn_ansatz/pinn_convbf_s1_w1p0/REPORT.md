# Phase analysis: N=6, omega=1.0, arch=pinn

- params: 67,248
- **E (unclipped) = 20.197567 +/- 0.004693 Ha**  (exact 20.15932, err +0.190%)
- E (zero-variance extrap, 5 pts) = 20.212138 Ha  (err +0.262%)
- E (clipped est.) = 20.198312 Ha; var(E_L) = 1.8573e-01
- QGT/NTK: eff_rank = 1.66, kappa(S) = 9.997e+11, numerical rank = 1189/67248 (alignment on 2048 samples)
- alignment (final): cos(SR)=0.685, cos(plain)=0.044, NTK kappa=9.929e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
