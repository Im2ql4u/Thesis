# Phase analysis: N=6, omega=0.01, arch=pinn

- params: 67,248
- **E (unclipped) = 0.693820 +/- 0.000088 Ha**  (exact 0.69036, err +0.501%)
- E (zero-variance extrap, 5 pts) = 0.692481 Ha  (err +0.307%)
- E (clipped est.) = 0.693903 Ha; var(E_L) = 8.5698e-05
- QGT/NTK: eff_rank = 1.94, kappa(S) = 9.878e+11, numerical rank = 1094/67248 (alignment on 2048 samples)
- alignment (final): cos(SR)=0.677, cos(plain)=0.029, NTK kappa=9.942e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
