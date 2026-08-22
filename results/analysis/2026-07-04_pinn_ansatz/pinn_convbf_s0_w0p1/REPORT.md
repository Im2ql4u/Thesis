# Phase analysis: N=6, omega=0.1, arch=pinn

- params: 67,248
- **E (unclipped) = 3.570866 +/- 0.000644 Ha**  (exact 3.55385, err +0.479%)
- E (zero-variance extrap, 5 pts) = 3.568356 Ha  (err +0.408%)
- E (clipped est.) = 3.571265 Ha; var(E_L) = 5.5423e-03
- QGT/NTK: eff_rank = 1.94, kappa(S) = 9.908e+11, numerical rank = 1098/67248 (alignment on 2048 samples)
- alignment (final): cos(SR)=0.640, cos(plain)=0.027, NTK kappa=9.873e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
