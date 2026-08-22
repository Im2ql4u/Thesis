# Phase analysis: N=6, omega=0.01, arch=pinn

- params: 100,080
- **E (unclipped) = 0.694077 +/- 0.000079 Ha**  (exact 0.69036, err +0.538%)
- E (zero-variance extrap, 5 pts) = 0.687652 Ha  (err -0.392%)
- E (clipped est.) = 0.694243 Ha; var(E_L) = 1.1550e-04
- QGT/NTK: eff_rank = 1.10, kappa(S) = 9.968e+11, numerical rank = 509/100080 (alignment on 2048 samples)
- alignment (final): cos(SR)=0.548, cos(plain)=0.023, NTK kappa=9.471e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
