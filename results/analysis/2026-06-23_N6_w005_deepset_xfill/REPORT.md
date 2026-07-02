# Phase analysis: N=6, omega=0.05, arch=deepset_big

- params: 47,669
- **E (unclipped) = 2.156817 +/- 0.000204 Ha**
- E (zero-variance extrap, 6 pts) = 2.156490 Ha
- E (clipped est.) = 2.156797 Ha; var(E_L) = 5.3084e-04
- QGT/NTK: eff_rank = 4.24, kappa(S) = 9.893e+11, numerical rank = 1920/47669 (alignment on 2048 samples)
- alignment (final): cos(SR)=0.850, cos(plain)=0.074, NTK kappa=9.997e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
