# Phase analysis: N=6, omega=0.1, arch=pinn

- params: 100,080
- **E (unclipped) = 3.574226 +/- 0.000907 Ha**  (exact 3.55385, err +0.573%)
- E (zero-variance extrap, 5 pts) = 3.566690 Ha  (err +0.361%)
- E (clipped est.) = 3.574244 Ha; var(E_L) = 5.8321e-03
- QGT/NTK: eff_rank = 1.69, kappa(S) = 9.872e+11, numerical rank = 792/100080 (alignment on 2048 samples)
- alignment (final): cos(SR)=0.668, cos(plain)=0.037, NTK kappa=9.868e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
