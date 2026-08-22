# Phase analysis: N=6, omega=0.001, arch=pinn

- params: 105,136
- **E (unclipped) = 0.175621 +/- 0.000144 Ha**  (exact 0.140832, err +24.703%)
- E (zero-variance extrap, 5 pts) = 0.167422 Ha  (err +18.881%)
- E (clipped est.) = 0.175705 Ha; var(E_L) = 1.4754e-04
- QGT/NTK: eff_rank = 1.90, kappa(S) = 9.869e+11, numerical rank = 468/105136 (alignment on 2048 samples)
- alignment (final): cos(SR)=0.869, cos(plain)=0.132, NTK kappa=9.943e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
