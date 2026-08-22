# Phase analysis: N=6, omega=0.001, arch=pinn

- params: 236,336
- **E (unclipped) = 0.171834 +/- 0.000067 Ha**  (exact 0.140832, err +22.013%)
- E (zero-variance extrap, 5 pts) = 0.177411 Ha  (err +25.974%)
- E (clipped est.) = 0.170137 Ha; var(E_L) = 3.7165e-05
- QGT/NTK: eff_rank = 2.67, kappa(S) = 4.918e+09, numerical rank = 2047/236336 (alignment on 2048 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.312, NTK kappa=4.918e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
