# Phase analysis: N=6, omega=0.28, arch=pinn

- params: 236,336
- **E (unclipped) = 7.597697 +/- 0.000547 Ha**  (exact 7.60019, err -0.033%)
- E (zero-variance extrap, 5 pts) = 7.595560 Ha  (err -0.061%)
- E (clipped est.) = 7.597484 Ha; var(E_L) = 1.7245e-03
- QGT/NTK: eff_rank = 5.19, kappa(S) = 6.317e+06, numerical rank = 2047/236336 (alignment on 2048 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.070, NTK kappa=6.317e+06

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
