# Phase analysis: N=6, omega=0.1, arch=pinn

- params: 236,336
- **E (unclipped) = 3.552969 +/- 0.000209 Ha**  (exact 3.55385, err -0.025%)
- E (zero-variance extrap, 5 pts) = 3.545073 Ha  (err -0.247%)
- E (clipped est.) = 3.552846 Ha; var(E_L) = 6.6610e-04
- QGT/NTK: eff_rank = 6.50, kappa(S) = 3.867e+06, numerical rank = 2047/236336 (alignment on 2048 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.094, NTK kappa=3.867e+06

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
