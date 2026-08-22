# Phase analysis: N=6, omega=0.1, arch=pinn

- params: 236,336
- **E (unclipped) = 3.552905 +/- 0.000260 Ha**  (exact 3.55385, err -0.027%)
- E (zero-variance extrap, 5 pts) = 3.551375 Ha  (err -0.070%)
- E (clipped est.) = 3.552755 Ha; var(E_L) = 4.2149e-04
- QGT/NTK: eff_rank = 6.49, kappa(S) = 2.111e+06, numerical rank = 2047/236336 (alignment on 2048 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.058, NTK kappa=2.111e+06

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
