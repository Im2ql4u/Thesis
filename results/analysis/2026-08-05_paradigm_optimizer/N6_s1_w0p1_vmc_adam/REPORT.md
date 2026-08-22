# Phase analysis: N=6, omega=0.1, arch=pinn

- params: 236,336
- **E (unclipped) = 3.553430 +/- 0.000297 Ha**  (exact 3.55385, err -0.012%)
- E (zero-variance extrap, 5 pts) = 3.551647 Ha  (err -0.062%)
- E (clipped est.) = 3.553082 Ha; var(E_L) = 6.6879e-04
- QGT/NTK: eff_rank = 6.30, kappa(S) = 5.659e+06, numerical rank = 2047/236336 (alignment on 2048 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.097, NTK kappa=5.659e+06

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
