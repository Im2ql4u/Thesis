# Phase analysis: N=6, omega=0.1, arch=pinn

- params: 236,336
- **E (unclipped) = 3.552389 +/- 0.000224 Ha**  (exact 3.55385, err -0.041%)
- E (zero-variance extrap, 5 pts) = 3.551585 Ha  (err -0.064%)
- E (clipped est.) = 3.552208 Ha; var(E_L) = 3.6136e-04
- QGT/NTK: eff_rank = 6.93, kappa(S) = 2.067e+06, numerical rank = 2047/236336 (alignment on 2048 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.059, NTK kappa=2.067e+06

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
