# Phase analysis: N=20, omega=0.5, arch=pinn

- params: 236,336
- **E (unclipped) = 93.875654 +/- 0.006481 Ha**  (exact 93.8752, err +0.000%)
- E (zero-variance extrap, 5 pts) = 93.762270 Ha  (err -0.120%)
- E (clipped est.) = 93.875622 Ha; var(E_L) = 1.1322e-01
- QGT/NTK: eff_rank = 3.82, kappa(S) = 1.713e+06, numerical rank = 511/236336 (alignment on 512 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.094, NTK kappa=1.713e+06

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
