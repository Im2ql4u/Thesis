# Phase analysis: N=20, omega=0.5, arch=pinn

- params: 236,336
- **E (unclipped) = 93.936918 +/- 0.011439 Ha**  (exact 93.8752, err +0.066%)
- E (zero-variance extrap, 5 pts) = 93.859893 Ha  (err -0.016%)
- E (clipped est.) = 93.934696 Ha; var(E_L) = 2.5669e-01
- QGT/NTK: eff_rank = 3.42, kappa(S) = 2.905e+06, numerical rank = 511/236336 (alignment on 512 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.085, NTK kappa=2.905e+06

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
