# Phase analysis: N=6, omega=1.0, arch=pinn

- params: 236,336
- **E (unclipped) = 21.678386 +/- 0.012759 Ha**  (exact 20.15932, err +7.535%)
- E (zero-variance extrap, 5 pts) = 16.951402 Ha  (err -15.913%)
- E (clipped est.) = 20.170960 Ha; var(E_L) = 2.3186e+00
- QGT/NTK: eff_rank = 7.39, kappa(S) = 4.778e+06, numerical rank = 2047/236336 (alignment on 2048 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.189, NTK kappa=4.778e+06

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
