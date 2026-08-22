# Phase analysis: N=6, omega=1.0, arch=pinn

- params: 236,336
- **E (unclipped) = 20.467835 +/- 0.011846 Ha**  (exact 20.15932, err +1.530%)
- E (zero-variance extrap, 5 pts) = 20.153867 Ha  (err -0.027%)
- E (clipped est.) = 20.262601 Ha; var(E_L) = 1.7975e+00
- QGT/NTK: eff_rank = 4.21, kappa(S) = 2.563e+07, numerical rank = 2047/236336 (alignment on 2048 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.131, NTK kappa=2.563e+07

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
