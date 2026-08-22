# Phase analysis: N=12, omega=0.5, arch=pinn

- params: 236,336
- **E (unclipped) = 39.156065 +/- 0.003309 Ha**  (exact 39.1596, err -0.009%)
- E (zero-variance extrap, 5 pts) = 39.148984 Ha  (err -0.027%)
- E (clipped est.) = 39.155995 Ha; var(E_L) = 3.9011e-02
- QGT/NTK: eff_rank = 4.00, kappa(S) = 2.011e+06, numerical rank = 1023/236336 (alignment on 1024 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.095, NTK kappa=2.011e+06

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
