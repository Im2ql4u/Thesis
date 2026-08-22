# Phase analysis: N=12, omega=0.1, arch=pinn

- params: 236,336
- **E (unclipped) = 12.267090 +/- 0.000986 Ha**  (exact 12.26984, err -0.022%)
- E (zero-variance extrap, 5 pts) = 12.262062 Ha  (err -0.063%)
- E (clipped est.) = 12.267083 Ha; var(E_L) = 3.0471e-03
- QGT/NTK: eff_rank = 5.26, kappa(S) = 2.153e+05, numerical rank = 1023/236336 (alignment on 1024 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.067, NTK kappa=2.153e+05

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
