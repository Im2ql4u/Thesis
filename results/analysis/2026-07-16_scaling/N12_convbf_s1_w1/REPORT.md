# Phase analysis: N=12, omega=1.0, arch=pinn

- params: 105,136
- **E (unclipped) = 65.750745 +/- 0.007981 Ha**  (exact 65.7001, err +0.077%)
- E (zero-variance extrap, 1 pts) = 65.750745 Ha  (err +0.077%)
- E (clipped est.) = 65.752509 Ha; var(E_L) = 3.5911e-01
- QGT/NTK: eff_rank = 4.48, kappa(S) = 1.718e+08, numerical rank = 1023/105136 (alignment on 1024 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.095, NTK kappa=1.718e+08

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
