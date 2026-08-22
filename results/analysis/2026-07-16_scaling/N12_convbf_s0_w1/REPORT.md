# Phase analysis: N=12, omega=1.0, arch=pinn

- params: 105,136
- **E (unclipped) = 65.737842 +/- 0.010332 Ha**  (exact 65.7001, err +0.057%)
- E (zero-variance extrap, 1 pts) = 65.737842 Ha  (err +0.057%)
- E (clipped est.) = 65.735603 Ha; var(E_L) = 3.7277e-01
- QGT/NTK: eff_rank = 3.90, kappa(S) = 1.937e+08, numerical rank = 1023/105136 (alignment on 1024 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.056, NTK kappa=1.937e+08

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
