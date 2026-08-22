# Phase analysis: N=20, omega=0.01, arch=pinn

- params: 105,136
- **E (unclipped) = 6.187695 +/- 0.000757 Ha**  (exact 6.14645, err +0.671%)
- E (zero-variance extrap, 5 pts) = 6.236232 Ha  (err +1.461%)
- E (clipped est.) = 6.187716 Ha; var(E_L) = 1.5541e-03
- QGT/NTK: eff_rank = 1.25, kappa(S) = 9.832e+11, numerical rank = 374/105136 (alignment on 512 samples)
- alignment (final): cos(SR)=0.857, cos(plain)=0.086, NTK kappa=9.373e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
