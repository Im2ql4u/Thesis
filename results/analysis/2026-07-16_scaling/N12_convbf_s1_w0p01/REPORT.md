# Phase analysis: N=12, omega=0.01, arch=pinn

- params: 105,136
- **E (unclipped) = 2.487152 +/- 0.000360 Ha**  (exact 2.47363, err +0.547%)
- E (zero-variance extrap, 5 pts) = 2.503838 Ha  (err +1.221%)
- E (clipped est.) = 2.487198 Ha; var(E_L) = 4.1132e-04
- QGT/NTK: eff_rank = 1.06, kappa(S) = 9.964e+11, numerical rank = 421/105136 (alignment on 1024 samples)
- alignment (final): cos(SR)=0.523, cos(plain)=0.051, NTK kappa=9.786e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
