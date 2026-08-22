# Phase analysis: N=6, omega=0.28, arch=pinn

- params: 105,136
- **E (unclipped) = 7.605507 +/- 0.001267 Ha**  (exact 7.60019, err +0.070%)
- E (zero-variance extrap, 5 pts) = 7.590545 Ha  (err -0.127%)
- E (clipped est.) = 7.604416 Ha; var(E_L) = 1.1810e-02
- QGT/NTK: eff_rank = 15.19, kappa(S) = 8.629e+06, numerical rank = 2047/105136 (alignment on 2048 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.089, NTK kappa=8.629e+06

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
