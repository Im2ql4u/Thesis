# Phase analysis: N=6, omega=0.28, arch=pinn

- params: 105,136
- **E (unclipped) = 7.605044 +/- 0.001318 Ha**  (exact 7.60019, err +0.064%)
- E (zero-variance extrap, 5 pts) = 7.583725 Ha  (err -0.217%)
- E (clipped est.) = 7.605166 Ha; var(E_L) = 1.2650e-02
- QGT/NTK: eff_rank = 15.72, kappa(S) = 6.245e+06, numerical rank = 2047/105136 (alignment on 2048 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.111, NTK kappa=6.245e+06

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
