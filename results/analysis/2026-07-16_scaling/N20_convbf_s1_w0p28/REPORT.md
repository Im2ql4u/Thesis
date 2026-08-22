# Phase analysis: N=20, omega=0.28, arch=pinn

- params: 105,136
- **E (unclipped) = 61.990337 +/- 0.006846 Ha**  (exact 61.9268, err +0.103%)
- E (zero-variance extrap, 5 pts) = 61.836274 Ha  (err -0.146%)
- E (clipped est.) = 61.991356 Ha; var(E_L) = 1.3798e-01
- QGT/NTK: eff_rank = 4.25, kappa(S) = 9.812e+06, numerical rank = 511/105136 (alignment on 512 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.116, NTK kappa=9.812e+06

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
