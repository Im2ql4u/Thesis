# Phase analysis: N=20, omega=0.1, arch=pinn

- params: 105,136
- **E (unclipped) = 30.024921 +/- 0.005353 Ha**  (exact 29.9779, err +0.157%)
- E (zero-variance extrap, 5 pts) = 29.977287 Ha  (err -0.002%)
- E (clipped est.) = 30.027133 Ha; var(E_L) = 4.0870e-02
- QGT/NTK: eff_rank = 4.05, kappa(S) = 5.922e+07, numerical rank = 511/105136 (alignment on 512 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.172, NTK kappa=5.922e+07

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
