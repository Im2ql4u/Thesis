# Phase analysis: N=20, omega=0.5, arch=pinn

- params: 105,136
- **E (unclipped) = 93.946067 +/- 0.009940 Ha**  (exact 93.8752, err +0.075%)
- E (zero-variance extrap, 5 pts) = 93.733896 Ha  (err -0.151%)
- E (clipped est.) = 93.946614 Ha; var(E_L) = 2.8497e-01
- QGT/NTK: eff_rank = 3.71, kappa(S) = 2.166e+07, numerical rank = 511/105136 (alignment on 512 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.110, NTK kappa=2.166e+07

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
