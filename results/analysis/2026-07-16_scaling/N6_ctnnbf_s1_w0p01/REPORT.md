# Phase analysis: N=6, omega=0.01, arch=pinn

- params: 236,336
- **E (unclipped) = 0.690182 +/- 0.000043 Ha**  (exact 0.69036, err -0.026%)
- E (zero-variance extrap, 5 pts) = 0.690882 Ha  (err +0.076%)
- E (clipped est.) = 0.690174 Ha; var(E_L) = 1.7186e-05
- QGT/NTK: eff_rank = 8.54, kappa(S) = 2.640e+06, numerical rank = 2047/236336 (alignment on 2048 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.181, NTK kappa=2.640e+06

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
