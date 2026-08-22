# Phase analysis: N=20, omega=0.1, arch=pinn

- params: 236,336
- **E (unclipped) = 29.981965 +/- 0.002302 Ha**  (exact 29.9779, err +0.014%)
- E (zero-variance extrap, 5 pts) = 29.965989 Ha  (err -0.040%)
- E (clipped est.) = 29.981695 Ha; var(E_L) = 9.3569e-03
- QGT/NTK: eff_rank = 5.06, kappa(S) = 1.969e+05, numerical rank = 511/236336 (alignment on 512 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.136, NTK kappa=1.969e+05

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
