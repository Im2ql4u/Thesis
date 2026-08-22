# Phase analysis: N=6, omega=0.1, arch=pinn

- params: 236,336
- **E (unclipped) = 3.553219 +/- 0.000538 Ha**  (exact 3.55385, err -0.018%)
- E (zero-variance extrap, 5 pts) = 3.549749 Ha  (err -0.115%)
- E (clipped est.) = 3.553063 Ha; var(E_L) = 6.3277e-04
- QGT/NTK: eff_rank = 7.11, kappa(S) = 1.151e+05, numerical rank = 511/236336 (alignment on 512 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.114, NTK kappa=1.151e+05

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
