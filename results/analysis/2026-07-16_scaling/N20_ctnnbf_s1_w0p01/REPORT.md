# Phase analysis: N=20, omega=0.01, arch=pinn

- params: 236,336
- **E (unclipped) = 6.152454 +/- 0.000717 Ha**  (exact 6.14645, err +0.098%)
- E (zero-variance extrap, 5 pts) = 6.099152 Ha  (err -0.770%)
- E (clipped est.) = 6.152502 Ha; var(E_L) = 6.8558e-04
- QGT/NTK: eff_rank = 5.99, kappa(S) = 4.990e+04, numerical rank = 511/236336 (alignment on 512 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.143, NTK kappa=4.990e+04

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
