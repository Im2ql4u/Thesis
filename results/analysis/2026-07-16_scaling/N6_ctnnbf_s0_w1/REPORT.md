# Phase analysis: N=6, omega=1.0, arch=pinn

- params: 236,336
- **E (unclipped) = 20.150304 +/- 0.001885 Ha**  (exact 20.15932, err -0.045%)
- E (zero-variance extrap, 1 pts) = 20.150304 Ha  (err -0.045%)
- E (clipped est.) = 20.152070 Ha; var(E_L) = 2.4892e-02
- QGT/NTK: eff_rank = 3.94, kappa(S) = 1.668e+08, numerical rank = 2047/236336 (alignment on 2048 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.043, NTK kappa=1.668e+08

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
