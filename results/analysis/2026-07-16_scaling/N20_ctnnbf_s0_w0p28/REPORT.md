# Phase analysis: N=20, omega=0.28, arch=pinn

- params: 236,336
- **E (unclipped) = 61.922939 +/- 0.005025 Ha**  (exact 61.9268, err -0.006%)
- E (zero-variance extrap, 5 pts) = 61.820265 Ha  (err -0.172%)
- E (clipped est.) = 61.922371 Ha; var(E_L) = 4.5866e-02
- QGT/NTK: eff_rank = 3.95, kappa(S) = 6.983e+05, numerical rank = 511/236336 (alignment on 512 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.118, NTK kappa=6.983e+05

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
