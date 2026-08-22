# Phase analysis: N=6, omega=1.0, arch=pinn

- params: 236,336
- **E (unclipped) = 20.155536 +/- 0.003525 Ha**  (exact 20.15932, err -0.019%)
- E (zero-variance extrap, 5 pts) = 20.124700 Ha  (err -0.172%)
- E (clipped est.) = 20.159442 Ha; var(E_L) = 2.2841e-02
- QGT/NTK: eff_rank = 3.94, kappa(S) = 8.874e+05, numerical rank = 511/236336 (alignment on 512 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.092, NTK kappa=8.874e+05

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
