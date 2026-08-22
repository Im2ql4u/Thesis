# Phase analysis: N=6, omega=0.01, arch=pinn

- params: 236,336
- **E (unclipped) = 0.691992 +/- 0.000160 Ha**  (exact 0.69036, err +0.236%)
- E (zero-variance extrap, 5 pts) = 0.693261 Ha  (err +0.420%)
- E (clipped est.) = 0.692171 Ha; var(E_L) = 5.8904e-05
- QGT/NTK: eff_rank = 6.14, kappa(S) = 9.998e+04, numerical rank = 511/236336 (alignment on 512 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.224, NTK kappa=9.998e+04

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
