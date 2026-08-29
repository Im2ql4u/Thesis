# Phase analysis: N=12, omega=1.0, arch=deepset_big

- params: 216,757
- **E (unclipped) = 65.777398 +/- 0.046195 Ha**  (exact 65.7001, err +0.118%)
- E (zero-variance extrap, 6 pts) = 65.428626 Ha  (err -0.413%)
- E (clipped est.) = 65.777398 Ha; var(E_L) = 6.7862e-01
- QGT/NTK: eff_rank = 4.48, kappa(S) = 6.417e+05, numerical rank = 127/216757 (alignment on 128 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.283, NTK kappa=6.417e+05

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
