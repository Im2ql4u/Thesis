# Phase analysis: N=6, omega=0.1, arch=deepset_big

- params: 34,354
- **E (unclipped) = 3.559235 +/- 0.000343 Ha**  (exact 3.55385, err +0.152%)
- E (zero-variance extrap, 5 pts) = 3.554243 Ha  (err +0.011%)
- E (clipped est.) = 3.559431 Ha; var(E_L) = 1.8788e-03
- QGT/NTK: eff_rank = 3.70, kappa(S) = 9.980e+11, numerical rank = 1452/34354 (alignment on 2048 samples)
- alignment (final): cos(SR)=0.736, cos(plain)=0.047, NTK kappa=1.000e+10

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
