# Phase analysis: N=6, omega=0.1, arch=deepset_big

- params: 34,354
- **E (unclipped) = 3.559536 +/- 0.000540 Ha**  (exact 3.55385, err +0.160%)
- E (zero-variance extrap, 5 pts) = 3.548972 Ha  (err -0.137%)
- E (clipped est.) = 3.559464 Ha; var(E_L) = 1.8798e-03
- QGT/NTK: eff_rank = 3.85, kappa(S) = 9.998e+11, numerical rank = 1577/34354 (alignment on 2048 samples)
- alignment (final): cos(SR)=0.794, cos(plain)=0.037, NTK kappa=9.975e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
