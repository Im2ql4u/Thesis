# Phase analysis: N=6, omega=0.1, arch=deepset_big

- params: 47,669
- **E (unclipped) = 3.558727 +/- 0.000508 Ha**  (exact 3.55385, err +0.137%)
- E (zero-variance extrap, 5 pts) = 3.548268 Ha  (err -0.157%)
- E (clipped est.) = 3.558942 Ha; var(E_L) = 1.8633e-03
- QGT/NTK: eff_rank = 3.63, kappa(S) = 9.941e+11, numerical rank = 1330/47669 (alignment on 2048 samples)
- alignment (final): cos(SR)=0.695, cos(plain)=0.021, NTK kappa=9.926e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
