# Phase analysis: N=6, omega=0.01, arch=deepset_big

- params: 34,354
- **E (unclipped) = 0.690719 +/- 0.000066 Ha**  (exact 0.69036, err +0.052%)
- E (zero-variance extrap, 5 pts) = 0.689139 Ha  (err -0.177%)
- E (clipped est.) = 0.690770 Ha; var(E_L) = 3.3719e-05
- QGT/NTK: eff_rank = 4.36, kappa(S) = 9.967e+11, numerical rank = 1986/34354 (alignment on 2048 samples)
- alignment (final): cos(SR)=0.890, cos(plain)=0.053, NTK kappa=9.952e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
