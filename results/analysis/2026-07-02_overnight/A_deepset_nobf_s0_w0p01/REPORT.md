# Phase analysis: N=6, omega=0.01, arch=deepset_big

- params: 34,354
- **E (unclipped) = 0.690767 +/- 0.000077 Ha**  (exact 0.69036, err +0.059%)
- E (zero-variance extrap, 5 pts) = 0.689111 Ha  (err -0.181%)
- E (clipped est.) = 0.690771 Ha; var(E_L) = 3.4921e-05
- QGT/NTK: eff_rank = 4.13, kappa(S) = 9.946e+11, numerical rank = 1834/34354 (alignment on 2048 samples)
- alignment (final): cos(SR)=0.843, cos(plain)=0.078, NTK kappa=9.964e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
