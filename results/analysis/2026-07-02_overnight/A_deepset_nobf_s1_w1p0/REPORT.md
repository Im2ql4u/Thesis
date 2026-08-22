# Phase analysis: N=6, omega=1.0, arch=deepset_big

- params: 34,354
- **E (unclipped) = 20.176770 +/- 0.003268 Ha**  (exact 20.15932, err +0.087%)
- E (zero-variance extrap, 5 pts) = 20.147179 Ha  (err -0.060%)
- E (clipped est.) = 20.176648 Ha; var(E_L) = 1.0043e-01
- QGT/NTK: eff_rank = 3.07, kappa(S) = 9.965e+11, numerical rank = 1383/34354 (alignment on 2048 samples)
- alignment (final): cos(SR)=0.785, cos(plain)=0.058, NTK kappa=9.950e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
