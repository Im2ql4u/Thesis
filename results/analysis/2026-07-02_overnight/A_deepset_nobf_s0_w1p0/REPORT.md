# Phase analysis: N=6, omega=1.0, arch=deepset_big

- params: 34,354
- **E (unclipped) = 20.175699 +/- 0.002995 Ha**  (exact 20.15932, err +0.081%)
- E (zero-variance extrap, 5 pts) = 20.150563 Ha  (err -0.043%)
- E (clipped est.) = 20.178651 Ha; var(E_L) = 1.0010e-01
- QGT/NTK: eff_rank = 2.99, kappa(S) = 9.989e+11, numerical rank = 1516/34354 (alignment on 2048 samples)
- alignment (final): cos(SR)=0.804, cos(plain)=0.042, NTK kappa=9.974e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
