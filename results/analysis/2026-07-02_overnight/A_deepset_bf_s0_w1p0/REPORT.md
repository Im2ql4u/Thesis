# Phase analysis: N=6, omega=1.0, arch=deepset_big

- params: 47,669
- **E (unclipped) = 20.180074 +/- 0.003839 Ha**  (exact 20.15932, err +0.103%)
- E (zero-variance extrap, 5 pts) = 20.168913 Ha  (err +0.048%)
- E (clipped est.) = 20.179817 Ha; var(E_L) = 1.0222e-01
- QGT/NTK: eff_rank = 3.16, kappa(S) = 9.988e+11, numerical rank = 1708/47669 (alignment on 2048 samples)
- alignment (final): cos(SR)=0.841, cos(plain)=0.035, NTK kappa=9.983e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
