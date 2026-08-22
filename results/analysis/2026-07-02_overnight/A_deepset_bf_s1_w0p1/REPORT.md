# Phase analysis: N=6, omega=0.1, arch=deepset_big

- params: 47,669
- **E (unclipped) = 3.558963 +/- 0.000485 Ha**  (exact 3.55385, err +0.144%)
- E (zero-variance extrap, 5 pts) = 3.556058 Ha  (err +0.062%)
- E (clipped est.) = 3.559059 Ha; var(E_L) = 1.8205e-03
- QGT/NTK: eff_rank = 3.71, kappa(S) = 9.980e+11, numerical rank = 1411/47669 (alignment on 2048 samples)
- alignment (final): cos(SR)=0.736, cos(plain)=0.054, NTK kappa=9.875e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
