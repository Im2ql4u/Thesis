# Phase analysis: N=6, omega=0.1, arch=ctnn_vcycle_big

- params: 79,813
- **E (unclipped) = 3.555846 +/- 0.000388 Ha**  (exact 3.55385, err +0.056%)
- E (zero-variance extrap, 5 pts) = 3.554289 Ha  (err +0.012%)
- E (clipped est.) = 3.556275 Ha; var(E_L) = 1.1813e-03
- QGT/NTK: eff_rank = 1.44, kappa(S) = 1.124e+10, numerical rank = 2047/79813 (alignment on 2048 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.035, NTK kappa=9.928e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
