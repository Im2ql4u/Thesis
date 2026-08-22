# Phase analysis: N=6, omega=0.1, arch=ctnn_vcycle_big

- params: 79,813
- **E (unclipped) = 3.555838 +/- 0.000316 Ha**  (exact 3.55385, err +0.056%)
- E (zero-variance extrap, 5 pts) = 3.554766 Ha  (err +0.026%)
- E (clipped est.) = 3.555890 Ha; var(E_L) = 1.0591e-03
- QGT/NTK: eff_rank = 1.65, kappa(S) = 4.841e+10, numerical rank = 2047/79813 (alignment on 2048 samples)
- alignment (final): cos(SR)=0.991, cos(plain)=0.066, NTK kappa=9.955e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
