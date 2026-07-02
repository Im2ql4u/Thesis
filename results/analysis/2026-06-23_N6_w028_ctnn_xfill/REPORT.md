# Phase analysis: N=6, omega=0.28, arch=ctnn_vcycle_big

- params: 79,813
- **E (unclipped) = 7.598218 +/- 0.000737 Ha**  (exact 7.60019, err -0.026%)
- E (zero-variance extrap, 6 pts) = 7.596570 Ha  (err -0.048%)
- E (clipped est.) = 7.597978 Ha; var(E_L) = 4.0700e-03
- QGT/NTK: eff_rank = 1.77, kappa(S) = 1.362e+08, numerical rank = 2047/79813 (alignment on 2048 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.038, NTK kappa=1.362e+08

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
