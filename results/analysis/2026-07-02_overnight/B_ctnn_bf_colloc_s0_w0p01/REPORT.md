# Phase analysis: N=6, omega=0.01, arch=ctnn_vcycle_big

- params: 79,813
- **E (unclipped) = -29.069856 +/- 0.139901 Ha**  (exact 0.69036, err -4310.826%)
- E (zero-variance extrap, 5 pts) = -1.028828 Ha  (err -249.028%)
- E (clipped est.) = -1.503929 Ha; var(E_L) = 2.7379e+02
- QGT/NTK: eff_rank = 1.46, kappa(S) = 9.804e+11, numerical rank = 1041/79813 (alignment on 2048 samples)
- alignment (final): cos(SR)=0.999, cos(plain)=0.294, NTK kappa=9.916e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
