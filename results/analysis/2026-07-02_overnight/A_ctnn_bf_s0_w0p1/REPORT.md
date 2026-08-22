# Phase analysis: N=6, omega=0.1, arch=ctnn_vcycle_big

- params: 79,813
- **E (unclipped) = 3.556256 +/- 0.000347 Ha**  (exact 3.55385, err +0.068%)
- E (zero-variance extrap, 5 pts) = 3.555812 Ha  (err +0.055%)
- E (clipped est.) = 3.556278 Ha; var(E_L) = 9.4499e-04
- QGT/NTK: eff_rank = 1.44, kappa(S) = 2.352e+10, numerical rank = 2047/79813 (alignment on 2048 samples)
- alignment (final): cos(SR)=0.997, cos(plain)=0.025, NTK kappa=9.938e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
