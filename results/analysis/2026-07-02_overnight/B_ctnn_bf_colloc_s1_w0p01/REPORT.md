# Phase analysis: N=6, omega=0.01, arch=ctnn_vcycle_big

- params: 79,813
- **E (unclipped) = 7298.295813 +/- 100.829129 Ha**  (exact 0.69036, err +1057072.463%)
- E (zero-variance extrap, 5 pts) = -19.941485 Ha  (err -2988.563%)
- E (clipped est.) = 7033.799298 Ha; var(E_L) = 7.9097e+07
- QGT/NTK: eff_rank = 1.12, kappa(S) = 9.550e+11, numerical rank = 330/79813 (alignment on 2048 samples)
- alignment (final): cos(SR)=0.824, cos(plain)=0.374, NTK kappa=9.203e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
