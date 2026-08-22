# Phase analysis: N=6, omega=0.1, arch=ctnn_vcycle_big

- params: 79,813
- **E (unclipped) = 3.610765 +/- 0.001763 Ha**  (exact 3.55385, err +1.601%)
- E (zero-variance extrap, 5 pts) = 3.590087 Ha  (err +1.020%)
- E (clipped est.) = 3.607857 Ha; var(E_L) = 2.4694e-02
- QGT/NTK: eff_rank = 1.73, kappa(S) = 2.125e+09, numerical rank = 2047/79813 (alignment on 2048 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.277, NTK kappa=2.125e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
