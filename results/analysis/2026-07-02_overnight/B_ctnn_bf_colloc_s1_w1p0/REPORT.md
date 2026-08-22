# Phase analysis: N=6, omega=1.0, arch=ctnn_vcycle_big

- params: 79,813
- **E (unclipped) = 20.466842 +/- 0.010069 Ha**  (exact 20.15932, err +1.525%)
- E (zero-variance extrap, 5 pts) = 20.129343 Ha  (err -0.149%)
- E (clipped est.) = 20.378158 Ha; var(E_L) = 9.0097e-01
- QGT/NTK: eff_rank = 1.35, kappa(S) = 9.991e+11, numerical rank = 1742/79813 (alignment on 2048 samples)
- alignment (final): cos(SR)=0.999, cos(plain)=0.106, NTK kappa=9.900e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
