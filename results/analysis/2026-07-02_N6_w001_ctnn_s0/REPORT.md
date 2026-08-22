# Phase analysis: N=6, omega=0.01, arch=ctnn_vcycle_big

- params: 79,813
- **E (unclipped) = 0.690200 +/- 0.000045 Ha**  (exact 0.69036, err -0.023%)
- E (zero-variance extrap, 5 pts) = 0.690999 Ha  (err +0.093%)
- E (clipped est.) = 0.690208 Ha; var(E_L) = 1.9454e-05
- QGT/NTK: eff_rank = 3.76, kappa(S) = 7.971e+09, numerical rank = 2047/79813 (alignment on 2048 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.103, NTK kappa=7.971e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
