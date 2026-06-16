# Phase analysis: N=6, omega=0.1, arch=ctnn_vcycle_big

- params: 79,813
- **E (unclipped) = 3.553515 +/- 0.000359 Ha**  (exact 3.55385, err -0.009%)
- E (zero-variance extrap, 6 pts) = 3.551520 Ha  (err -0.066%)
- E (clipped est.) = 3.553774 Ha; var(E_L) = 5.6328e-04
- QGT/NTK: eff_rank = 2.37, kappa(S) = 1.350e+07, numerical rank = 1023/79813 (alignment on 1024 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.178, NTK kappa=1.350e+07

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
