# Phase analysis: N=6, omega=0.01, arch=ctnn_vcycle_big

- params: 79,813
- **E (unclipped) = 0.689457 +/- 0.000026 Ha**  (exact 0.69036, err -0.131%)
- E (zero-variance extrap, 6 pts) = 0.689290 Ha  (err -0.155%)
- E (clipped est.) = 0.689521 Ha; var(E_L) = 4.1102e-06
- QGT/NTK: eff_rank = 5.02, kappa(S) = 7.949e+07, numerical rank = 1023/79813 (alignment on 1024 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.120, NTK kappa=7.949e+07

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
