# Phase analysis: N=6, omega=1.0, arch=ctnn_vcycle

- params: 12,373
- **E (unclipped) = 20.157654 +/- 0.003524 Ha**  (exact 20.15932, err -0.008%)
- E (zero-variance extrap, 6 pts) = 20.156383 Ha  (err -0.015%)
- E (clipped est.) = 20.162725 Ha; var(E_L) = 4.3291e-02
- QGT/NTK: eff_rank = 1.94, kappa(S) = 9.808e+09, numerical rank = 1023/12373 (alignment on 1024 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.092, NTK kappa=9.808e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
