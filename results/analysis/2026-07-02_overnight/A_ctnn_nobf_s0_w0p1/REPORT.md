# Phase analysis: N=6, omega=0.1, arch=ctnn_vcycle_big

- params: 66,498
- **E (unclipped) = 3.557035 +/- 0.000413 Ha**  (exact 3.55385, err +0.090%)
- E (zero-variance extrap, 5 pts) = 3.552937 Ha  (err -0.026%)
- E (clipped est.) = 3.556978 Ha; var(E_L) = 1.3505e-03
- QGT/NTK: eff_rank = 1.82, kappa(S) = 5.146e+10, numerical rank = 2047/66498 (alignment on 2048 samples)
- alignment (final): cos(SR)=0.976, cos(plain)=0.042, NTK kappa=9.943e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
