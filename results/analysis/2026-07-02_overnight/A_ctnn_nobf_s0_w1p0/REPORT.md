# Phase analysis: N=6, omega=1.0, arch=ctnn_vcycle_big

- params: 66,498
- **E (unclipped) = 20.173149 +/- 0.003409 Ha**  (exact 20.15932, err +0.069%)
- E (zero-variance extrap, 5 pts) = 20.160821 Ha  (err +0.007%)
- E (clipped est.) = 20.174308 Ha; var(E_L) = 7.0467e-02
- QGT/NTK: eff_rank = 1.24, kappa(S) = 2.292e+11, numerical rank = 2047/66498 (alignment on 2048 samples)
- alignment (final): cos(SR)=0.941, cos(plain)=0.024, NTK kappa=9.939e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
