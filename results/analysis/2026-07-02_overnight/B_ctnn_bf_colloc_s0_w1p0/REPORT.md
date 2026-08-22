# Phase analysis: N=6, omega=1.0, arch=ctnn_vcycle_big

- params: 79,813
- **E (unclipped) = 20.502958 +/- 0.010899 Ha**  (exact 20.15932, err +1.705%)
- E (zero-variance extrap, 5 pts) = 20.191102 Ha  (err +0.158%)
- E (clipped est.) = 20.420195 Ha; var(E_L) = 1.0447e+00
- QGT/NTK: eff_rank = 1.22, kappa(S) = 6.909e+10, numerical rank = 2047/79813 (alignment on 2048 samples)
- alignment (final): cos(SR)=0.999, cos(plain)=0.119, NTK kappa=9.908e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
