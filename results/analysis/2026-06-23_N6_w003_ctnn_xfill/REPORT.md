# Phase analysis: N=6, omega=0.03, arch=ctnn_vcycle_big

- params: 79,813
- **E (unclipped) = 1.495697 +/- 0.000149 Ha**
- E (zero-variance extrap, 6 pts) = 1.493451 Ha
- E (clipped est.) = 1.495795 Ha; var(E_L) = 1.3542e-04
- QGT/NTK: eff_rank = 3.92, kappa(S) = 1.150e+09, numerical rank = 2047/79813 (alignment on 2048 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.098, NTK kappa=1.150e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
