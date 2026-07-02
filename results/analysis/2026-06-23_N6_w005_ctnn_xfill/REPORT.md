# Phase analysis: N=6, omega=0.05, arch=ctnn_vcycle_big

- params: 79,813
- **E (unclipped) = 2.153641 +/- 0.000162 Ha**
- E (zero-variance extrap, 6 pts) = 2.151770 Ha
- E (clipped est.) = 2.153628 Ha; var(E_L) = 2.9769e-04
- QGT/NTK: eff_rank = 2.72, kappa(S) = 1.450e+08, numerical rank = 2047/79813 (alignment on 2048 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.069, NTK kappa=1.450e+08

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
