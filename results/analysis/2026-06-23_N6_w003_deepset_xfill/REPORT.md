# Phase analysis: N=6, omega=0.03, arch=deepset_big

- params: 47,669
- **E (unclipped) = 1.497321 +/- 0.000146 Ha**
- E (zero-variance extrap, 6 pts) = 1.493461 Ha
- E (clipped est.) = 1.497357 Ha; var(E_L) = 2.0854e-04
- QGT/NTK: eff_rank = 4.07, kappa(S) = 3.631e+11, numerical rank = 2047/47669 (alignment on 2048 samples)
- alignment (final): cos(SR)=0.914, cos(plain)=0.057, NTK kappa=9.999e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
