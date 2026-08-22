# Phase analysis: N=6, omega=0.01, arch=deepset_big

- params: 47,669
- **E (unclipped) = 0.690752 +/- 0.000067 Ha**  (exact 0.69036, err +0.057%)
- E (zero-variance extrap, 5 pts) = 0.688962 Ha  (err -0.202%)
- E (clipped est.) = 0.690771 Ha; var(E_L) = 3.3579e-05
- QGT/NTK: eff_rank = 3.32, kappa(S) = 9.854e+11, numerical rank = 1892/47669 (alignment on 2048 samples)
- alignment (final): cos(SR)=0.895, cos(plain)=0.090, NTK kappa=9.929e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
