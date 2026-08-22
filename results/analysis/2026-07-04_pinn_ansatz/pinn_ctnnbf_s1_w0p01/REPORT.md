# Phase analysis: N=6, omega=0.01, arch=pinn

- params: 100,080
- **E (unclipped) = 0.694414 +/- 0.000146 Ha**  (exact 0.69036, err +0.587%)
- E (zero-variance extrap, 5 pts) = 0.691045 Ha  (err +0.099%)
- E (clipped est.) = 0.694509 Ha; var(E_L) = 1.0418e-04
- QGT/NTK: eff_rank = 1.23, kappa(S) = 9.887e+11, numerical rank = 611/100080 (alignment on 2048 samples)
- alignment (final): cos(SR)=0.605, cos(plain)=0.039, NTK kappa=9.906e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
