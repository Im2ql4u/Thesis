# Phase analysis: N=6, omega=0.1, arch=pinn

- params: 67,248
- **E (unclipped) = 3.569502 +/- 0.000754 Ha**  (exact 3.55385, err +0.440%)
- E (zero-variance extrap, 5 pts) = 3.593847 Ha  (err +1.125%)
- E (clipped est.) = 3.569950 Ha; var(E_L) = 5.6454e-03
- QGT/NTK: eff_rank = 2.44, kappa(S) = 9.985e+11, numerical rank = 1392/67248 (alignment on 2048 samples)
- alignment (final): cos(SR)=0.735, cos(plain)=0.049, NTK kappa=9.964e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
