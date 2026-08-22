# Phase analysis: N=6, omega=1.0, arch=pinn

- params: 100,080
- **E (unclipped) = 20.200188 +/- 0.004429 Ha**  (exact 20.15932, err +0.203%)
- E (zero-variance extrap, 5 pts) = 20.090451 Ha  (err -0.342%)
- E (clipped est.) = 20.201766 Ha; var(E_L) = 1.9729e-01
- QGT/NTK: eff_rank = 1.12, kappa(S) = 9.989e+11, numerical rank = 424/100080 (alignment on 2048 samples)
- alignment (final): cos(SR)=0.541, cos(plain)=0.065, NTK kappa=9.945e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
