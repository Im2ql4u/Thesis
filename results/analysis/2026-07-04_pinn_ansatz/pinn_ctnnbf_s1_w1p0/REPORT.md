# Phase analysis: N=6, omega=1.0, arch=pinn

- params: 100,080
- **E (unclipped) = 20.204064 +/- 0.004844 Ha**  (exact 20.15932, err +0.222%)
- E (zero-variance extrap, 5 pts) = 20.200699 Ha  (err +0.205%)
- E (clipped est.) = 20.205470 Ha; var(E_L) = 2.0065e-01
- QGT/NTK: eff_rank = 1.17, kappa(S) = 9.965e+11, numerical rank = 523/100080 (alignment on 2048 samples)
- alignment (final): cos(SR)=0.538, cos(plain)=0.031, NTK kappa=9.786e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
