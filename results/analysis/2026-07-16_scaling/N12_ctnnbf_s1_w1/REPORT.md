# Phase analysis: N=12, omega=1.0, arch=pinn

- params: 236,336
- **E (unclipped) = 65.705213 +/- 0.006570 Ha**  (exact 65.7001, err +0.008%)
- E (zero-variance extrap, 1 pts) = 65.705213 Ha  (err +0.008%)
- E (clipped est.) = 65.703829 Ha; var(E_L) = 1.3961e-01
- QGT/NTK: eff_rank = 3.88, kappa(S) = 1.653e+07, numerical rank = 1023/236336 (alignment on 1024 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.073, NTK kappa=1.653e+07

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
