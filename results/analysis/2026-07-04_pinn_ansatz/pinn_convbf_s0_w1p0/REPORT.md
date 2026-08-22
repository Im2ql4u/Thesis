# Phase analysis: N=6, omega=1.0, arch=pinn

- params: 67,248
- **E (unclipped) = 20.198036 +/- 0.004277 Ha**  (exact 20.15932, err +0.192%)
- E (zero-variance extrap, 5 pts) = 20.206955 Ha  (err +0.236%)
- E (clipped est.) = 20.199734 Ha; var(E_L) = 1.9322e-01
- QGT/NTK: eff_rank = 1.88, kappa(S) = 9.868e+11, numerical rank = 910/67248 (alignment on 2048 samples)
- alignment (final): cos(SR)=0.591, cos(plain)=0.038, NTK kappa=9.970e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
