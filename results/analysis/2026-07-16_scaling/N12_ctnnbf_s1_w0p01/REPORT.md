# Phase analysis: N=12, omega=0.01, arch=pinn

- params: 236,336
- **E (unclipped) = 2.473291 +/- 0.000184 Ha**  (exact 2.47363, err -0.014%)
- E (zero-variance extrap, 5 pts) = 2.482198 Ha  (err +0.346%)
- E (clipped est.) = 2.473353 Ha; var(E_L) = 1.3786e-04
- QGT/NTK: eff_rank = 9.28, kappa(S) = 1.070e+05, numerical rank = 1023/236336 (alignment on 1024 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.121, NTK kappa=1.070e+05

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
