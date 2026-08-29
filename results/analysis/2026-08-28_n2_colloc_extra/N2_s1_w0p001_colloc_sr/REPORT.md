# Phase analysis: N=2, omega=0.001, arch=pinn

- params: 236,336
- **E (unclipped) = 0.013853 +/- 0.000005 Ha**  (exact 0.013778, err +0.546%)
- E (zero-variance extrap, 5 pts) = 0.013814 Ha  (err +0.264%)
- E (clipped est.) = 0.013814 Ha; var(E_L) = 2.2754e-07
- QGT/NTK: eff_rank = 1.32, kappa(S) = 9.981e+11, numerical rank = 379/236336 (alignment on 2048 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.309, NTK kappa=9.709e+09

## Exact-truth checks (N=2)
- exact energy = 0.013768 Ha; exact Jastrow cusp dJ/dr|0 = 0.9929
- **|<Psi_net|Psi_exact>|^2 = 0.986333**

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
