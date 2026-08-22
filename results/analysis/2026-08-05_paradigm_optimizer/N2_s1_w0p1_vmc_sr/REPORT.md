# Phase analysis: N=2, omega=0.1, arch=pinn

- params: 236,336
- **E (unclipped) = 0.440812 +/- 0.000002 Ha**  (exact 0.44079, err +0.005%)
- E (zero-variance extrap, 5 pts) = 0.440652 Ha  (err -0.031%)
- E (clipped est.) = 0.440818 Ha; var(E_L) = 3.1982e-08
- QGT/NTK: eff_rank = 2.13, kappa(S) = 9.911e+11, numerical rank = 1160/236336 (alignment on 2048 samples)
- alignment (final): cos(SR)=0.990, cos(plain)=0.054, NTK kappa=9.990e+09

## Exact-truth checks (N=2)
- exact energy = 0.440792 Ha; exact Jastrow cusp dJ/dr|0 = 0.9990
- **|<Psi_net|Psi_exact>|^2 = 1.000000**

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
