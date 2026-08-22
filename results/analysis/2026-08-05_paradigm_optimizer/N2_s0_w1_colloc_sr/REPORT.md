# Phase analysis: N=2, omega=1.0, arch=pinn

- params: 236,336
- **E (unclipped) = 3.001054 +/- 0.000443 Ha**  (exact 3.0, err +0.035%)
- E (zero-variance extrap, 5 pts) = 3.000350 Ha  (err +0.012%)
- E (clipped est.) = 2.992173 Ha; var(E_L) = 1.8322e-03
- QGT/NTK: eff_rank = 4.17, kappa(S) = 9.996e+11, numerical rank = 813/236336 (alignment on 2048 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.198, NTK kappa=9.972e+09

## Exact-truth checks (N=2)
- exact energy = 3.000000 Ha; exact Jastrow cusp dJ/dr|0 = 0.9991
- **|<Psi_net|Psi_exact>|^2 = 0.999709**

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
