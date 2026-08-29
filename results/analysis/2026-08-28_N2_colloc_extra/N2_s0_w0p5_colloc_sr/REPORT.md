# Phase analysis: N=2, omega=0.5, arch=pinn

- params: 236,336
- **E (unclipped) = 1.659452 +/- 0.000801 Ha**  (exact 1.65977, err -0.019%)
- E (zero-variance extrap, 5 pts) = 1.660144 Ha  (err +0.023%)
- E (clipped est.) = 1.654857 Ha; var(E_L) = 1.5013e-03
- QGT/NTK: eff_rank = 3.05, kappa(S) = 9.792e+11, numerical rank = 786/236336 (alignment on 1024 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.269, NTK kappa=9.930e+09

## Exact-truth checks (N=2)
- exact energy = 1.659772 Ha; exact Jastrow cusp dJ/dr|0 = 0.9993
- **|<Psi_net|Psi_exact>|^2 = 0.998952**

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
