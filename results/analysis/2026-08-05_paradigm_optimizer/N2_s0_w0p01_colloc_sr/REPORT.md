# Phase analysis: N=2, omega=0.01, arch=pinn

- params: 236,336
- **E (unclipped) = 0.073911 +/- 0.000011 Ha**  (exact 0.073839, err +0.097%)
- E (zero-variance extrap, 5 pts) = 0.073840 Ha  (err +0.002%)
- E (clipped est.) = 0.073887 Ha; var(E_L) = 1.2114e-06
- QGT/NTK: eff_rank = 1.22, kappa(S) = 9.988e+11, numerical rank = 1566/236336 (alignment on 2048 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.258, NTK kappa=9.909e+09

## Exact-truth checks (N=2)
- exact energy = 0.073835 Ha; exact Jastrow cusp dJ/dr|0 = 0.9976
- **|<Psi_net|Psi_exact>|^2 = 0.998213**

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
