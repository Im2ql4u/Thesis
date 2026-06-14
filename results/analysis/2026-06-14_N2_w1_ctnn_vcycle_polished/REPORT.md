# Phase analysis: N=2, omega=1.0, arch=ctnn_vcycle

- params: 9,842
- **E (unclipped) = 3.000127 +/- 0.000024 Ha**  (exact 3.0, err +0.004%)
- E (zero-variance extrap, 6 pts) = 3.001723 Ha  (err +0.057%)
- E (clipped est.) = 3.000174 Ha; var(E_L) = 5.0527e-06
- QGT/NTK: eff_rank = 1.24, kappa(S) = 9.367e+11, numerical rank = 156/9842 (alignment on 2048 samples)
- alignment (final): cos(SR)=0.997, cos(plain)=0.041, NTK kappa=9.573e+09

## Exact-truth checks (N=2)
- exact energy = 3.000000 Ha; exact Jastrow cusp dJ/dr|0 = 0.9991
- **|<Psi_net|Psi_exact>|^2 = 1.000000**

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
