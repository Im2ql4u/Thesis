# Phase analysis: N=2, omega=1.0, arch=ctnn_vcycle

- params: 9,842
- **E (unclipped) = 3.000639 +/- 0.000037 Ha**  (exact 3.0, err +0.021%)
- E (zero-variance extrap, 5 pts) = 3.001842 Ha  (err +0.061%)
- E (clipped est.) = 2.999235 Ha; var(E_L) = 3.0127e-06
- QGT/NTK: eff_rank = 1.12, kappa(S) = 9.832e+11, numerical rank = 176/9842 (alignment on 1024 samples)
- alignment (final): cos(SR)=0.983, cos(plain)=0.125, NTK kappa=9.716e+09

## Exact-truth checks (N=2)
- exact energy = 3.000000 Ha; exact Jastrow cusp dJ/dr|0 = 0.9991
- **|<Psi_net|Psi_exact>|^2 = 0.999999**

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
