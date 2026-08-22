# Phase analysis: N=2, omega=0.1, arch=pinn

- params: 236,336
- **E (unclipped) = 0.440829 +/- 0.000041 Ha**  (exact 0.44079, err +0.009%)
- E (zero-variance extrap, 5 pts) = 0.440918 Ha  (err +0.029%)
- E (clipped est.) = 0.440669 Ha; var(E_L) = 1.7904e-05
- QGT/NTK: eff_rank = 1.37, kappa(S) = 9.983e+11, numerical rank = 662/236336 (alignment on 2048 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.372, NTK kappa=9.927e+09

## Exact-truth checks (N=2)
- exact energy = 0.440792 Ha; exact Jastrow cusp dJ/dr|0 = 0.9990
- **|<Psi_net|Psi_exact>|^2 = 0.999662**

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
