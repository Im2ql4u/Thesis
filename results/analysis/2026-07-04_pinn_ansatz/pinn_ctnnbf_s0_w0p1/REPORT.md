# Phase analysis: N=6, omega=0.1, arch=pinn

- params: 100,080
- **E (unclipped) = 3.573411 +/- 0.000725 Ha**  (exact 3.55385, err +0.550%)
- E (zero-variance extrap, 5 pts) = 3.569233 Ha  (err +0.433%)
- E (clipped est.) = 3.573663 Ha; var(E_L) = 6.4250e-03
- QGT/NTK: eff_rank = 1.49, kappa(S) = 9.945e+11, numerical rank = 585/100080 (alignment on 2048 samples)
- alignment (final): cos(SR)=0.511, cos(plain)=0.019, NTK kappa=9.944e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
