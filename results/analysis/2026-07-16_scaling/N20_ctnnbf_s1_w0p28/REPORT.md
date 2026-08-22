# Phase analysis: N=20, omega=0.28, arch=pinn

- params: 236,336
- **E (unclipped) = 61.949016 +/- 0.006581 Ha**  (exact 61.9268, err +0.036%)
- E (zero-variance extrap, 5 pts) = 61.869671 Ha  (err -0.092%)
- E (clipped est.) = 61.950074 Ha; var(E_L) = 8.5063e-02
- QGT/NTK: eff_rank = 3.61, kappa(S) = 4.203e+05, numerical rank = 511/236336 (alignment on 512 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.100, NTK kappa=4.203e+05

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
