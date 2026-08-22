# Phase analysis: N=20, omega=0.1, arch=pinn

- params: 105,136
- **E (unclipped) = 30.033725 +/- 0.004095 Ha**  (exact 29.9779, err +0.186%)
- E (zero-variance extrap, 5 pts) = 29.985088 Ha  (err +0.024%)
- E (clipped est.) = 30.033725 Ha; var(E_L) = 3.4661e-02
- QGT/NTK: eff_rank = 5.01, kappa(S) = 1.406e+07, numerical rank = 511/105136 (alignment on 512 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.147, NTK kappa=1.406e+07

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
