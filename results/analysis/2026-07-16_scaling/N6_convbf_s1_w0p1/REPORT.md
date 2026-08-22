# Phase analysis: N=6, omega=0.1, arch=pinn

- params: 105,136
- **E (unclipped) = 3.558263 +/- 0.000687 Ha**  (exact 3.55385, err +0.124%)
- E (zero-variance extrap, 5 pts) = 3.550944 Ha  (err -0.082%)
- E (clipped est.) = 3.558217 Ha; var(E_L) = 3.1759e-03
- QGT/NTK: eff_rank = 17.83, kappa(S) = 4.683e+06, numerical rank = 2047/105136 (alignment on 2048 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.101, NTK kappa=4.683e+06

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
