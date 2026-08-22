# Phase analysis: N=12, omega=0.1, arch=pinn

- params: 105,136
- **E (unclipped) = 12.293808 +/- 0.002127 Ha**  (exact 12.26984, err +0.195%)
- E (zero-variance extrap, 5 pts) = 12.261349 Ha  (err -0.069%)
- E (clipped est.) = 12.294002 Ha; var(E_L) = 1.3828e-02
- QGT/NTK: eff_rank = 6.21, kappa(S) = 1.058e+07, numerical rank = 1023/105136 (alignment on 1024 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.104, NTK kappa=1.058e+07

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
