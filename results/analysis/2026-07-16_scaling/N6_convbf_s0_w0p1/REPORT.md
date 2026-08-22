# Phase analysis: N=6, omega=0.1, arch=pinn

- params: 105,136
- **E (unclipped) = 3.559939 +/- 0.000679 Ha**  (exact 3.55385, err +0.171%)
- E (zero-variance extrap, 5 pts) = 3.557829 Ha  (err +0.112%)
- E (clipped est.) = 3.559579 Ha; var(E_L) = 3.4875e-03
- QGT/NTK: eff_rank = 17.20, kappa(S) = 6.398e+06, numerical rank = 2047/105136 (alignment on 2048 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.141, NTK kappa=6.398e+06

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
