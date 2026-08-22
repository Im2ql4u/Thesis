# Phase analysis: N=12, omega=0.5, arch=pinn

- params: 105,136
- **E (unclipped) = 39.184000 +/- 0.005865 Ha**  (exact 39.1596, err +0.062%)
- E (zero-variance extrap, 5 pts) = 39.110167 Ha  (err -0.126%)
- E (clipped est.) = 39.185504 Ha; var(E_L) = 1.2171e-01
- QGT/NTK: eff_rank = 4.90, kappa(S) = 1.316e+07, numerical rank = 1023/105136 (alignment on 1024 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.074, NTK kappa=1.316e+07

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
