# Phase analysis: N=12, omega=0.28, arch=pinn

- params: 105,136
- **E (unclipped) = 25.661820 +/- 0.003167 Ha**  (exact 25.63577, err +0.102%)
- E (zero-variance extrap, 5 pts) = 25.609445 Ha  (err -0.103%)
- E (clipped est.) = 25.661711 Ha; var(E_L) = 4.9255e-02
- QGT/NTK: eff_rank = 6.64, kappa(S) = 3.983e+06, numerical rank = 1023/105136 (alignment on 1024 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.116, NTK kappa=3.983e+06

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
