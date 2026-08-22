# Phase analysis: N=6, omega=0.5, arch=pinn

- params: 105,136
- **E (unclipped) = 11.789921 +/- 0.001957 Ha**  (exact 11.78484, err +0.043%)
- E (zero-variance extrap, 5 pts) = 11.764724 Ha  (err -0.171%)
- E (clipped est.) = 11.790053 Ha; var(E_L) = 3.0358e-02
- QGT/NTK: eff_rank = 11.76, kappa(S) = 2.474e+07, numerical rank = 2047/105136 (alignment on 2048 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.141, NTK kappa=2.474e+07

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
