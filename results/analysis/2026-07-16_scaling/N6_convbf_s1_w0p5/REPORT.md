# Phase analysis: N=6, omega=0.5, arch=pinn

- params: 105,136
- **E (unclipped) = 11.791543 +/- 0.001747 Ha**  (exact 11.78484, err +0.057%)
- E (zero-variance extrap, 5 pts) = 11.770462 Ha  (err -0.122%)
- E (clipped est.) = 11.791800 Ha; var(E_L) = 2.8911e-02
- QGT/NTK: eff_rank = 11.40, kappa(S) = 2.689e+07, numerical rank = 2047/105136 (alignment on 2048 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.096, NTK kappa=2.689e+07

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
