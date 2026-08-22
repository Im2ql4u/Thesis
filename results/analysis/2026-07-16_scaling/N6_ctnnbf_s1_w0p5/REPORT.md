# Phase analysis: N=6, omega=0.5, arch=pinn

- params: 236,336
- **E (unclipped) = 11.783894 +/- 0.000838 Ha**  (exact 11.78484, err -0.008%)
- E (zero-variance extrap, 5 pts) = 11.784685 Ha  (err -0.001%)
- E (clipped est.) = 11.783693 Ha; var(E_L) = 6.4026e-03
- QGT/NTK: eff_rank = 4.57, kappa(S) = 9.190e+06, numerical rank = 2047/236336 (alignment on 2048 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.042, NTK kappa=9.190e+06

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
