# Phase analysis: N=6, omega=1.0, arch=pinn

- params: 236,336
- **E (unclipped) = 20.158111 +/- 0.001734 Ha**  (exact 20.15932, err -0.006%)
- E (zero-variance extrap, 5 pts) = 20.187326 Ha  (err +0.139%)
- E (clipped est.) = 20.158864 Ha; var(E_L) = 2.3126e-02
- QGT/NTK: eff_rank = 4.09, kappa(S) = 2.184e+07, numerical rank = 2047/236336 (alignment on 2048 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.088, NTK kappa=2.184e+07

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
