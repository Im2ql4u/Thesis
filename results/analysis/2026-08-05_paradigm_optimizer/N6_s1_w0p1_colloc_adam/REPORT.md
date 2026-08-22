# Phase analysis: N=6, omega=0.1, arch=pinn

- params: 236,336
- **E (unclipped) = 3.609221 +/- 0.001237 Ha**  (exact 3.55385, err +1.558%)
- E (zero-variance extrap, 5 pts) = 3.550927 Ha  (err -0.082%)
- E (clipped est.) = 3.607801 Ha; var(E_L) = 1.8117e-02
- QGT/NTK: eff_rank = 5.22, kappa(S) = 2.162e+07, numerical rank = 2047/236336 (alignment on 2048 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.376, NTK kappa=2.162e+07

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
