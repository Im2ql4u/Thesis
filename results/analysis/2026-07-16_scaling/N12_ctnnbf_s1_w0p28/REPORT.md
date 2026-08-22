# Phase analysis: N=12, omega=0.28, arch=pinn

- params: 236,336
- **E (unclipped) = 25.630955 +/- 0.001765 Ha**  (exact 25.63577, err -0.019%)
- E (zero-variance extrap, 5 pts) = 25.631739 Ha  (err -0.016%)
- E (clipped est.) = 25.630560 Ha; var(E_L) = 1.3343e-02
- QGT/NTK: eff_rank = 4.48, kappa(S) = 5.440e+05, numerical rank = 1023/236336 (alignment on 1024 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.072, NTK kappa=5.440e+05

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
