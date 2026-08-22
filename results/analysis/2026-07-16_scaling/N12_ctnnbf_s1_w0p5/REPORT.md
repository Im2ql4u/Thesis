# Phase analysis: N=12, omega=0.5, arch=pinn

- params: 236,336
- **E (unclipped) = 39.152286 +/- 0.002889 Ha**  (exact 39.1596, err -0.019%)
- E (zero-variance extrap, 5 pts) = 39.097661 Ha  (err -0.158%)
- E (clipped est.) = 39.153086 Ha; var(E_L) = 3.8052e-02
- QGT/NTK: eff_rank = 4.17, kappa(S) = 1.897e+06, numerical rank = 1023/236336 (alignment on 1024 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.064, NTK kappa=1.897e+06

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
