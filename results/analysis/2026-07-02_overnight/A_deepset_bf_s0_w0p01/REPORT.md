# Phase analysis: N=6, omega=0.01, arch=deepset_big

- params: 47,669
- **E (unclipped) = 0.690853 +/- 0.000082 Ha**  (exact 0.69036, err +0.071%)
- E (zero-variance extrap, 5 pts) = 0.688567 Ha  (err -0.260%)
- E (clipped est.) = 0.690869 Ha; var(E_L) = 3.5917e-05
- QGT/NTK: eff_rank = 3.96, kappa(S) = 9.992e+11, numerical rank = 1695/47669 (alignment on 2048 samples)
- alignment (final): cos(SR)=0.856, cos(plain)=0.053, NTK kappa=9.992e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
