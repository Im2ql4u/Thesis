# Phase analysis: N=6, omega=0.01, arch=deepset_big

- params: 47,669
- **E (unclipped) = 0.690512 +/- 0.000086 Ha**  (exact 0.69036, err +0.022%)
- E (zero-variance extrap, 6 pts) = 0.690032 Ha  (err -0.048%)
- E (clipped est.) = 0.690524 Ha; var(E_L) = 3.0382e-05
- QGT/NTK: eff_rank = 3.23, kappa(S) = 1.723e+09, numerical rank = 1023/47669 (alignment on 1024 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.085, NTK kappa=1.723e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
