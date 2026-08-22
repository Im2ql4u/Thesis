# Phase analysis: N=12, omega=1.0, arch=deepset_big

- params: 47,669
- **E (unclipped) = 68.609187 +/- 0.214814 Ha**  (exact 65.7001, err +4.428%)
- E (zero-variance extrap, 6 pts) = 67.109214 Ha  (err +2.145%)
- E (clipped est.) = 68.609187 Ha; var(E_L) = 1.6180e+01
- QGT/NTK: eff_rank = 4.82, kappa(S) = 5.199e+07, numerical rank = 255/47669 (alignment on 256 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.516, NTK kappa=5.199e+07

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
