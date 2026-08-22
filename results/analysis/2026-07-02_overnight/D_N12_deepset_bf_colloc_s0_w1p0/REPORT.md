# Phase analysis: N=12, omega=1.0, arch=deepset_big

- params: 47,669
- **E (unclipped) = 69.158779 +/- 0.151959 Ha**  (exact 65.7001, err +5.264%)
- E (zero-variance extrap, 5 pts) = 66.159308 Ha  (err +0.699%)
- E (clipped est.) = 69.158779 Ha; var(E_L) = 1.6896e+01
- QGT/NTK: eff_rank = 4.76, kappa(S) = 9.085e+08, numerical rank = 383/47669 (alignment on 384 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.374, NTK kappa=9.085e+08

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
