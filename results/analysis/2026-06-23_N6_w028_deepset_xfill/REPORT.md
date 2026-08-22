# Phase analysis: N=6, omega=0.28, arch=deepset_big

- params: 47,669
- **E (unclipped) = 7.600285 +/- 0.000708 Ha**  (exact 7.60019, err +0.001%)
- E (zero-variance extrap, 6 pts) = 7.587786 Ha  (err -0.163%)
- E (clipped est.) = 7.600531 Ha; var(E_L) = 4.9711e-03
- QGT/NTK: eff_rank = 7.51, kappa(S) = 1.139e+08, numerical rank = 2047/47669 (alignment on 2048 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.072, NTK kappa=1.139e+08

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
