# Phase analysis: N=6, omega=0.1, arch=deepset_big

- params: 47,669
- **E (unclipped) = 3.554012 +/- 0.000587 Ha**  (exact 3.55385, err +0.005%)
- E (zero-variance extrap, 6 pts) = 3.546378 Ha  (err -0.210%)
- E (clipped est.) = 3.554320 Ha; var(E_L) = 1.1410e-03
- QGT/NTK: eff_rank = 10.36, kappa(S) = 9.740e+06, numerical rank = 1023/47669 (alignment on 1024 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.119, NTK kappa=9.740e+06

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
