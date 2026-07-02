# Phase analysis: N=6, omega=0.01, arch=deepset_big

- params: 47,669
- **E (unclipped) = 0.690809 +/- 0.000056 Ha**  (exact 0.69036, err +0.065%)
- E (zero-variance extrap, 5 pts) = 0.690170 Ha  (err -0.028%)
- E (clipped est.) = 0.690826 Ha; var(E_L) = 3.4149e-05
- QGT/NTK: eff_rank = 3.98, kappa(S) = 9.974e+11, numerical rank = 1659/47669 (alignment on 2048 samples)
- alignment (final): cos(SR)=0.810, cos(plain)=0.031, NTK kappa=9.976e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
