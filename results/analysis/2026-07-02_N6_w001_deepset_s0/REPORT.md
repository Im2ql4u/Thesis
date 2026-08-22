# Phase analysis: N=6, omega=0.01, arch=deepset_big

- params: 47,669
- **E (unclipped) = 0.690829 +/- 0.000075 Ha**  (exact 0.69036, err +0.068%)
- E (zero-variance extrap, 5 pts) = 0.689872 Ha  (err -0.071%)
- E (clipped est.) = 0.690854 Ha; var(E_L) = 3.3958e-05
- QGT/NTK: eff_rank = 3.77, kappa(S) = 9.989e+11, numerical rank = 1502/47669 (alignment on 2048 samples)
- alignment (final): cos(SR)=0.783, cos(plain)=0.040, NTK kappa=9.998e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
