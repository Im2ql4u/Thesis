# Phase analysis: N=6, omega=1.0, arch=deepset_match

- params: 89,421
- **E (unclipped) = 20.179267 +/- 0.006967 Ha**  (exact 20.15932, err +0.099%)
- E (zero-variance extrap, 6 pts) = 20.134284 Ha  (err -0.124%)
- E (clipped est.) = 20.179055 Ha; var(E_L) = 9.1712e-02
- QGT/NTK: eff_rank = 3.60, kappa(S) = 2.098e+10, numerical rank = 767/89421 (alignment on 768 samples)
- alignment (final): cos(SR)=0.985, cos(plain)=0.070, NTK kappa=9.951e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
