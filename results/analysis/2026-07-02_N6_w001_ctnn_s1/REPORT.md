# Phase analysis: N=6, omega=0.01, arch=ctnn_vcycle_big

- params: 79,813
- **E (unclipped) = 0.689960 +/- 0.000054 Ha**  (exact 0.69036, err -0.058%)
- E (zero-variance extrap, 5 pts) = 0.690890 Ha  (err +0.077%)
- E (clipped est.) = 0.689981 Ha; var(E_L) = 1.6322e-05
- QGT/NTK: eff_rank = 3.59, kappa(S) = 7.201e+09, numerical rank = 2047/79813 (alignment on 2048 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.040, NTK kappa=7.201e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
