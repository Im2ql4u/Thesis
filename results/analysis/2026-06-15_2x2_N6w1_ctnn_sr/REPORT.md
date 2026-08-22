# Phase analysis: N=6, omega=1.0, arch=ctnn_vcycle_big

- params: 79,813
- **E (unclipped) = 20.172702 +/- 0.002807 Ha**  (exact 20.15932, err +0.066%)
- E (zero-variance extrap, 6 pts) = 20.162875 Ha  (err +0.018%)
- E (clipped est.) = 20.165196 Ha; var(E_L) = 2.6396e-02
- QGT/NTK: eff_rank = 1.60, kappa(S) = 8.273e+07, numerical rank = 767/79813 (alignment on 768 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.079, NTK kappa=8.273e+07

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
