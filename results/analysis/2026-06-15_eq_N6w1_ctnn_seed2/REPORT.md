# Phase analysis: N=6, omega=1.0, arch=ctnn_vcycle_big

- params: 79,813
- **E (unclipped) = 20.153167 +/- 0.003276 Ha**  (exact 20.15932, err -0.031%)
- E (zero-variance extrap, 6 pts) = 20.148912 Ha  (err -0.052%)
- E (clipped est.) = 20.156864 Ha; var(E_L) = 2.2825e-02
- QGT/NTK: eff_rank = 1.39, kappa(S) = 5.162e+07, numerical rank = 767/79813 (alignment on 768 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.047, NTK kappa=5.162e+07

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
