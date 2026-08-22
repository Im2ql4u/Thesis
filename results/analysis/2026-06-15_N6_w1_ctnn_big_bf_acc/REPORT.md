# Phase analysis: N=6, omega=1.0, arch=ctnn_vcycle_big

- params: 79,813
- **E (unclipped) = 20.153586 +/- 0.001669 Ha**  (exact 20.15932, err -0.028%)
- E (zero-variance extrap, 6 pts) = 20.147149 Ha  (err -0.060%)
- E (clipped est.) = 20.155486 Ha; var(E_L) = 1.0923e-02
- QGT/NTK: eff_rank = 1.46, kappa(S) = 3.141e+07, numerical rank = 1023/79813 (alignment on 1024 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.061, NTK kappa=3.141e+07

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
