# Phase analysis: N=6, omega=0.5, arch=ctnn_vcycle_big

- params: 79,813
- **E (unclipped) = 11.784733 +/- 0.001115 Ha**  (exact 11.78484, err -0.001%)
- E (zero-variance extrap, 6 pts) = 11.790089 Ha  (err +0.045%)
- E (clipped est.) = 11.783417 Ha; var(E_L) = 4.6209e-03
- QGT/NTK: eff_rank = 1.56, kappa(S) = 1.942e+07, numerical rank = 1023/79813 (alignment on 1024 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.074, NTK kappa=1.942e+07

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
