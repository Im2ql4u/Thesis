# Phase analysis: N=6, omega=1.0, arch=deepset_xl

- params: 164,069
- **E (unclipped) = 20.148588 +/- 0.003619 Ha**  (exact 20.15932, err -0.053%)
- E (zero-variance extrap, 6 pts) = 20.127066 Ha  (err -0.160%)
- E (clipped est.) = 20.148809 Ha; var(E_L) = 3.3240e-02
- QGT/NTK: eff_rank = 4.76, kappa(S) = 1.441e+07, numerical rank = 767/164069 (alignment on 768 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.156, NTK kappa=1.441e+07

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
