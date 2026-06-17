# Phase analysis: N=6, omega=1.0, arch=deepset_big

- params: 47,669
- **E (unclipped) = 20.155640 +/- 0.002326 Ha**  (exact 20.15932, err -0.018%)
- E (zero-variance extrap, 6 pts) = 20.149015 Ha  (err -0.051%)
- E (clipped est.) = 20.156532 Ha; var(E_L) = 2.5094e-02
- QGT/NTK: eff_rank = 4.80, kappa(S) = 6.707e+07, numerical rank = 1023/47669 (alignment on 1024 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.112, NTK kappa=6.707e+07

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
