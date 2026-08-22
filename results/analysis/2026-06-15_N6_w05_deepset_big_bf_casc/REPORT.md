# Phase analysis: N=6, omega=0.5, arch=deepset_big

- params: 47,669
- **E (unclipped) = 11.783486 +/- 0.001367 Ha**  (exact 11.78484, err -0.011%)
- E (zero-variance extrap, 6 pts) = 11.794139 Ha  (err +0.079%)
- E (clipped est.) = 11.782256 Ha; var(E_L) = 1.0141e-02
- QGT/NTK: eff_rank = 6.17, kappa(S) = 1.645e+07, numerical rank = 1023/47669 (alignment on 1024 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.085, NTK kappa=1.645e+07

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
