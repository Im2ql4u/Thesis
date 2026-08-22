# Phase analysis: N=6, omega=1.0, arch=deepset_big

- params: 47,669
- **E (unclipped) = 20.159223 +/- 0.002196 Ha**  (exact 20.15932, err -0.000%)
- E (zero-variance extrap, 5 pts) = 20.151124 Ha  (err -0.041%)
- E (clipped est.) = 20.166210 Ha; var(E_L) = 5.9937e-02
- QGT/NTK: eff_rank = 3.06, kappa(S) = 3.418e+10, numerical rank = 2047/47669 (alignment on 2048 samples)
- alignment (final): cos(SR)=0.998, cos(plain)=0.071, NTK kappa=9.970e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
