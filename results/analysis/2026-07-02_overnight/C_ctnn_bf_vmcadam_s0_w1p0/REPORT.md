# Phase analysis: N=6, omega=1.0, arch=ctnn_vcycle_big

- params: 79,813
- **E (unclipped) = 20.166171 +/- 0.003237 Ha**  (exact 20.15932, err +0.034%)
- E (zero-variance extrap, 5 pts) = 20.142135 Ha  (err -0.085%)
- E (clipped est.) = 20.168409 Ha; var(E_L) = 6.0431e-02
- QGT/NTK: eff_rank = 1.31, kappa(S) = 1.825e+10, numerical rank = 2047/79813 (alignment on 2048 samples)
- alignment (final): cos(SR)=0.998, cos(plain)=0.082, NTK kappa=9.996e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
