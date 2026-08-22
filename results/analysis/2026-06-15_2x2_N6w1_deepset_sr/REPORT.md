# Phase analysis: N=6, omega=1.0, arch=deepset_big

- params: 47,669
- **E (unclipped) = 20.174107 +/- 0.007669 Ha**  (exact 20.15932, err +0.073%)
- E (zero-variance extrap, 6 pts) = 20.008538 Ha  (err -0.748%)
- E (clipped est.) = 20.173634 Ha; var(E_L) = 9.6599e-02
- QGT/NTK: eff_rank = 3.35, kappa(S) = 1.940e+10, numerical rank = 767/47669 (alignment on 768 samples)
- alignment (final): cos(SR)=0.987, cos(plain)=0.068, NTK kappa=9.888e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
