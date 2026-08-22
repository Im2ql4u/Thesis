# Phase analysis: N=12, omega=0.1, arch=deepset_big

- params: 47,669
- **E (unclipped) = 13.880434 +/- 0.051567 Ha**  (exact 12.26984, err +13.126%)
- E (zero-variance extrap, 5 pts) = 12.576467 Ha  (err +2.499%)
- E (clipped est.) = 13.880434 Ha; var(E_L) = 1.0744e+00
- QGT/NTK: eff_rank = 2.34, kappa(S) = 2.111e+09, numerical rank = 383/47669 (alignment on 384 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.695, NTK kappa=2.111e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
