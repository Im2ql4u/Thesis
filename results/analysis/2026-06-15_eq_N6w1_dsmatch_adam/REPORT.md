# Phase analysis: N=6, omega=1.0, arch=deepset_match

- params: 89,421
- **E (unclipped) = 20.171214 +/- 0.006617 Ha**  (exact 20.15932, err +0.059%)
- E (zero-variance extrap, 6 pts) = 20.128963 Ha  (err -0.151%)
- E (clipped est.) = 20.170252 Ha; var(E_L) = 9.5820e-02
- QGT/NTK: eff_rank = 3.61, kappa(S) = 3.239e+10, numerical rank = 767/89421 (alignment on 768 samples)
- alignment (final): cos(SR)=0.987, cos(plain)=0.063, NTK kappa=9.961e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
