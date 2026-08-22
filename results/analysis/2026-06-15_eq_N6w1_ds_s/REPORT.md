# Phase analysis: N=6, omega=1.0, arch=deepset_s

- params: 20,189
- **E (unclipped) = 20.171039 +/- 0.007465 Ha**  (exact 20.15932, err +0.058%)
- E (zero-variance extrap, 6 pts) = 20.138712 Ha  (err -0.102%)
- E (clipped est.) = 20.171042 Ha; var(E_L) = 9.0389e-02
- QGT/NTK: eff_rank = 3.56, kappa(S) = 3.240e+09, numerical rank = 767/20189 (alignment on 768 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.075, NTK kappa=3.240e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
