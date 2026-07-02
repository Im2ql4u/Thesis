# Phase analysis: N=6, omega=1.0, arch=deepset_m

- params: 47,669
- **E (unclipped) = 20.165954 +/- 0.002960 Ha**  (exact 20.15932, err +0.033%)
- E (zero-variance extrap, 6 pts) = 20.189160 Ha  (err +0.148%)
- E (clipped est.) = 20.162129 Ha; var(E_L) = 3.0397e-02
- QGT/NTK: eff_rank = 4.35, kappa(S) = 3.360e+07, numerical rank = 767/47669 (alignment on 768 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.118, NTK kappa=3.360e+07

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
