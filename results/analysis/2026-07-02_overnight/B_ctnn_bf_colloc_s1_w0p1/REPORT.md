# Phase analysis: N=6, omega=0.1, arch=ctnn_vcycle_big

- params: 79,813
- **E (unclipped) = 3.601182 +/- 0.001901 Ha**  (exact 3.55385, err +1.332%)
- E (zero-variance extrap, 5 pts) = 3.569902 Ha  (err +0.452%)
- E (clipped est.) = 3.599895 Ha; var(E_L) = 1.9985e-02
- QGT/NTK: eff_rank = 1.52, kappa(S) = 9.959e+11, numerical rank = 1407/79813 (alignment on 2048 samples)
- alignment (final): cos(SR)=0.986, cos(plain)=0.111, NTK kappa=9.902e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
