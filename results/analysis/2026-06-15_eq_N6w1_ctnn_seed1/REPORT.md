# Phase analysis: N=6, omega=1.0, arch=ctnn_vcycle_big

- params: 79,813
- **E (unclipped) = 20.163162 +/- 0.002689 Ha**  (exact 20.15932, err +0.019%)
- E (zero-variance extrap, 6 pts) = 20.163660 Ha  (err +0.022%)
- E (clipped est.) = 20.167411 Ha; var(E_L) = 2.5644e-02
- QGT/NTK: eff_rank = 1.46, kappa(S) = 6.936e+07, numerical rank = 767/79813 (alignment on 768 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=0.097, NTK kappa=6.936e+07

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
