# Phase analysis: N=6, omega=1.0, arch=ctnn_vcycle_big

- params: 79,813
- **E (unclipped) = 20.161993 +/- 0.002421 Ha**  (exact 20.15932, err +0.013%)
- E (zero-variance extrap, 5 pts) = 20.154613 Ha  (err -0.023%)
- E (clipped est.) = 20.168041 Ha; var(E_L) = 6.2649e-02
- QGT/NTK: eff_rank = 1.19, kappa(S) = 2.342e+10, numerical rank = 2047/79813 (alignment on 2048 samples)
- alignment (final): cos(SR)=0.980, cos(plain)=0.042, NTK kappa=9.895e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
