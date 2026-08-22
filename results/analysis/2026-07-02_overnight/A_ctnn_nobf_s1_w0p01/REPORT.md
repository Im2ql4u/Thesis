# Phase analysis: N=6, omega=0.01, arch=ctnn_vcycle_big

- params: 66,498
- **E (unclipped) = -31041466977512872494269950099854978182660992911342000668404866675853854456032488635796752059967928795136.000000 +/- 630601281979842911519088165676201208288715025960553805623929732203115942278082986722175061833805725696.000000 Ha**  (exact 0.69036, err -4496417373183972807941229625535728817883978924007831534420731760105292688798753781763493716885266535284736.000%)
- E (zero-variance extrap, 5 pts) = -0.000000 Ha  (err -100.000%)
- E (clipped est.) = -31041466977512872494269950099854978182660992911342000668404866675853854456032488635796752059967928795136.000000 Ha; var(E_L) = 2.2516e+205
- QGT/NTK: eff_rank = 1.00, kappa(S) = 8.503e+11, numerical rank = 26/66498 (alignment on 77 samples)
- alignment (final): cos(SR)=1.000, cos(plain)=nan, NTK kappa=7.395e+09

## Data files (all plot inputs saved)
- plot_data.npz : every array behind every figure
- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, data_ntk_whitening.csv, data_jastrow.csv
- summary.json : all scalar metrics; checkpoint.pt : trained weights

Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening.
