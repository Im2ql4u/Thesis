# Collocation Learning Best Results (Corrected)

- Generated: 2026-03-18 14:39:54
- Source files scanned: 20
- Parsed valid records: 55
- Ranking criterion: lowest absolute percent error using best-observed metric when available (`best_probe_err_pct` or `best_err`), otherwise final error.

## Why Adam + ESS Is Often Better At Low Omega

- Low omega stretches support and increases importance-weight skew; this makes minibatch estimates high-variance and heavy-tail dominated.
- ESS controllers actively reject low-diversity weighted batches and increase effective sample quality before taking optimizer steps.
- Adam's adaptive moments damp per-parameter gradient heteroscedasticity that is common when tails/cusp regions intermittently dominate.
- SR/Natural gradient can outperform when Fisher estimates are stable, but in low-ESS regimes Fisher noise can make updates brittle.

## Best Per (N, omega)

| N | omega | E | err (%) | method | tag | campaign | source |
|---:|---:|---:|---:|---|---|---|---|
| 6 | 0.001 | 0.141302 | +0.334 | best_probe | regime_low_L_n6_o0p001_s11 | 2026-03-15_1122_regime_policy_campaign_tmux | outputs/2026-03-15_1122_regime_policy_campaign_tmux/summary.json |
| 6 | 0.01 | 0.691693 | +0.193 | best_probe | regime_low_L_n6_o0p01_s11 | 2026-03-15_1122_regime_policy_campaign_tmux | outputs/2026-03-15_1122_regime_policy_campaign_tmux/summary.json |
| 6 | 0.1 | 3.557090 | +0.091 | final | 20260318_0858_n6w01_keep | 2026-03-18_0858_n20_w01_to_w001_transfer | outputs/2026-03-18_0858_n20_w01_to_w001_transfer/results.json |
| 6 | 0.5 | 11.784560 | +0.002 | best | sr_n6w01_v2 | 2026-03-16_1959_sr_gen_v2 | outputs/2026-03-16_1959_sr_gen_v2/summary.json |
| 6 | 1.0 | 20.159705 | +0.002 | best_probe | regime_high_H_n6_o1p0_s11 | 2026-03-15_1122_regime_policy_campaign_tmux | outputs/2026-03-15_1122_regime_policy_campaign_tmux/summary.json |
| 12 | 0.1 | 12.284870 | +0.122 | final | 20260318_1149_n12w01_keep | 2026-03-18_1149_n20_w01_to_w001_transfer | outputs/2026-03-18_1149_n20_w01_to_w001_transfer/results.json |
| 12 | 0.5 | 39.170600 | +0.028 | final | long_n12w05 | 2026-03-17_1816_long_campaign | outputs/2026-03-17_1816_long_campaign/results.json |
| 12 | 1.0 | 65.712080 | +0.018 | final | long_n12w1 | 2026-03-17_1816_long_campaign | outputs/2026-03-17_1816_long_campaign/results.json |
| 20 | 0.1 | 31.636560 | +5.533 | final | 20260318_0858_n20w01_keep_a | 2026-03-18_0858_n20_w01_to_w001_transfer | outputs/2026-03-18_0858_n20_w01_to_w001_transfer/results.json |
| 20 | 1.0 | 207.149700 | +32.889 | best | camp_n20w1_smoke | phase1_verify | outputs/2026-03-17_0031_cgsr_campaign/phase1_verify/summary.json |

## Global Top 25 By |err|

| rank | N | omega | E | err (%) | method | tag | campaign |
|---:|---:|---:|---:|---:|---|---|---|
| 1 | 6 | 1.0 | 20.159705 | +0.002 | best_probe | regime_high_H_n6_o1p0_s11 | 2026-03-15_1122_regime_policy_campaign_tmux |
| 2 | 6 | 0.5 | 11.784560 | +0.002 | best | sr_n6w01_v2 | 2026-03-16_1959_sr_gen_v2 |
| 3 | 6 | 1.0 | 20.160180 | +0.004 | best | camp_n6w1_verify | phase1_verify |
| 4 | 6 | 1.0 | 20.160800 | +0.007 | best | sr_n6w01_v2 | 2026-03-16_1959_sr_gen_v2 |
| 5 | 6 | 1.0 | 20.161670 | +0.012 | final | camp_n6w1_verify | phase1_verify |
| 6 | 12 | 1.0 | 65.712080 | +0.018 | final | long_n12w1 | 2026-03-17_1816_long_campaign |
| 7 | 12 | 0.5 | 39.170600 | +0.028 | final | long_n12w05 | 2026-03-17_1816_long_campaign |
| 8 | 6 | 1.0 | 20.165490 | +0.031 | final | long_n6w1 | 2026-03-17_1816_long_campaign |
| 9 | 6 | 0.5 | 11.789120 | +0.036 | final | long_n6w05 | 2026-03-17_1816_long_campaign |
| 10 | 6 | 1.0 | 20.167474 | +0.040 | final | regime_high_H_n6_o1p0_s11 | 2026-03-15_1122_regime_policy_campaign_tmux |
| 11 | 6 | 1.0 | 20.172800 | +0.067 | final | sr_n6w01_v2 | 2026-03-16_1959_sr_gen_v2 |
| 12 | 6 | 0.1 | 3.557090 | +0.091 | final | 20260318_0858_n6w01_keep | 2026-03-18_0858_n20_w01_to_w001_transfer |
| 13 | 6 | 0.1 | 3.557090 | +0.091 | final | 20260318_1149_n6w01_keep | 2026-03-18_1149_n20_w01_to_w001_transfer |
| 14 | 12 | 0.1 | 12.284870 | +0.122 | final | 20260318_1149_n12w01_keep | 2026-03-18_1149_n20_w01_to_w001_transfer |
| 15 | 6 | 0.5 | 11.800520 | +0.133 | best | camp_n6w05_fix | phase1_verify |
| 16 | 12 | 0.1 | 12.288870 | +0.155 | final | long_n12w01_v3 | 2026-03-17_1816_long_campaign |
| 17 | 6 | 0.5 | 11.803670 | +0.160 | final | sr_n6w01_v2 | 2026-03-16_1959_sr_gen_v2 |
| 18 | 6 | 0.5 | 11.804400 | +0.166 | final | camp_n6w05_fix | phase1_verify |
| 19 | 6 | 0.1 | 3.560320 | +0.182 | final | long_n6w01 | 2026-03-17_1816_long_campaign |
| 20 | 12 | 0.5 | 39.232297 | +0.186 | best_probe | regime_high_H_n12_o0p5_s11 | 2026-03-15_1122_regime_policy_campaign_tmux |
| 21 | 6 | 0.01 | 0.691693 | +0.193 | best_probe | regime_low_L_n6_o0p01_s11 | 2026-03-15_1122_regime_policy_campaign_tmux |
| 22 | 6 | 0.01 | 0.692042 | +0.244 | final | regime_low_L_n6_o0p01_s11 | 2026-03-15_1122_regime_policy_campaign_tmux |
| 23 | 6 | 0.5 | 11.816950 | +0.272 | final | sr_n6w01_v2 | 2026-03-16_1959_sr_gen_v2 |
| 24 | 6 | 0.5 | 11.818570 | +0.286 | best | sr_n6w01_v2 | 2026-03-16_1959_sr_gen_v2 |
| 25 | 12 | 0.5 | 39.282875 | +0.315 | final | regime_high_H_n12_o0p5_s11 | 2026-03-15_1122_regime_policy_campaign_tmux |