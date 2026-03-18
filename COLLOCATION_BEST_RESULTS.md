# Collocation Learning Best Results (to date)

- Generated: 2026-03-18 14:33:50
- Source files scanned: 20
- Parsed valid result entries: 24

## Why Adam + ESS Often Helps at Low Omega

- Low omega broadens the wavefunction and amplifies importance-weight variance, so naive updates become dominated by a few high-weight samples.
- ESS control acts as an online guardrail against this weight-collapse regime by forcing better sample diversity before updates are trusted.
- Adam's per-parameter adaptive scaling is robust when gradient statistics are heteroscedastic (common in low-omega tails and near-cusp events).
- SR/NG steps can still win when Fisher estimates are stable, but low-ESS Fisher estimates are noisy; in that regime Adam + ESS is usually more forgiving.

## Best Per Target (Lowest Absolute Percent Error)

| N | omega | E | err (%) | tag | campaign | source |
|---:|---:|---:|---:|---|---|---|
| 6 | 0.001 | 0.141328 | +0.352 | regime_low_L_n6_o0p001_s11 | 2026-03-15_1122_regime_policy_campaign_tmux | outputs/2026-03-15_1122_regime_policy_campaign_tmux/summary.json |
| 6 | 0.01 | 0.692042 | +0.244 | regime_low_L_n6_o0p01_s11 | 2026-03-15_1122_regime_policy_campaign_tmux | outputs/2026-03-15_1122_regime_policy_campaign_tmux/summary.json |
| 6 | 0.1 | 3.557090 | +0.091 | 20260318_0858_n6w01_keep | 2026-03-18_0858_n20_w01_to_w001_transfer | outputs/2026-03-18_0858_n20_w01_to_w001_transfer/results.json |
| 6 | 0.5 | 11.789120 | +0.036 | long_n6w05 | 2026-03-17_1816_long_campaign | outputs/2026-03-17_1816_long_campaign/results.json |
| 6 | 1.0 | 20.165490 | +0.031 | long_n6w1 | 2026-03-17_1816_long_campaign | outputs/2026-03-17_1816_long_campaign/results.json |
| 12 | 0.1 | 12.284870 | +0.122 | 20260318_1149_n12w01_keep | 2026-03-18_1149_n20_w01_to_w001_transfer | outputs/2026-03-18_1149_n20_w01_to_w001_transfer/results.json |
| 12 | 0.5 | 39.170600 | +0.028 | long_n12w05 | 2026-03-17_1816_long_campaign | outputs/2026-03-17_1816_long_campaign/results.json |
| 12 | 1.0 | 65.712080 | +0.018 | long_n12w1 | 2026-03-17_1816_long_campaign | outputs/2026-03-17_1816_long_campaign/results.json |
| 20 | 0.1 | 31.636560 | +5.533 | 20260318_0858_n20w01_keep_a | 2026-03-18_0858_n20_w01_to_w001_transfer | outputs/2026-03-18_0858_n20_w01_to_w001_transfer/results.json |
| 20 | 1.0 | 210.984671 | +35.349 | regime_high_H_n20_o1p0_s11 | 2026-03-15_1122_regime_policy_campaign_tmux | outputs/2026-03-15_1122_regime_policy_campaign_tmux/summary.json |

## Global Top 20 Runs by Absolute Error

| rank | N | omega | E | err (%) | tag | campaign |
|---:|---:|---:|---:|---:|---|---|
| 1 | 12 | 1.0 | 65.712080 | +0.018 | long_n12w1 | 2026-03-17_1816_long_campaign |
| 2 | 12 | 0.5 | 39.170600 | +0.028 | long_n12w05 | 2026-03-17_1816_long_campaign |
| 3 | 6 | 1.0 | 20.165490 | +0.031 | long_n6w1 | 2026-03-17_1816_long_campaign |
| 4 | 6 | 0.5 | 11.789120 | +0.036 | long_n6w05 | 2026-03-17_1816_long_campaign |
| 5 | 6 | 1.0 | 20.167474 | +0.040 | regime_high_H_n6_o1p0_s11 | 2026-03-15_1122_regime_policy_campaign_tmux |
| 6 | 6 | 0.1 | 3.557090 | +0.091 | 20260318_0858_n6w01_keep | 2026-03-18_0858_n20_w01_to_w001_transfer |
| 7 | 6 | 0.1 | 3.557090 | +0.091 | 20260318_1149_n6w01_keep | 2026-03-18_1149_n20_w01_to_w001_transfer |
| 8 | 12 | 0.1 | 12.284870 | +0.122 | 20260318_1149_n12w01_keep | 2026-03-18_1149_n20_w01_to_w001_transfer |
| 9 | 12 | 0.1 | 12.288870 | +0.155 | long_n12w01_v3 | 2026-03-17_1816_long_campaign |
| 10 | 6 | 0.1 | 3.560320 | +0.182 | long_n6w01 | 2026-03-17_1816_long_campaign |
| 11 | 6 | 0.01 | 0.692042 | +0.244 | regime_low_L_n6_o0p01_s11 | 2026-03-15_1122_regime_policy_campaign_tmux |
| 12 | 12 | 0.5 | 39.282875 | +0.315 | regime_high_H_n12_o0p5_s11 | 2026-03-15_1122_regime_policy_campaign_tmux |
| 13 | 12 | 0.1 | 12.310430 | +0.331 | long_n12w01 | 2026-03-17_1816_long_campaign |
| 14 | 6 | 0.001 | 0.141328 | +0.352 | regime_low_L_n6_o0p001_s11 | 2026-03-15_1122_regime_policy_campaign_tmux |
| 15 | 12 | 0.1 | 12.407560 | +1.122 | long_n12w01_v2 | 2026-03-17_1816_long_campaign |
| 16 | 6 | 0.01 | 0.718780 | +4.116 | long_n6w001_v2 | 2026-03-17_1816_long_campaign |
| 17 | 20 | 0.1 | 31.636560 | +5.533 | 20260318_0858_n20w01_keep_a | 2026-03-18_0858_n20_w01_to_w001_transfer |
| 18 | 20 | 0.1 | 31.694020 | +5.725 | 20260318_0858_n20w01_keep_b | 2026-03-18_0858_n20_w01_to_w001_transfer |
| 19 | 6 | 0.01 | 0.737550 | +6.836 | long_n6w001 | 2026-03-17_1816_long_campaign |
| 20 | 6 | 0.01 | 0.753470 | +9.142 | long_n6w001_v3 | 2026-03-17_1816_long_campaign |

## Notes

- This report compares final reported errors from campaign summaries/results artifacts only.
- Runs with non-zero return code are excluded from best-table selection.
- If a target has no known DMC reference, err-based ranking is unavailable for that target unless the run supplied a manual reference.