# Best Final-Evaluation Results (Not Probe VMC)

Date: 2026-04-08

## Scope and Method

This report uses only **final evaluation lines** from completed training logs:

- Parsed pattern: `E = <value> ± <uncertainty>   err = <percent>%`
- Source search: all `outputs/**/*.log` excluding launcher/tmux wrapper logs
- Selection rule: for each `(N, omega)` in the allowed set `{1.0, 0.5, 0.1, 0.01, 0.001}`, choose the run with the smallest absolute final `% error`
- Probe VMC checkpoints are **not** used as final metrics in this table

## Best Historical Final-Eval Results (Allowed Omegas Only)

| N | omega | Best E | Uncertainty | err (%) | Best run log |
|---:|---:|---:|---:|---:|---|
| 2 | 0.001 | 0.013775 | 0.000001 | +88.705 | [outputs/2026-03-21_1151_campaign_v7_n2_exact_n12_continue/logs/v8_n2w0001_finetune.log](outputs/2026-03-21_1151_campaign_v7_n2_exact_n12_continue/logs/v8_n2w0001_finetune.log) |
| 2 | 0.01 | 0.073864 | 0.000016 | +0.032 | [outputs/2026-03-21_1151_campaign_v7_n2_exact_n12_continue/logs/v7_n2w001_exact.log](outputs/2026-03-21_1151_campaign_v7_n2_exact_n12_continue/logs/v7_n2w001_exact.log) |
| 2 | 0.1 | 0.440808 | 0.000153 | +0.004 | [outputs/2026-03-21_1151_campaign_v7_n2_exact_n12_continue/logs/v7_n2w01_exact.log](outputs/2026-03-21_1151_campaign_v7_n2_exact_n12_continue/logs/v7_n2w01_exact.log) |
| 2 | 0.5 | 1.659755 | 0.000224 | -0.001 | [outputs/higher_n/phase5_overnight_n2_n20_n12/n2ovr2_w05_20260407_233933.log](outputs/higher_n/phase5_overnight_n2_n20_n12/n2ovr2_w05_20260407_233933.log) |
| 2 | 1.0 | 3.000055 | 0.000354 | +0.002 | [outputs/higher_n/phase5_overnight_n2_n20_n12/n2ovr2_w1_20260407_233933.log](outputs/higher_n/phase5_overnight_n2_n20_n12/n2ovr2_w1_20260407_233933.log) |
| 6 | 0.001 | 0.140956 | 0.000005 | +0.088 | [outputs/higher_n/phase4_n20_lowomega_escalation/n6x2_adam_w0001_20260407_083247.log](outputs/higher_n/phase4_n20_lowomega_escalation/n6x2_adam_w0001_20260407_083247.log) |
| 6 | 0.01 | 0.691286 | 0.000086 | +0.134 | [outputs/consistency_campaign/phase5_probe/p5probe_ng_n6w001_s42_20260401_080353.log](outputs/consistency_campaign/phase5_probe/p5probe_ng_n6w001_s42_20260401_080353.log) |
| 6 | 0.1 | 3.556028 | 0.000324 | +0.061 | [outputs/tournament/phase2/t2_df_re_w01_20260405_121720.log](outputs/tournament/phase2/t2_df_re_w01_20260405_121720.log) |
| 6 | 0.5 | 11.784662 | 0.001987 | -0.002 | [outputs/2026-03-17_0851_cascade_campaign/wave_1/logs/w1_n6w05_hisamp.log](outputs/2026-03-17_0851_cascade_campaign/wave_1/logs/w1_n6w05_hisamp.log) |
| 6 | 1.0 | 20.161193 | 0.004827 | +0.009 | [outputs/2026-03-15_1909_natgrad_sweep/logs/adam_control_v1.log](outputs/2026-03-15_1909_natgrad_sweep/logs/adam_control_v1.log) |
| 12 | 0.01 | 2.501093 | 0.000262 | +25.055 | [outputs/2026-03-20_2114_campaign_v5_10h_fix/logs/v5_n12w001_bf_cascade.log](outputs/2026-03-20_2114_campaign_v5_10h_fix/logs/v5_n12w001_bf_cascade.log) |
| 12 | 0.1 | 12.282367 | 0.000637 | +0.102 | [outputs/2026-03-21_1151_campaign_v7_n2_exact_n12_continue/logs/v7_n12w01_continue.log](outputs/2026-03-21_1151_campaign_v7_n2_exact_n12_continue/logs/v7_n12w01_continue.log) |
| 12 | 0.5 | 39.169168 | 0.001872 | +0.024 | [outputs/2026-03-21_1920_campaign_v9_long24h/logs/v9_n12w05_polish_24h.log](outputs/2026-03-21_1920_campaign_v9_long24h/logs/v9_n12w05_polish_24h.log) |
| 12 | 1.0 | 65.706011 | 0.004489 | +0.009 | [outputs/2026-03-21_1151_campaign_v7_n2_exact_n12_continue/logs/v7_n12w1_continue.log](outputs/2026-03-21_1151_campaign_v7_n2_exact_n12_continue/logs/v7_n12w1_continue.log) |
| 20 | 0.1 | 31.636564 | 0.007926 | +5.533 | [outputs/2026-03-18_0858_n20_w01_to_w001_transfer/logs/20260318_0858_n20w01_keep_a.log](outputs/2026-03-18_0858_n20_w01_to_w001_transfer/logs/20260318_0858_n20w01_keep_a.log) |
| 20 | 0.5 | 96.446917 | 0.015192 | +2.740 | [outputs/2026-03-19_1909_campaign_v3/logs/v3_n20w05_polish.log](outputs/2026-03-19_1909_campaign_v3/logs/v3_n20w05_polish.log) |
| 20 | 1.0 | 157.934280 | 0.018020 | +1.316 | [outputs/higher_n/phase4_n20_lowomega_escalation/n20x2_adam_w1_20260407_083247.log](outputs/higher_n/phase4_n20_lowomega_escalation/n20x2_adam_w1_20260407_083247.log) |

## What Was Done to Get These

- N=6 low-omega and omega=0.1 bests came from optimizer tournament and continuation campaigns (Adam and DiagFisher variants)
- N=12 bests at omega in {0.1, 0.5, 1.0} came from continuation/polish chains started from earlier N=12 checkpoints
- N=20 bests came from long jastrow-only continuation lines, with best current values at omega=1.0 and 0.5 from Adam continuation
- N=2 bests at omega in {0.5, 1.0} were improved in the latest overnight continuation; omega=0.01 and 0.1 were already very strong historically

## Were These Lucky?

Short answer: partly unknown due to seed coverage.

- Likely more robust (less likely pure luck):
  - Regimes where multiple campaigns converge to similarly low errors (e.g., N=12 at omega=1.0/0.5, N=20 at omega=1.0 trending down through multiple Adam continuations)
- Potentially luck-sensitive:
  - Any regime currently supported by only one seed/run lineage without multi-seed revalidation
  - N=2 omega=0.5 and 1.0 latest improvements, until replicated with additional seeds
- Clearly unresolved structurally:
  - N=20 omega=0.1 (still far from DMC)
  - N=12 omega=0.01 and N=2 omega=0.001 (large persistent errors)

## Overnight Campaign Status (Strict Omega Set)

Session: [outputs/higher_n/phase5_overnight_tmux_20260407_233929.log](outputs/higher_n/phase5_overnight_tmux_20260407_233929.log)

As of latest check:
- Completed: `n2ovr2_w01`, `n2ovr2_w05`, `n2ovr2_w1`
- Still running: `n2ovr2_w0001`, `n2ovr2_w001`, `n20ovr2_w1`, `n20ovr2_w05`, `n12ovr2_w01`

Latest in-run ETA snapshot showed:
- `n2ovr2_w001` ~43 min remaining
- `n2ovr2_w0001` ~184 min remaining
- `n12ovr2_w01` ~82 min remaining
- `n20ovr2_w1` and `n20ovr2_w05` ~109 min remaining

So: not finished yet, but partially complete.
