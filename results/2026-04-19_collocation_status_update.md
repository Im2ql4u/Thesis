# Collocation Status Update

Date: 2026-04-19

Scope: weak-form collocation, non-MCMC results only.

This note supersedes the low-omega interpretation in
`results/2026-04-08_best-final-eval-report.md`.

## Reference correction first

The current source of truth is `src/config.py`. It now states:

- `N=2, ω=0.001` has no DMC reference; use thesis `PINN+CTNN` surrogate `0.013778`.
- `N=6, 12, 20`, `ω in {0.01, 0.001}` have no DMC references; use thesis `PINN+CTNN` surrogates.

That means several older low-omega `% err` values written inside historical logs are
numerically stale because those runs were logged against placeholder or outdated
reference values. In this update, every percentage below is recomputed against the
current `src/config.py` table when needed.

## Best current grid

| N | omega | Best E | Current ref | Ref type | Current err (%) | Model | Status | Best log |
|---:|---:|---:|---:|---|---:|---|---|---|
| 2 | 1.0 | 3.000055 | 3.000000 | DMC | +0.002 | Jastrow | solved | `outputs/higher_n/phase5_overnight_n2_n20_n12/n2ovr2_w1_20260407_233933.log` |
| 2 | 0.5 | 1.659755 | 1.659770 | DMC | -0.001 | Jastrow | solved | `outputs/higher_n/phase5_overnight_n2_n20_n12/n2ovr2_w05_20260407_233933.log` |
| 2 | 0.1 | 0.440808 | 0.440790 | DMC | +0.004 | BF+Jastrow | solved | `outputs/2026-03-21_1151_campaign_v7_n2_exact_n12_continue/logs/v7_n2w01_exact.log` |
| 2 | 0.01 | 0.073830 | 0.073839 | DMC | -0.014 | Jastrow | solved | `outputs/higher_n/phase5_overnight_n2_n20_n12/n2ovr2_w001_20260407_233933.log` |
| 2 | 0.001 | 0.013774 | 0.013778 | surrogate | -0.029 | Jastrow | strong surrogate match | `outputs/higher_n/phase5_overnight_n2_n20_n12/n2ovr2_w0001_20260407_233933.log` |
| 6 | 1.0 | 20.161193 | 20.159320 | DMC | +0.009 | BF+Jastrow | excellent | `outputs/2026-03-15_1909_natgrad_sweep/logs/adam_control_v1.log` |
| 6 | 0.5 | 11.784662 | 11.784840 | DMC | -0.002 | BF+Jastrow | excellent | `outputs/2026-03-17_0851_cascade_campaign/wave_1/logs/w1_n6w05_hisamp.log` |
| 6 | 0.1 | 3.556028 | 3.553850 | DMC | +0.061 | BF+Jastrow | strong | `outputs/tournament/phase2/t2_df_re_w01_20260405_121720.log` |
| 6 | 0.01 | 0.691286 | 0.690360 | surrogate | +0.134 | BF+Jastrow | strong | `outputs/consistency_campaign/phase5_probe/p5probe_ng_n6w001_s42_20260401_080353.log` |
| 6 | 0.001 | 0.140956 | 0.140832 | surrogate | +0.088 | BF+Jastrow | strong | `outputs/higher_n/phase4_n20_lowomega_escalation/n6x2_adam_w0001_20260407_083247.log` |
| 12 | 1.0 | 65.706011 | 65.700100 | DMC | +0.009 | BF+Jastrow | strong | `outputs/2026-03-21_1151_campaign_v7_n2_exact_n12_continue/logs/v7_n12w1_continue.log` |
| 12 | 0.5 | 39.169168 | 39.159600 | DMC | +0.024 | BF+Jastrow | strong | `outputs/2026-03-21_1920_campaign_v9_long24h/logs/v9_n12w05_polish_24h.log` |
| 12 | 0.1 | 12.282367 | 12.269840 | DMC | +0.102 | BF+Jastrow | strong | `outputs/2026-03-21_1151_campaign_v7_n2_exact_n12_continue/logs/v7_n12w01_continue.log` |
| 12 | 0.01 | 2.501093 | 2.473630 | surrogate | +1.110 | BF+Jastrow | boundary, not solved | `outputs/2026-03-20_2114_campaign_v5_10h_fix/logs/v5_n12w001_bf_cascade.log` |
| 12 | 0.001 | 0.888340 | 0.515365 | surrogate | +72.371 | BF+Jastrow | poor historical only | `outputs/2026-03-19_1622_n20_langevin_campaign/logs/camp2_n12w0001_cascade.log` |
| 20 | 1.0 | 157.283168 | 155.882200 | DMC | +0.899 | Jastrow | usable but not near-DMC | `outputs/higher_n/phase5_overnight_n2_n20_n12/n20ovr2_w1_20260407_233933.log` |
| 20 | 0.5 | 96.119158 | 93.875200 | DMC | +2.390 | Jastrow | usable but weak | `outputs/higher_n/phase5_overnight_n2_n20_n12/n20ovr2_w05_20260407_233933.log` |
| 20 | 0.1 | 31.636564 | 29.977900 | DMC | +5.533 | Jastrow | unresolved | `outputs/2026-03-18_0858_n20_w01_to_w001_transfer/logs/20260318_0858_n20w01_keep_a.log` |
| 20 | 0.01 | 12.295074 | 6.146450 | surrogate | +100.035 | Jastrow+ShellFlow proposal | blocked | `outputs/2026-04-17_shellflow_n20_overnight_curriculum_w0p01_v1/logs/n20_ess2_overnight_w0p01_s42.log` |
| 20 | 0.001 | — | 1.293033 | surrogate | — | — | no usable final non-MCMC result found | — |

## Overall address

### `N=2`

`N=2` is effectively done. The current bests are near-exact across the DMC-backed
grid, and even the `ω=0.001` surrogate cell now lines up with the corrected
reference to within `0.03%`.

### `N=6`

`N=6` remains the cleanest non-MCMC success story in the repository. High and
moderate omega are already thesis-quality, and the low-omega surrogate-backed
cells are still very close.

### `N=12`

`N=12` is strong for `ω >= 0.1`, but low omega is where the picture changes.
The important correction is that `ω=0.01` is not `+25%` against the current
reference table; it is `+1.11%`. That is much better than the old report
suggested, but still not in the same category as the high-omega cells.
At `ω=0.001`, the only historical final is still far off.

### `N=20`

`N=20` is still the real wall.

- For high omega, Jastrow-only continuation gives workable numbers:
  `+0.899%` at `ω=1.0`, `+2.390%` at `ω=0.5`.
- At `ω=0.1`, performance is clearly degraded but still physically connected.
- At `ω=0.01`, the recent ShellFlow work fixed proposal collapse much more than
  it fixed the energy. ESS recovery happened, but energies stayed around
  `+100%`.

So the present frontier is no longer only "can we stop the proposal from
collapsing?" It is also "is the current Jastrow-only wavefunction family capable
of representing the `N=20`, low-omega state once sampling is no longer totally
broken?"

## Exact protocol lineage

Below, "exact protocol" means the actual flags used for the winning run lineage,
not a reconstructed guess.

### Overnight launcher used for the latest `N=2` and `N=20` winners

From `scripts/launch_overnight_n2_n20_n12.sh`:

```bash
N2_BASE="--mode jastrow --n-elec 2 --seed 42 --loss-type reinforce --direct-weight 0 --clip-el 5.0 --n-coll 4096 --micro-batch 1024 --epochs 110000 --lr 6e-5 --lr-jas 6e-6 --lr-min-frac 0.01 --lr-warmup-epochs 30 --lr-warmup-init-frac 0.1 --patience 0 --vmc-every 1000 --vmc-n 15000 --save-best-window 30"
N20_BASE="--mode jastrow --n-elec 20 --seed 42 --loss-type reinforce --direct-weight 0 --clip-el 5.0 --n-coll 1024 --oversample 8 --micro-batch 128 --epochs 5600 --lr 8e-5 --lr-jas 8e-6 --lr-min-frac 0.01 --lr-warmup-epochs 30 --lr-warmup-init-frac 0.1 --patience 0 --vmc-every 500 --vmc-n 15000 --save-best-window 30"
N12_BASE="--mode bf --n-elec 12 --seed 42 --loss-type reinforce --direct-weight 0 --clip-el 5.0 --n-coll 2048 --oversample 8 --micro-batch 256 --epochs 5600 --lr 1.2e-4 --lr-jas 1.2e-5 --lr-min-frac 0.01 --lr-warmup-epochs 30 --lr-warmup-init-frac 0.1 --patience 0 --vmc-every 500 --vmc-n 15000 --save-best-window 30"
```

Best-cell overrides from that launcher:

- `N=2, ω=1.0`: `N2_BASE` + `--resume results/arch_colloc/smoke_n2_o1p0.pt --omega 1.0 --oversample 8 --tag n2ovr2_w1`
- `N=2, ω=0.5`: `N2_BASE` + `--resume results/arch_colloc/smoke_n2_o0p5.pt --omega 0.5 --oversample 8 --tag n2ovr2_w05`
- `N=2, ω=0.01`: `N2_BASE` + `--resume results/arch_colloc/smoke_n2_o0p01.pt --omega 0.01 --oversample 16 --tag n2ovr2_w001`
- `N=2, ω=0.001`: `N2_BASE` + `--resume results/arch_colloc/v16_n2w0001_transfer.pt --omega 0.001 --oversample 32 --tag n2ovr2_w0001`
- `N=20, ω=1.0`: `N20_BASE` + `--resume results/arch_colloc/n20x2_adam_w1_best.pt --omega 1.0 --tag n20ovr2_w1`
- `N=20, ω=0.5`: `N20_BASE` + `--resume results/arch_colloc/n20x2_adam_w05_best.pt --omega 0.5 --tag n20ovr2_w05`

### Exact historical winner commands outside the overnight launcher

`N=6, ω=1.0`, from `outputs/2026-03-15_1909_natgrad_sweep/logs/adam_control_v1.log`:

```bash
python run_weak_form.py --mode bf --n-elec 6 --omega 1.0 --epochs 500 --n-coll 4096 --oversample 8 --micro-batch 512 --grad-clip 1.0 --clip-el 5.0 --direct-weight 0.0 --vmc-every 40 --vmc-n 10000 --n-eval 20000 --seed 42 --resume /itf-fi-ml/home/aleksns/Thesis_repo/results/arch_colloc/bf_ctnn_vcycle.pt --tag adam_control_v1 --lr 5e-4 --lr-jas 5e-5
```

`N=6, ω=0.5`, from `outputs/2026-03-17_0851_cascade_campaign/wave_1/logs/w1_n6w05_hisamp.log`:

```bash
python run_weak_form.py --mode bf --n-elec 6 --omega 0.5 --epochs 600 --n-coll 4096 --oversample 16 --micro-batch 512 --natural-grad --sr-mode cg --lr 5e-3 --lr-jas 5e-4 --fisher-damping 1e-3 --fisher-damping-end 5e-5 --fisher-damping-anneal 300 --fisher-subsample 1024 --sr-cg-iters 20 --sr-max-param-change 0.05 --sr-trust-region 0.5 --nat-momentum 0.95 --grad-clip 0.5 --clip-el 3.0 --direct-weight 0.0 --vmc-every 30 --vmc-n 15000 --n-eval 50000 --seed 123 --resume /itf-fi-ml/home/aleksns/Thesis_repo/results/arch_colloc/camp_n6w05_fix.pt --tag w1_n6w05_hisamp
```

`N=6, ω=0.1`, from `scripts/launch_tournament_phase2.sh`:

```bash
python3 scripts/instrumented_run.py --tag t2_df_re_w01 --output-dir outputs/tournament/phase2 -- --mode bf --resume results/arch_colloc/p4m1_fdmatched_n6w01.pt --n-elec 6 --omega 0.1 --seed 42 --loss-type reinforce --direct-weight 0 --clip-el 5.0 --n-coll 8192 --oversample 8 --micro-batch 1024 --epochs 6000 --lr 2e-4 --lr-jas 2e-5 --lr-min-frac 0.01 --lr-warmup-epochs 30 --lr-warmup-init-frac 0.1 --natural-grad --sr-mode diagonal --fisher-damping 0.01 --fisher-ema 0.99 --fisher-probes 4 --fisher-subsample 2048 --nat-momentum 0.9 --patience 0 --vmc-every 500 --vmc-n 15000 --save-best-window 30
```

`N=6, ω=0.01`, from `scripts/launch_consistency_phase5_w001_probe.sh`:

```bash
python3 scripts/instrumented_run.py --tag p5probe_ng_n6w001_s42 --output-dir outputs/consistency_campaign/phase5_probe --run-weak-form-args --mode bf --resume results/arch_colloc/p4m2_nogate_n6w001.pt --n-elec 6 --omega 0.01 --seed 42 --loss-type reinforce --n-coll 8192 --oversample 64 --epochs 300 --lr 5e-4 --lr-jas 5e-5 --lr-warmup-epochs 60 --lr-min-frac 0.02 --micro-batch 1024 --patience 300 --vmc-every 50 --vmc-n 10000 --save-best-window 20
```

`N=6, ω=0.001`, from log header `outputs/higher_n/phase4_n20_lowomega_escalation/n6x2_adam_w0001_20260407_083247.log`:

```bash
run_weak_form.py --mode bf --resume results/arch_colloc/p4m2_nogate_n6w001_best.pt --n-elec 6 --omega 0.001 --seed 42 --epochs 10000 --n-coll 4096 --lr 8e-5 --lr-jas 8e-6 --loss-type reinforce --direct-weight 0 --clip-el 5.0 --micro-batch 1024 --lr-warmup-epochs 30 --lr-warmup-init-frac 0.1 --lr-min-frac 0.01
```

`N=2, ω=0.1` and `N=12, ω in {1.0, 0.1}`, from `scripts/campaign_v7_n2_exact_n12_continue.py`:

```bash
v7_n2w01_exact:
  --mode bf --n-elec 2 --omega 0.1 --e-dmc 0.44079 --epochs 3400 --n-coll 4096 --oversample 14 --micro-batch 1024 --lr 8e-5 --lr-jas 8e-5 --direct-weight 0.0 --clip-el 4.0 --reward-qtrim 0.01 --ess-floor-ratio 0.03 --ess-oversample-max 28 --ess-oversample-step 2 --ess-resample-tries 2 --rollback-decay 0.96 --rollback-err-pct 0.0 --rollback-jump-sigma 4.0 --vmc-every 60 --vmc-n 15000 --n-eval 60000 --seed 53 --resume results/arch_colloc/v4_n2w01_bf.pt --no-pretrained

v7_n12w1_continue:
  --mode bf --n-elec 12 --omega 1.0 --e-dmc 65.70010 --epochs 2600 --n-coll 4096 --oversample 10 --micro-batch 512 --lr 1e-4 --lr-jas 1e-5 --direct-weight 0.0 --clip-el 3.0 --reward-qtrim 0.01 --ess-floor-ratio 0.01 --ess-oversample-max 20 --ess-oversample-step 2 --ess-resample-tries 2 --rollback-decay 0.95 --rollback-err-pct 4.0 --rollback-jump-sigma 3.5 --vmc-every 40 --vmc-n 12000 --n-eval 40000 --seed 61 --resume results/arch_colloc/long_n12w1.pt

v7_n12w01_continue:
  --mode bf --n-elec 12 --omega 0.1 --e-dmc 12.26984 --epochs 3000 --n-coll 4096 --oversample 12 --micro-batch 512 --lr 8e-5 --lr-jas 8e-6 --direct-weight 0.0 --clip-el 4.0 --reward-qtrim 0.02 --ess-floor-ratio 0.03 --ess-oversample-max 24 --ess-oversample-step 2 --ess-resample-tries 2 --rollback-decay 0.95 --rollback-err-pct 0.0 --rollback-jump-sigma 4.0 --vmc-every 40 --vmc-n 12000 --n-eval 40000 --seed 62 --resume results/arch_colloc/20260318_1149_n12w01_keep.pt
```

`N=12, ω=0.5`, from `scripts/run_v9_n12_chain_24h.sh`:

```bash
python3 src/run_weak_form.py --mode bf --n-elec 12 --omega 0.5 --bf-hidden 64 --bf-msg-hidden 64 --bf-layers 2 --epochs 8000 --n-coll 4096 --oversample 16 --micro-batch 1024 --lr 4e-5 --lr-jas 4e-5 --direct-weight 0.0 --rollback-decay 0.97 --rollback-err-pct 0.0 --rollback-jump-sigma 4.5 --vmc-every 100 --vmc-n 25000 --n-eval 100000 --seed 202 --resume /itf-fi-ml/home/aleksns/Thesis_repo/results/curated_low_error_0p1pct_2026-03-21/long_n12w05.pt --no-pretrained --tag v9_n12w05_polish_24h
```

`N=12, ω=0.01`, from `scripts/campaign_v5_10h_fix.py`:

```bash
--mode bf --n-elec 12 --omega 0.01 --e-dmc 2.0 --epochs 10000 --n-coll 6144 --oversample 16 --micro-batch 512 --sigma-fs 0.8,1.3,2.0,3.5,6.0 --lr 1.5e-4 --lr-jas 1.5e-5 --clip-el 5.0 --reward-qtrim 0.02 --ess-floor-ratio 0.10 --ess-oversample-max 24 --ess-oversample-step 2 --ess-resample-tries 2 --rollback-decay 0.90 --rollback-err-pct 0.0 --rollback-jump-sigma 5.0 --vmc-every 80 --vmc-n 20000 --n-eval 60000 --seed 42 --init-jas results/arch_colloc/20260318_1149_n12w01_keep.pt --init-bf results/arch_colloc/20260318_1149_n12w01_keep.pt --no-pretrained --tag v5_n12w001_bf_cascade
```

`N=12, ω=0.001`, from `outputs/2026-03-19_1622_n20_langevin_campaign/logs/camp2_n12w0001_cascade.log`:

```bash
python run_weak_form.py --mode bf --n-elec 12 --omega 0.001 --epochs 3000 --n-coll 4096 --oversample 12 --micro-batch 512 --lr 4e-4 --lr-jas 4e-5 --direct-weight 0.0 --clip-el 5.0 --reward-qtrim 0.02 --ess-floor-ratio 0.10 --ess-oversample-max 18 --ess-oversample-step 2 --ess-resample-tries 2 --rollback-decay 0.85 --rollback-err-pct 20.0 --rollback-jump-sigma 8.0 --vmc-every 50 --vmc-n 8000 --n-eval 25000 --seed 42 --no-pretrained --tag camp2_n12w0001_cascade
```

`N=20, ω=0.1`, from `outputs/2026-03-18_0858_n20_w01_to_w001_transfer/logs/20260318_0858_n20w01_keep_a.log`:

```bash
python3 run_weak_form.py --mode jastrow --n-elec 20 --omega 0.1 --epochs 600 --n-coll 4096 --oversample 10 --micro-batch 512 --lr 5e-4 --lr-jas 5e-4 --direct-weight 0.1 --clip-el 0.0 --reward-qtrim 0.02 --vmc-every 40 --vmc-n 12000 --n-eval 30000 --seed 11 --resume /itf-fi-ml/home/aleksns/Thesis_repo/results/arch_colloc/camp_jastrow_transfer_stabilized_n20_o0p1_s11.pt --tag 20260318_0858_n20w01_keep_a
```

### Latest `N=20, ω=0.01` ShellFlow protocol

Best current low-omega `N=20` result is the `ess2` overnight run from
`scripts/launch_shellflow_n20_overnight_curriculum_w0p01.sh`. Its exact args were:

```bash
--mode jastrow --n-elec 20 --omega 0.01
--epochs 720 --lr 8e-5 --lr-jas 8e-6
--n-coll 2048 --oversample 128 --micro-batch 256
--loss-type reinforce --direct-weight 0.0 --clip-el 4.0 --grad-clip 1.0 --reward-qtrim 0.01
--rollback-decay 0.98 --rollback-err-pct 0.0 --rollback-jump-sigma 5.0
--adaptive-proposal --proposal-model shellflow
--ess-floor-ratio 0.01 --ess-oversample-max 256 --ess-oversample-step 32 --ess-resample-tries 6
--ess-update-floor 16
--resample-weight-temp 0.60 --resample-logw-clip-q 0.995
--top1-mass-update-ceiling 0.22 --top10-mass-update-ceiling 0.70
--shell-templates "20-0,1-19" --shell-mix-logits-init "0.0,0.0"
--shell-radii-init "0.0,1.35" --shell-sigmas "0.60,1.80"
--shell-refit-steps 160 --shell-refit-lr 1e-3
--shell-flow-layers 0 --shell-flow-hidden 128
--shell-ring-warmup-steps 0 --shell-radius-anchor-weight 0.0
--shell-curriculum-mode ess
--shell-curriculum-unlock-ess 24
--shell-curriculum-unlock-patience 2
--shell-curriculum-radius-quantile 0.85
--shell-curriculum-radius-threshold-aho 0
--shell-curriculum-inactive-logit-offset -12.0
--gmm-refit-every 10 --gmm-refit-min-samples 2048
--jas-input-radius-cap-aho 2.0
--vmc-every 60 --vmc-n 15000 --n-eval 20000
--seed 42 --tag shellflow_n20_ess2_overnight_w0p01_s42
--no-pretrained
```

## What the latest proposal sweeps actually say

From the April 18 all-proposals quick sweep:

| Variant | Final E | err (%) |
|---|---:|---:|
| `2shell` | 12.83966 | +108.895 |
| `epoch2` | 12.85901 | +109.210 |
| `ess2` | 12.85901 | +109.210 |
| `floor020` | 12.86243 | +109.266 |
| `3shell` | 12.86548 | +109.316 |
| `radius3` | 12.86876 | +109.369 |
| `floor010` | 12.92611 | +110.302 |

Centered-floor quick follow-up:

| Variant | Final E | err (%) |
|---|---:|---:|
| `floor035` | 12.84159 | +108.927 |
| `floor020` | 12.86243 | +109.266 |
| `floor010` | 12.92611 | +110.302 |

Interpretation:

- The proposal variants now look much closer to each other than before.
- That is evidence that proposal collapse is less dominant than it was in the
  earlier broken ShellFlow runs.
- The remaining error is probably not just a "pick the best proposal hyperparameter"
  problem.

## Brainstorm: what to try next

### 1. Smaller `N=20` network

Yes, this is worth trying, especially for high omega and for any attempt to bring
back backflow at `N=20`.

- CTNN parameter count is mostly `N`-independent.
- But compute and memory are not `N`-independent because message passing and
  pairwise structure still scale with the particle graph, effectively `O(N^2)`.
- So "CTNN is independent of `N`" is true for weights, but false for runtime and
  memory pressure.

Concretely, I would try:

- `bf-hidden 48`, `bf-msg-hidden 48`, `bf-layers 2`
- then `bf-hidden 32`, `bf-msg-hidden 32`, `bf-layers 2`
- and a Jastrow-hidden downsize as a control, just to see whether the current
  `N=20` failures are partly optimization-noise from an overlarge network

### 2. Transfer `N=12 -> 20` at low omega

Yes, I think this is one of the most plausible overlooked ideas.

- The correlation module is permutation-equivariant and not tied to a fixed `N`
  in parameter count.
- The determinant/orbital occupancy changes with `N`, so a full "just resume the
  whole wavefunction" transfer is not trivial.
- But Jastrow-only or partial CTNN transfer is plausible.

I would try a staged curriculum:

- `N=12, ω=0.1 -> 0.01 -> 0.001`
- transfer Jastrow weights to `N=16`
- solve `N=16` first
- then transfer `N=16 -> 20`

That is safer than one direct `12 -> 20` jump.

### 3. Bring back a tiny backflow at `N=20`

Your current evidence actually points the other way from "Jastrow alone is
sufficient at low omega." At `N=20`, Jastrow alone is what is currently
surviving high and moderate omega. Low omega is precisely where it still fails.

So I would test:

- Jastrow-only baseline
- tiny BF head on top of the same Jastrow
- freeze Jastrow for the first stage, train only tiny BF
- then unfreeze both

If that beats ShellFlow+Jastrow, we learn the bottleneck is ansatz capacity, not
proposal mechanics.

### 4. `N`-curriculum, not only `omega`-curriculum

This feels overlooked.

- We already rely on `omega` continuation.
- We have not really exploited `N` continuation as a first-class training axis.

I would explicitly test:

- `6 -> 12 -> 16 -> 20` at fixed `ω`
- especially for `ω=0.01`

### 5. Higher sample budget on the hard cells

The April 10 plan already flagged this, and I still think it is unfinished work.

- `--n-coll 8192 --oversample 16` for `N=12, ω=0.01`
- if feasible, the same for `N=20, ω=0.01`

Right now the hard low-omega runs are still underexplored in the high-sample regime.

### 6. One ShellFlow layer, not zero

The current stabilized ShellFlow runs all use:

```bash
--shell-flow-layers 0 --shell-flow-hidden 128
```

That means the proposal is still basically a structured mixture with refits, not
yet a learned transport map. If ESS is no longer catastrophically collapsing,
the next missed step may be to allow a very small learned deformation:

- `--shell-flow-layers 1 --shell-flow-hidden 32`
- keep every other stabilization trick unchanged

### 7. Separate center-of-mass from internal coordinates

At low omega and large `N`, the system becomes very extended. One thing we may be
overlooking is that the current parameterization still spends too much effort on
global drift and too little on relative geometry.

Two directions:

- center-of-mass-subtracted Jastrow inputs
- proposal templates expressed in internal shell radii around the instantaneous
  center of mass rather than the origin

### 8. Check whether `jas-input-radius-cap-aho 2.0` is too restrictive

This cap appears in the current ShellFlow scripts. It may be stabilizing, but it
may also be hiding exactly the tail information that matters most at low omega.

I would run a small ablation:

- `2.0`
- `3.0`
- uncapped

with all else fixed.

## Bottom line

- `N=2`: solved.
- `N=6`: strongest non-MCMC regime in the repo.
- `N=12`: strong at `ω >= 0.1`, borderline at `ω=0.01`, poor at `ω=0.001`.
- `N=20`: high omega partially works with Jastrow-only; low omega is still blocked.

The biggest conceptual shift from the April 8 report is that the low-omega
reference correction makes `N=12, ω=0.01` look much better than before, while
`N=20, ω=0.01` remains genuinely bad even after sampling stabilization.
