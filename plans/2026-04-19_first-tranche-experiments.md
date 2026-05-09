# First Tranche Experiment Queue

Date: 2026-04-19
Status: ready to launch

This is the first runnable batch derived from
`results/2026-04-19_collocation_status_update.md`.

The goal is not to cover every brainstorm branch at once. The goal is to answer
the first three bottleneck questions with the smallest batch that can still move
the frontier:

1. Is `N=20` low-omega still mostly a proposal problem, or is ansatz capacity now
   the larger limiter?
2. Is `N=12, ω=0.01` actually blocked by sample budget?
3. Can a tiny backflow recover useful accuracy at `N=20` without exploding memory?

## Ranked queue

### Priority 1: `N=20` tiny-BF reentry

Script: `scripts/launch_n20_tinybf_reentry.sh`

Question:
- Does a very small BF head improve the current `N=20` Jastrow baselines at
  `ω in {1.0, 0.5, 0.1}`?

Why first:
- If tiny BF helps, the current low-omega wall is at least partly ansatz-limited.
- If tiny BF does not help even at `ω=1.0` and `0.5`, then reintroducing BF at
  `N=20` is probably not the fastest route.

Runs:
- 6 runs total
- 2 BF sizes (`48`, `32`) across `ω = 1.0, 0.5, 0.1`

Expected upside:
- High for diagnosis
- Moderate for immediate accuracy at `ω=1.0`
- Lower but still meaningful at `ω=0.1`

Estimated wall-clock:
- About 6 to 10 hours on 6 GPUs, depending on queue load and throughput

### Priority 2: `N=12, ω=0.01` high-sample ablation

Script: `scripts/launch_n12_lowomega_hisamp_v2.sh`

Question:
- Does higher candidate budget unlock the `N=12, ω=0.01` cell?
- Is the current `+1.11%` gap mostly a sampling problem?

Why second:
- This is the cheapest path to a likely near-term scientific win.
- `N=12, ω=0.01` is already close enough that sample-budget changes could matter.

Runs:
- 4 runs total
- `cap=2`, `cap=3`, `uncapped`, and one heavier `oversample=24` variant

Expected upside:
- High
- This is the single most plausible route to turning `N=12, ω=0.01` from
  "boundary" into "strong".

Estimated wall-clock:
- About 4 to 7 hours on 4 GPUs

### Priority 3: `N=20, ω=0.01` ShellFlow gap ablation

Script: `scripts/launch_shellflow_n20_ess2_gap_ablation_w0p01.sh`

Question:
- Have we been suppressing exactly the tail geometry we need via
  `--jas-input-radius-cap-aho 2.0`?
- Does a single learned ShellFlow layer help once the proposal is no longer in
  the catastrophic-collapse regime?

Why third:
- The recent proposal sweeps clustered tightly around `+109%`.
- That makes this the right time to test the remaining high-leverage proposal
  blind spots directly.

Runs:
- 4 runs total
- `cap=3`, uncapped, `1-layer+cap3`, `1-layer+uncapped`

Expected upside:
- Moderate for immediate accuracy
- High for deciding whether more ShellFlow investment is warranted

Estimated wall-clock:
- About 2 to 4 hours on 4 GPUs

## Deferred for tranche 2

These are still good ideas, but they are intentionally deferred until the first
batch answers the sampling-vs-ansatz split more clearly:

- `N`-curriculum `12 -> 16 -> 20`
- center-of-mass-subtracted inputs
- explicit radial tail-guard ablations
- multi-seed reliability sweeps of any newly improved regime

## One implementation note

The existing script `scripts/launch_shellflow_n20_layer1_curriculum_w0p01.sh`
is historically useful but not safe to reuse as a "real layer-1" launcher:
despite its name, it currently passes `--shell-flow-layers 0`.

To avoid changing the provenance of old runs, the new tranche uses a fresh
launcher with explicit `--shell-flow-layers 1` variants instead of editing the
old historical script.
