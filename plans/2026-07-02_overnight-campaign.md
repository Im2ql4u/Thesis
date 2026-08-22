# Overnight Campaign — 2026-07-02 (Grand Mechanism Program, first big run)

**Goal:** produce a matched set of trained checkpoints spanning {architecture} × {backflow} × {paradigm}
× {seed} × {ω-cascade} (and N=12), so the post-run mechanism analysis answers, in one sweep: what is
message passing worth (confound-free), does the *paradigm* shape internal structure (the backflow-rank
thread), the SR-vs-Adam-on-collocation question, and the N-scaling. 8 GPUs, ~8 h.

**Launch:** `setsid python3 -u scripts/orchestrate_overnight.py --gpus 0-7 --include-n12 > <log> 2>&1 &`
(colloc+SR EXCLUDED — its N=2 smoke did not converge; WIP.)

## What runs (each job = an ω-cascade chain 1→0.1→0.01, warm-started, one GPU)

| Group | chains | purpose |
|---|---|---|
| **A** architecture ablation (VMC, adam+SR-polish) | {ctnn,deepset}×{bf,nobf}×2 seeds = **8** | *what is MP worth* — the confound-free 2×2 (Jastrow-MP × backflow-MP); the (deepset,nobf) arm is the pure separable FFNN |
| **B** paradigm = collocation | ctnn+bf collocation × 2 seeds = **2** | *does the paradigm shape internal structure* — compare backflow rank / kinetic / d_eff to the VMC ctnn+bf (Group A). The rank-1-vs-rank-10 backflow thread. |
| **C** optimizer×paradigm | ctnn+bf vmc-adam × 1 = **1** | with A's vmc-sr + B's colloc-adam → 3 of the 4 Phase-O quadrants (colloc-sr deferred, WIP) |
| **D** scaling (collocation, no Laplacian OOM) | N=12 {ctnn,deepset}+bf colloc, ω{1,0.1} = **2** | *trained-state* N-scaling of the mechanism; validated OOM-free (ESS ~0.42) |

Total ~13 chains. On 8 GPUs: ~2 rounds. Each N=6 chain ≈ 100–140 min; N=12 chain ≈ 60–90 min.

## Recipes
- **VMC** (A): `--paradigm vmc` steps 500 / n-seg 4 / polish 120 / sr-polish 200 (adam warm + annealed SR polish → ~DMC).
- **VMC-adam** (C): sr-polish 0 (pure Adam, for the optimizer contrast).
- **Collocation** (B, D): `--paradigm colloc` steps 1400 / polish 300 (weak-form importance-sampled; more steps).
- **N=12** (D): chunked exact-Laplacian eval + eval/align 384, final 512, batch 512 (tested OOM-free).

## Pre-launch validation (all passed)
- `--paradigm colloc` @ N=6+backflow: trains, ESS ~0.35, saves loadable checkpoint. ✓
- N=12 collocation: no OOM, ESS ~0.42, full pipeline saves. ✓
- Orchestrator mechanics (subprocess, --init cascade chaining, GPU queue, error isolation): smoke ✓.
- Robustness: per-chain try/except isolates failures; a bad chain stops itself, not the night.

## Post-run (tomorrow)
`python3 scripts/analyze_overnight.py` → `master.csv` (energy, var, d_eff, backflow-rank,
message-ablation kinetic dT per checkpoint). Then the decisive slices:
1. **What MP is worth** (Group A): energy + kinetic decomposition across {ctnn,deepset}×{bf,nobf}.
2. **Paradigm × structure** (A-ctnn+bf VMC vs B collocation): backflow rank, d_eff, kinetic at matched energy.
3. **Phase O** (A vmc-sr, B colloc-adam, C vmc-adam): optimizer×paradigm energy/var.
4. **Scaling** (D): does the compression gap / mechanism hold at N=12.

## Deferred (flagged, not tonight)
- collocation+SR (natural-gradient collocation) — WIP, does not converge; debug the weighted Woodbury step.
- N=12 ω=0.01 (hardest) and N=20 — after N=12 ω∈{1,0.1} validates overnight.
- The M0 leftovers (T1.3 edge-scalar decode, T1.7 force/attribution at N=12) — analysis, next session.
