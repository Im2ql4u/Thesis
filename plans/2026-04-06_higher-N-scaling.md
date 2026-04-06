# Plan: Higher-N Scaling — DiagFisher+REINFORCE at N=12 and N=20

Date: 2026-04-06
Status: confirmed

## Project objective
Produce a defensible, multi-seed consistency matrix of weak-form collocation results across the (N, ω) grid, with honest characterization of achievable accuracy per regime.

## Objective
Apply the tournament-winning DiagFisher+REINFORCE recipe to N=12 (BF+Jastrow, 49K params) and N=20 (Jastrow-only, 25K params), establishing whether the N=6 ω=0.1 improvement (+0.033% best, 2.3× over Adam) transfers to higher particle counts.

Success condition: For N=12, beat the current best at ω=0.1 (+0.122%) and at ω=1.0 (+0.018%). For N=20, establish any result below +1% at ω=1.0 (current: +32.889%, all pre-bugfix).

## Context

### What triggered this plan

The Optimizer Tournament (Phase 2, 2026-04-05) is complete. Key findings:

| Run | Optimizer | Loss | ω | Best VMC err (%) | vs Adam |
|---|---|---|---:|---:|---|
| t2_df_re_w01 | DiagFisher | REINFORCE | 0.1 | **+0.033** | 2.3× better |
| t2_cg_fd_w01 | CG-SR | FD-colloc | 0.1 | +0.044 | oscillated/diverged |
| t2_adam_fd_w01 | Adam | FD-colloc | 0.1 | +0.052 | control (+0.148% final) |
| t2_df_re_w001 | DiagFisher | REINFORCE | 0.001 | +0.095 | architecture/sampling limited |

DiagFisher+REINFORCE is the clear winner: cheapest (4.4s/ep vs 9.4s/ep Adam, 15.6s/ep CG-SR) AND best accuracy. The recipe uses diagonal natural gradient with minimal overhead.

### Current state of N=12 and N=20

**N=12 (BF+Jastrow, 49,373 params — same architecture as N=6):**

| ω | Best err (%) | Checkpoint | Recipe |
|---:|---:|---|---|
| 1.0 | +0.018 | v12b_n12w1.pt | Adam, long campaign |
| 0.5 | +0.028 | v12b_n12w05.pt | Adam, long campaign |
| 0.1 | +0.122 | v12b_n12w01.pt | Adam with transfer |

All N=12 results used Adam only. DiagFisher was **never tested** at N=12. Memory profiling (2026-04-05) confirmed BF+DiagFisher fits easily: ~4GB at n-coll=2048, 7.6s/epoch.

**N=20 (Jastrow-only, 25,562 params):**

| ω | Best err (%) | Checkpoint | Recipe |
|---:|---:|---|---|
| 1.0 | +32.889 | smoke_n20_o1p0.pt | Adam, pre-bugfix |
| 0.1 | +5.533 | smoke_n20_o0p1.pt | Adam, pre-bugfix |

**CRITICAL:** ALL N=20 results predate the importance-sampling bugfix (2026-03-24). Post-fix N=20 was NEVER retested. The bugfix fixed exponential corruption of importance weights that is worst at high dimensionality — exactly N=20's problem (40D config space, ESS≈1 was observed).

Decision [2026-03-19]: N=20 uses Jastrow-only (not BF) due to BF's O(N²×hidden) memory. Memory profiling confirmed Jastrow+DiagFisher fits: ~4GB at n-coll=1024.

### The transfer hypothesis

At N=6 ω=0.1:
1. DiagFisher+REINFORCE achieved +0.033%, beating Adam's +0.076% by 2.3×
2. The advantage came from better per-parameter scaling (diagonal Fisher separates parameter curvature from gradient variance), low overhead (4.4s/ep), and stable convergence (no oscillation unlike CG-SR)
3. The recipe requires only diagonal Fisher estimation — no O(P²) matrix operations

At N=12, the same architecture is used (BF+Jastrow, 49K params). The Fisher diagonal should scale identically since it's per-parameter. This is Layer 5 (hyperparameters/optimizer) — the architecture (Layer 3) and data (Layer 1) are unchanged.

At N=20, the situation is different: different architecture (Jastrow-only), higher dimensionality (40D), and all prior results predate the bugfix. Layer 1 (data/sampling) must be verified first: does the post-bugfix importance sampling produce reasonable ESS at N=20?

## Approach

**N=12:** Direct transfer of the winning recipe to warm-start from existing v12b checkpoints. Test across ω ∈ {0.1, 0.5, 1.0} to see if DiagFisher improves over Adam uniformly. Include an Adam+REINFORCE control for fair comparison (tournament showed REINFORCE itself may contribute, not just DiagFisher).

**N=20:** Diagnostic-first approach. Before optimization experiments, verify that post-bugfix sampling produces reasonable ESS at N=20. If ESS is reasonable, apply DiagFisher+REINFORCE. If ESS is still catastrophic, the problem is Layer 1 (sampling) not Layer 5 (optimizer), and different interventions are needed.

**Hardware:** 8× RTX 2080 Ti, all available. N=12 fits ~4GB/GPU; N=20 fits ~4GB/GPU. Can run up to 8 parallel experiments in tmux.

## Foundation checks (must pass before new work)

- [x] Data pipeline known-input check: importance-sampling bugfix verified at N=6 (tournament Phase 2 all completed successfully)
- [x] Existing N=12 checkpoints load and run: v12b_n12w1.pt confirmed working in memory profiling (2026-04-05)
- [x] Existing N=20 checkpoints load and run: smoke_n20_o1p0.pt confirmed working in memory profiling (2026-04-05)
- [ ] N=20 post-bugfix ESS check: **NOT YET DONE** — must verify before optimization experiments
- [ ] N=12 DiagFisher smoke test (100ep): verify recipe runs without error at N=12

## Scope

**In scope:**
- DiagFisher+REINFORCE at N=12 across ω ∈ {0.1, 0.5, 1.0}
- Adam+REINFORCE controls at N=12 for fair comparison
- N=20 post-bugfix diagnostic (ESS, loss, basic convergence)
- DiagFisher+REINFORCE at N=20 if post-bugfix ESS is reasonable
- 15k VMC evaluations during training; deferred 100k heavy-VMC for winners only

**Out of scope:**
- N=20 backflow (decided against, DECISIONS.md 2026-03-19; revisit only if Jastrow-only reaches <+1%)
- CG-SR at N=12/N=20 (too unstable at N=6 ω=0.1, no reason to expect better at higher N)
- N=2 (already excellent at all ω)
- New architecture changes
- Multi-seed runs (deferred to next plan after winners identified)
- ω=0.001 at N=12/N=20 (no DMC reference for N=12 ω=0.001; N=20 doesn't have it either)

---

## Phase 1 — N=12 Smoke Tests + N=20 Diagnostic (one session)

**Goal:** Verify DiagFisher recipe runs correctly at N=12 (100ep smoke), and check N=20 post-bugfix ESS/loss behavior. After this phase, we know whether to proceed with full N=12 runs and whether N=20 is tractable.

**Estimated scope:** 1 launch script, 5-6 parallel GPU jobs, ~1-2 hours wall time

### Step 1.1 — N=12 DiagFisher+REINFORCE smoke tests (3 ω values, 100ep each)

**What:** Launch 3 parallel 100-epoch smoke tests of DiagFisher+REINFORCE at N=12, warm-starting from v12b checkpoints, at ω ∈ {0.1, 0.5, 1.0}. Use the exact tournament-winning recipe adapted for N=12: `--mode bf --n-elec 12 --loss-type reinforce --direct-weight 0 --clip-el 5.0 --n-coll 2048 --oversample 8 --micro-batch 256 --natural-grad --sr-mode diagonal --fisher-damping 0.01 --fisher-ema 0.99 --fisher-probes 4 --fisher-subsample 2048 --nat-momentum 0.9`. LR: 2e-4/2e-5 with 10ep warmup.

**Files:**
- `scripts/launch_higher_n_phase1.sh` (create)
- Resumes from: `results/arch_colloc/v12b_n12w{01,05,1}.pt`

**Acceptance check:** `ls outputs/higher_n/phase1/smoke_n12_df_w*.jsonl | wc -l` → expected: `3` (one JSONL per ω value)

**Risk:** DiagFisher may behave differently with 12 electrons (more particles → different gradient structure). Smoke test length (100ep) may be too short to see improvement — but it's enough to verify the recipe runs without errors and loss decreases.

### Step 1.2 — N=12 Adam+REINFORCE control (1 ω value, 100ep)

**What:** Single Adam+REINFORCE smoke test at N=12 ω=0.1 (the hardest N=12 case) warm-starting from v12b_n12w01.pt. This separates the REINFORCE contribution from the DiagFisher contribution at N=12.

**Files:** Same launch script as 1.1

**Acceptance check:** `ls outputs/higher_n/phase1/smoke_n12_adam_w01.jsonl | wc -l` → expected: `1`

**Risk:** Low — Adam+REINFORCE is well-tested at N=6.

### Step 1.3 — N=20 post-bugfix diagnostic (2 ω values, 200ep each)

**What:** Launch 2 diagnostic runs at N=20 (ω=1.0 and ω=0.1) using Adam+REINFORCE from existing Jastrow checkpoints. The primary goal is NOT accuracy — it is to measure post-bugfix ESS. Key flags: `--mode jastrow --n-elec 20 --loss-type reinforce --direct-weight 0 --clip-el 5.0 --n-coll 1024 --oversample 8 --micro-batch 128`. Use moderate LR (1e-4/1e-5).

Monitor ESS in training logs. If ESS > 10 consistently, the bugfix materially helped. If ESS ≈ 1           still, sampling is the bottleneck and optimizer changes alone won't help.

**Files:** Same launch script as 1.1
**Resumes from:** `results/arch_colloc/smoke_n20_o{1p0,0p1}.pt`

**Acceptance check:** `grep "ESS" outputs/higher_n/phase1/smoke_n20_diag_w1_training.log | tail -5` → expected: ESS values visible; check if ESS > 5 on average

**Risk:** ESS may still be ≈1 at N=20 post-bugfix. If so, N=20 requires sampling improvements (not optimizer improvements), and the plan must pivot.

### Step 1.4 — Analyze Phase 1 results

**What:** For N=12 smoke tests: compare training loss trajectory of DiagFisher vs Adam controls. For N=20 diagnostics: extract ESS statistics and loss trajectory. Write summary.

**Acceptance check:** `cat outputs/higher_n/phase1/phase1_summary.txt` → expected: contains ESS stats for N=20 and loss comparison for N=12

**Risk:** None.

---

## Phase 2 — N=12 Full Campaign (one session, ~8h)

**Depends on:** Phase 1 smoke tests show DiagFisher loss decreasing at N=12

**Goal:** Run DiagFisher+REINFORCE to convergence at N=12 for ω ∈ {0.1, 0.5, 1.0} with Adam controls. Produce 15k-VMC numbers that beat the current bests.

**Estimated scope:** 1 launch script, 6 parallel GPU jobs, ~8-12 hours wall time

### Step 2.1 — Launch N=12 DiagFisher+REINFORCE full runs (3 ω values)

**What:** Resume from v12b checkpoints. Give 4000 epochs with cosine LR decay. VMC eval every 500ep at 15k samples. Key hyperparameters same as tournament winner, except:
- `--n-coll 2048` (N=12 needs more memory, profiled at 2048)
- `--micro-batch 256` (profiled for N=12)
- `--fisher-subsample 2048`

Training time estimate: 7.6s/ep × 4000ep = ~8.4 hours per run.

| GPU | Tag | ω | Resume from |
|---:|---|---:|---|
| 0 | n12_df_w01 | 0.1 | v12b_n12w01.pt |
| 1 | n12_df_w05 | 0.5 | v12b_n12w05.pt |
| 2 | n12_df_w1 | 1.0 | v12b_n12w1.pt |

**Files:** `scripts/launch_higher_n_phase2.sh` (create)

**Acceptance check:** `for w in w01 w05 w1; do cat outputs/higher_n/phase2/n12_df_${w}_summary.json | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'{d[\"tag\"]}: {d[\"epochs_logged\"]} epochs')"; done` → expected: 3 lines showing ~4000 epochs each

**Risk:** N=12 may not benefit from DiagFisher as much as N=6. The existing Adam results at ω=1.0 (+0.018%) and ω=0.5 (+0.028%) are already good — improvement margin is small.

### Step 2.2 — Launch N=12 Adam+REINFORCE controls (3 ω values)

**What:** Same setup but without natural gradient (pure Adam). This isolates the DiagFisher contribution from the REINFORCE-loss contribution. Same LR, epochs, and evaluation schedule.

| GPU | Tag | ω | Resume from |
|---:|---|---:|---|
| 3 | n12_adam_w01 | 0.1 | v12b_n12w01.pt |
| 4 | n12_adam_w05 | 0.5 | v12b_n12w05.pt |
| 5 | n12_adam_w1 | 1.0 | v12b_n12w1.pt |

**Acceptance check:** Same as 2.1 but for adam tags

**Risk:** Low.

### Step 2.3 — Analyze N=12 results and update COLLOCATION_BEST_RESULTS.md

**What:** Compare DiagFisher vs Adam across all 3 ω values. Record any new bests. Key question: does DiagFisher's advantage scale with ω (i.e., bigger advantage at lower ω where variance is higher)?

**Acceptance check:** `grep "N=12" COLLOCATION_BEST_RESULTS.md` → shows updated best values if any improved

**Risk:** None.

---

## Phase 3 — N=20 Campaign (one session, depends on Phase 1 ESS diagnostic)

**Depends on:** Phase 1 Step 1.3 shows ESS > 5 at N=20. If ESS ≈ 1, this phase is BLOCKED and a sampling-focused plan is needed instead.

**Goal:** Apply DiagFisher+REINFORCE to N=20 at ω ∈ {0.1, 1.0}, warm-starting from existing Jastrow checkpoints, and establish the first post-bugfix N=20 baselines.

**Estimated scope:** 1 launch script, 4 parallel GPU jobs, ~10-15 hours wall time

### Step 3.1 — N=20 Adam+REINFORCE baselines (post-bugfix)

**What:** First establish honest Adam baselines at N=20 post-bugfix. The current numbers (+32.889% at ω=1.0, +5.533% at ω=0.1) may be drastically different post-bugfix. Run Adam+REINFORCE for 4000 epochs at ω=1.0 and ω=0.1.

Key flags: `--mode jastrow --n-elec 20 --n-coll 1024 --oversample 8 --micro-batch 128 --loss-type reinforce --direct-weight 0 --clip-el 5.0 --epochs 4000 --lr 1e-4 --lr-jas 1e-5 --vmc-every 500 --vmc-n 15000`

| GPU | Tag | ω | Resume from |
|---:|---|---:|---|
| 0 | n20_adam_w1 | 1.0 | smoke_n20_o1p0.pt |
| 1 | n20_adam_w01 | 0.1 | smoke_n20_o0p1.pt |

**Files:** `scripts/launch_higher_n_phase3.sh` (create)

**Acceptance check:** `cat outputs/higher_n/phase3/n20_adam_w1_summary.json | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['epochs_logged'])"` → expected: ~4000

**Risk:** The existing N=20 checkpoints were trained pre-bugfix. They may be in a bad region of parameter space that the post-bugfix sampling exposes. If training diverges from these checkpoints, consider training from a fresh Jastrow initialization.

### Step 3.2 — N=20 DiagFisher+REINFORCE (post-bugfix)

**What:** Same runs as 3.1 but with DiagFisher. Key additional flags: `--natural-grad --sr-mode diagonal --fisher-damping 0.01 --fisher-ema 0.99 --fisher-probes 4 --fisher-subsample 512 --nat-momentum 0.9`

Note: `--fisher-subsample 512` (not 2048) because N=20 has larger per-sample cost. Profiled to fit in ~4GB.

| GPU | Tag | ω | Resume from |
|---:|---|---:|---|
| 2 | n20_df_w1 | 1.0 | smoke_n20_o1p0.pt |
| 3 | n20_df_w01 | 0.1 | smoke_n20_o0p1.pt |

**Acceptance check:** Same pattern as 3.1

**Risk:** DiagFisher at N=20 may not help if the fundamental problem is sampling quality (ESS). The Phase 1 diagnostic should have determined this.

### Step 3.3 — Analyze N=20 results

**What:** Compare post-bugfix Adam vs DiagFisher at N=20. Determine whether improvement came from (a) bugfix alone, (b) REINFORCE loss, or (c) DiagFisher. Update COLLOCATION_BEST_RESULTS.md.

Key question for thesis: if post-bugfix N=20 at ω=1.0 drops from +32% to, say, +5%, that's a sampling fix. If DiagFisher then drops it from +5% to <+1%, that's the optimizer contributing on top. Both are meaningful but for different reasons.

**Acceptance check:** `cat outputs/higher_n/phase3/phase3_analysis.md` → contains post-bugfix N=20 results with comparison to pre-bugfix baselines

**Risk:** None.

---

## Risks and mitigations

- **N=20 ESS still ≈1 post-bugfix:** If Phase 1 diagnostic shows no ESS improvement, N=20 requires sampling innovation (adaptive proposal, MCMC, normalizing flow) — not optimizer tuning. Document this honestly as a finding. Mitigation: Phase 1 diagnoses this before investing GPU-hours.

- **DiagFisher doesn't help at N=12:** The existing N=12 results at ω=1.0 (+0.018%) and ω=0.5 (+0.028%) are already competitive. DiagFisher may not improve them because Adam was already near-optimal at higher ω. Mitigation: the ω=0.1 case (+0.122%) has the most room for improvement — that's the primary comparison target.

- **N=20 checkpoints trained pre-bugfix are in wrong region:** Post-bugfix sampling may reveal the pre-bugfix checkpoint is at a false minimum. Mitigation: if warm-start diverges, include a from-scratch run (Jastrow initialized from HO orbitals).

- **Wall-time overrun:** N=12 at 7.6s/ep × 4000ep = ~8.4 hours per run. Acceptable for overnight tmux. N=20 will likely be slower. Mitigation: monitor after first 500 epochs, adjust epoch count if needed.

- **Interpreting negative results:** If DiagFisher doesn't help at N≥12 while it helped significantly at N=6, that itself is a finding — it means the N=6 improvement was specific to that regime's curvature landscape. Document honestly.

## Success criteria

- N=12 ω=0.1: DiagFisher best VMC < +0.08% (from current +0.122%)
- N=12 ω=1.0: DiagFisher best VMC < +0.015% (from current +0.018% — tight margin)
- N=20 ω=1.0: Any result < +5% (from current +32.889% — radical improvement expected from bugfix alone)
- N=20 ω=0.1: Any result < +3% (from current +5.533%)
- At least one verified finding about whether natural gradient benefits transfer across N

## Current State

**Active phase:** Phase 1 completed
**Active step:** Step 1.4 completed — phase summary generated
**Last evidence:** `cat outputs/higher_n/phase1/phase1_summary.txt` shows N=12 smoke outputs and N=20 ESS diagnostics (ESS mean: 6.75 at ω=0.1, 4.80 at ω=1.0)
**Current risk:** N=20 ESS remains weak (ω=1.0 mean ESS < 5), so Phase 3 may be blocked by sampling quality even post-bugfix
**Next action:** Confirm whether to proceed to Phase 2 (N=12 full campaign) or pivot to a sampling-focused N=20 plan
**Blockers:** None for Phase 2; conditional blocker for Phase 3 if ESS gate is enforced strictly
