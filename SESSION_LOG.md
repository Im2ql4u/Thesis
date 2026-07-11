# Session Log

Last session: [2026-07-02] — Results chapter + Phase A (Gate A): Q1 mechanism closed, manifold named

## What was done this session
1. **Consolidated + committed** a week of uncommitted in-flight work (three-questions rebalance,
   Phase 0/1/2, the unjournalled Jun-27 low-ω SR win). Two commits (a582d56 code+results+docs,
   c81e220 thesis chapter). Corrected the run_weak_form "eliminates the conditioning catastrophe"
   doc claim; added .ipynb_checkpoints to .gitignore.
2. **Wrote the thesis results chapter** `Thesis/results_kernel.tex` (Q1/Q2/Q3, tables+figures+
   discussion+conclusions; 8 figures in results/figures/results/kernel/). Self-contained; to be
   restructured/merged into results.tex.
3. **Phase A closers (no training):**
   - A2 (`run_phaseA_closers.py`): NO participation ratio for BOTH arches vs ω — architecture-
     independent physical mode count; CTNN tangent tracks it, DeepSet rigid and crosses it.
   - A3: cross-architecture tangent projection — shared leading mode at weak coupling, divergence at
     Wigner.
   - A4 (`run_response_projection.py`, extended `reference.py` with λ + excited states): NAMED the
     N=2 manifold — leading mode = breathing, effective space = {breathing, correlation-hole,
     rel. excitation}, R²=1.00.
   - A5: crossover sample-converged (DeepSet ω=0.01 flat at 3.24).
   - Journal 2026-07-02 (Gate A); tex updated with the both-arch mode-count table + naming subsection.
4. **Phase B (Gate B) — seeded the crossover, and it OVERTURNED the Phase-A headline.** Trained 3
   seeds × {CTNN, DeepSet} at ω=0.01 to matched analysis grade (`launch_phaseB_crossover.sh`, 6 runs
   on GPUs 1–6) + `run_phaseB_crossover_analysis.py`. The single-seed d_eff "inversion" (CTNN 5.21 vs
   DeepSet 3.24) does NOT survive: seeded, **CTNN 3.70±0.04 vs DeepSet 3.84±0.08 — they converge**.
   What survives: var(E_L) discriminator (CTNN 1.8× lower, seeded), cross-projection (leading modes
   orthogonal at the crystal), N=2 naming. Corrected results_kernel.tex throughout (dagger'd the
   retracted row, added seeded-correction table+figure, fixed interpretation/synthesis/conclusions/
   caveats); JOURNAL 2026-07-02b; DECISIONS 2026-07-02b (retraction). The seeding discipline working.
5. **Q1 anchor seeded (independent basins) — the last open Q1 claim, LOCKED.** No new training: reused
   the existing independent ω=1 runs (5 CTNN-big + 3 DeepSet-big) via `run_phaseB_seeded_analysis.py
   anchor`. Compression gap CONFIRMED: **CTNN 1.40±0.17 vs DeepSet 3.25±0.03 (~11σ)**. Full seeded Q1:
   strong compression at ω=1 → converges at Wigner. Expanded the seeded table to both regimes; JOURNAL
   2026-07-02c.

## Debugging notes (for future runs)
- Env: `source /etc/profile.d/lmod.sh; module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1`;
  pin `CUDA_VISIBLE_DEVICES=0`. Checkpoints are gitignored (*.pt) but present on disk.
- The natural-orbital 1-RDM grid truncates at low ω (trace<N_up even at grid_half~50, n_grid=26):
  the Wigner density is too spread for a coarse grid. Ratios across arches are fair (shared grid);
  absolute low-ω NO counts need a finer/adaptive grid.

## Continued this session — the four follow-up steps (all four addressed)
6. **Q1 N-scaling (init):** `run_dinit_scaling.py` — CTNN-vs-DeepSet d_eff gap at initialisation GROWS
   with N (1.7×→2.6× to N=20); trained-state N≥12 deferred (from-scratch diverges + Laplacian OOM).
7. **N=6 mode-naming:** `run_mode_naming_N6.py` (operator decomposition) — leading modes are physical
   collective modes; at Wigner DeepSet's leading mode goes non-physical (R²=0.01±0.01 vs CTNN 0.93,
   seed-robust). JOURNAL 2026-07-02d.
8. **Q2 low-ω SR trend:** ω=0.05/0.03 — SR's edge is MODEST (energy resolved only at ω=0.1; consistent
   ~10–20% lower dist-to-exact). Honest publishable-null verdict. ω=0.01 diverges (light recipe).
   JOURNAL 2026-07-02f.
9. **Q3 dual-track:** added `train_collocation_weak`; `run_dual_track_N2.py` — VMC and collocation reach
   the SAME state (overlap² 0.9992), collocation ~5× higher var(E_L). JOURNAL 2026-07-02g.
   All folded into results_kernel.tex (subsections, tables, figures) with honest caveats.

## Next session
**Recommended starting point:** the ω=0.01 crossover is now seeded (and the inversion retracted). Next:
(1) seed the ω=1 anchor to confirm the weak-coupling compression gap (1.2 vs 3.4 — the remaining
single-seed Q1 claim, though large); optionally the full independent-basin cascade ×3 from ω=1.
(2) N=6 response-projection (name the modes at N=6 — needs the finite-difference reference extended to
N≥6). (3) Q2 low-ω SR sweep (does the ω=0.1 SR>Adam edge become decisive at ω=0.01 / N=6). (4) Q3
dual-track (VMC+SR vs collocation, same ansatz).
**Context freshness:** current

---
Prior session below.

Last session: [2026-06-22/23] — Three-questions roadmap + Phase 0/1 execution through Gate 1

## What was done this session
1. **Rebalanced the program** to three co-equal questions (Q1 CTNN-vs-FFNN, Q2 SR-vs-Adam, Q3
   collocation-vs-VMC) with the kernel/dimension picture as shared lens (DECISIONS 2026-06-22b);
   roadmap `plans/2026-06-22_dimension-program-and-roadmap.md`, status report.
2. **Phase 0 (Gate 0):** consolidated the un-journalled 2026-06-15 runs — CTNN eff-rank ~1.5 vs
   DeepSet ~3.8 (architectural, present at init); SR≈Adam null extends to N=6. (no training)
3. **Phase 1 (Gate 1), all three questions at the anchor:**
   - Q1: built the fair common-probe wrapper (`src/analysis/fair_dimension.py` +
     `scripts/run_fair_dimension.py`); gap SURVIVES — CTNN 1.2–1.7 vs DeepSet 2.9–3.7, measure-robust.
   - Q2: built `--no-cusp` into `exp_sr_mechanism.py`; ran the cusp-OFF 2×2 (4 seeds). Cusp prior is a
     stiff-mode preconditioner (removing it ~2–3×'s the stiff error, flips ranking toward SR), but
     SR≥Adam is within seed noise at ω=1 → decisive SR test moves to low ω.
   - Q3: `scripts/run_estimator_variance.py` — weak form loses zero-variance 400–11000×; means agree.
   - Journals 2026-06-22c, 2026-06-23; GATE1.md in 2026-06-23_SRmech_N2_nocusp/.

## Debugging notes (for future runs)
- Detached long jobs with `setsid bash -c '... > /tmp/log 2>&1' &` to survive session teardown (a
  harness-tracked bg job died overnight). Pin GPU with `CUDA_VISIBLE_DEVICES=0` to avoid the
  cuda:0/cuda:1 device-split crash when ref and per-seed Systems pick different "best" GPUs.
- Output dirs are named by `date.today()` — a run relaunched on a later date lands in a different
  dated folder (don't grep the old date).

## Next session
**Recommended starting point:** Phase 2 — ω-sweep {1, 0.5, 0.28, 0.1, 0.01} for all three questions
(needs re-training analysis-grade low-ω GS for CTNN + DeepSet; cascade allowed). Predictions: d_eff
gap grows toward Wigner (Q1); SR finally beats Adam where var(E_L) explodes (Q2); var(e_w) and the
collocation-vs-VMC gap track ω (Q3).
**Context freshness:** current

---
Prior session below.

Last session: [2026-06-22] — Strategic review, consolidation + audit, effective-dimension spine

## What was done this session
1. **Strategic review.** Diagnosed the "standing still" pattern: single-seed promotion → retraction
   → re-planning (e.g. cusp×SR wall-break → Jun-20 null; kernel plan → collocation plan after one
   null). Saved as a memory (feedback-consolidate-dont-thrash). Reframed the "nihilistic" findings:
   SR=NTK-whitening is exact-validated and the Jun-20 null only tested the cusp-ON quadrant where SR
   is *predicted* not to help; CTNN var(E_L) advantage is real; ESS-collapse (3.0%→0.11%) is a clean
   positive.
2. **Audit.** Graded every kernel-plan question by status. Found untracked `2026-06-15_{2x2,eq}_*`
   dirs holding a never-journalled, ≥2-seed CTNN-vs-DeepSet kernel comparison at N=6 ω=1:
   **eff-rank(S) ≈ 1.5 (CTNN) vs 3.4–4.8 (DeepSet)**, var(E_L) 0.026 vs 0.033–0.097 (DeepSet short
   even at 2× params). This answers X2/D6/B3 — unwritten until now.
3. **Fairness check** of the dimension measurement (`build_O`/`kernel_spectrum`): formula/tol/
   centering/sample-count identical (fair); eff-rank resolved (≪768 samples); confound = each net
   sampled from its own |Ψ|² (needs a common probe set); rank can become sample-limited at higher N.
4. **Wrote:** [STATUS_REPORT_2026-06-22.md](STATUS_REPORT_2026-06-22.md) (state of work, the surfaced
   result, what the dimensions are + how to map to H), the long-term roadmap
   [plans/2026-06-22_dimension-program-and-roadmap.md](plans/2026-06-22_dimension-program-and-roadmap.md),
   plus JOURNAL + DECISIONS entries.

## Next session
**Recommended starting point:** Phase 0/1 of the new roadmap — finish collating the 2026-06-15 dirs
(incl. trajectory CSVs for lazy-vs-rich), then re-measure d_eff at N=6 ω=1 under the fair protocol
(common probe set, sample-convergence, matched-params + matched-accuracy, ≥3 seeds). Gate 1: does the
~1.5-vs-3.5 gap survive?
**Open questions:** Does d_eff drop / the CTNN gap grow toward Wigner? Does it grow sub-linearly in N?
Can each dimension be named as a physical response/excitation mode of H?
**Context freshness:** current

---
Prior session below.

Last session: [2026-06-14] — Kernel-analysis program Phase A (N=2, omega=1.0)

## What was done this session
1. **Built the analysis package `src/analysis/`** (generalizable, no hardcoded N/omega):
   - `reference.py`: exact 2-electron solver for any omega (finite-volume radial diagonalisation);
     validated to E=3.000000 at omega=1 and <0.005% vs DMC for omega in {0.5,0.1,0.01}; cusp -> 1.
   - `system.py`: closed-shell Slater x Jastrow (x optional backflow) builder reusing repo
     Slater/psi_fn + MCMC sampler.
   - `diagnostics.py`: O/S/K kernel spectrum, SR-vs-plain alignment (NTK whitening), GS-quality,
     learned two-body correlation, exact overlap.
   - `train.py`: fast Adam-VMC trainer (REINFORCE gradient; one batched backward/step).
   - `scripts/run_phase_analysis.py`: driver (any N, omega) -> figures + REPORT.md + diagnostics.npz.
2. **Ran Phase A (N=2, omega=1.0).** E=2.99586 Ha, **overlap^2 with exact = 0.999984**, learned
   Jastrow matches exact, cusp recovered. Kernel: eff_rank(S) ~1.2/9842, kappa(S) ~9.5e11,
   cos(SR)=1.0 vs cos(plain) 0.26->0.05. See JOURNAL 2026-06-14.
3. **Phase A polish (additive, trainer untouched):** unclipped + zero-variance energy reporting,
   alignment on 2048 samples (> rank), full data dumps (plot_data.npz + 5 CSVs + summary.json; every
   plot input saved), and a driver-only final settle. Result: **E=3.000127 (+0.004%), overlap^2 =
   1.000000, cos(SR)=0.997 vs cos(plain)=0.041.** Output: ..._polished/.

## Debugging notes (for future runs)
- HPC env: `source /etc/profile.d/lmod.sh; module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1`
  gives python3.11 + torch2.1.2 + CUDA. The repo `.venv` lacks torch.
- Python stdout is block-buffered when not a tty -> use `python3 -u` / PYTHONUNBUFFERED=1, and do
  NOT pipe through `tail -f`-style (tail buffers to EOF).
- `pkill -f run_phase_analysis` SELF-KILLS the launching shell (its argv contains the pattern). Kill
  by explicit PID instead.
- CG-SR per-step cost is dominated by `_score_rows` (per-sample backprop loop over total_rows) ->
  too slow for many steps on over-parameterised nets. Train with Adam-VMC; use SR/O only for
  diagnostics. CG-SR stable window here was step_size~0.01/damping~1e-2 (0.03 diverged).

## Next session
**Recommended starting point:** Phase A polish (lr decay / SR polish to <0.05%), then Phase B:
sweep omega in {1e-3..1.0} at N=2, tracking kappa(S) and the cos(SR)-vs-cos(plain) gap through the
Wigner crossover. Then N=6.
**Open questions:** Is the -0.14% an estimator clip-bias only (yes per overlap)? Does eff_rank(S)
stay ~1-2 across omega? Does cos_plain collapse harder at low omega (higher kappa)?
**Context freshness:** current

---
Prior session below.

Last session: [2026-06-13] — Repo re-sync + kernel-picture analysis program

## What was done this session
1. **Git re-sync.** `git pull` revealed origin/main had been force-rewritten ~8 months ahead of the
   local line. Backed up stale local source edits to branch `backup/pre-sync-2026-06-13` (commit
   `63d00ce`), reset `main` to origin/main (`6d7ff8f`). Untracked result dirs left intact.

2. **Familiarisation.** Read METHODOLOGY.md, NATURAL_GRADIENT_COLLOCATION.md, the CTNN
   `architecture_diagnostics/DIAGNOSTIC_SUMMARY.md`, results.tex "What the networks learn", the SR
   code (`sr_preconditioner.py`: Woodbury/CG/MinSR; `Stochastic_Reconfiguration.py`), and the CTNN
   V-cycle architecture (`jastrow_architectures.py`).

3. **Brainstorm** for three new thesis angles (SR↔NTK↔NG; why CTNN>FFNN; what networks learn),
   unified under the tangent-kernel picture (`O`, `S=OᵀO`, `K=OOᵀ`). Catalogued experiments A1–A6,
   B1–B9, C1–C9, X1–X6.

4. **Wrote the vision/phased plan:** `plans/2026-06-13_kernel-analysis-program.md`. Focus on
   VMC/SR-trained (genuine GS) wavefunctions, with a GS-quality gate, dual-track VMC-vs-collocation
   comparison, and a kernel-picture lens. Phases: N=2 ω=1.0 → N=2 all ω → N=6 → N=12 → N=20, with a
   consolidation gate before each scale-up.

## Next session
**Recommended starting point:** Phase A — secure an analysis-grade `N=2, ω=1.0` GS (must hit E=3.0
exact), and stand up infrastructure I1–I3 (`O/S/K` builder, GS-quality evaluator, imaginary-time
direction probe). Then run A1/A2/A3 and check the kernel story against the exact N=2 solution.
**Open questions:** Do SR-trained and collocation-trained models reach the same wavefunction or just
the same energy? Does `cos(δθ_SR, imag-time)` ≈ 1 hold at N=2 as the theory (T1–T3) predicts?
**Caveats to control:** several published single-seed N=6 results (100% kinetic, random=zero messages)
need seed sweeps; N=20 reduced architecture confounds cross-N claims.
**Context freshness:** current

See ARCHIVE.md for full history.

---

## [2026-07-11] The CTNN backflow was dead — tanh saturation x COM projection; two findings retracted

**Done:** Diagnosed why `CTNNBackflowNet` produced |dx| = 0.00000 with fully-trained weights. Cause:
`dx = tanh(dx_head(h_v))` followed by a zero-mean (centre-of-mass) projection. My trainer's lr=3e-3
single-group Adam blew the node features up ~400x (|h| 0.047 at init -> std ~20 trained), driving the
dx_head pre-activation to ~148. tanh saturates there, every particle gets the identical +-1, and the
zero-mean projection cancels it to exactly zero -> tanh' = 0 -> gradient dies -> backflow is
unrecoverable. Reproduced live: it dies in under 100 steps. Fixed by adopting the thesis optimiser
(split LR groups 10:1, grad_clip=1.0, bf hidden 128); A/B under identical seeds: old |dx|=0.0000 /
sat=1.00 (dead), fixed |dx|=0.038 / sat=0.00 / com_kill=0.05 (alive, stable). Added `backflow_health()`
to every Adam and SR log line. Launched the v2 campaign (4 chains, GPUs 1-2).

**Retracted:** (1) "The rank-1 collapse at Wigner is a message-passing signature" — `backflow_arch`
defaulted to "conv" and the ctnn option did not exist before 2026-07-04, so EVERY earlier campaign used
the conventional per-particle `BackflowNet`. There was no message passing in the thing I measured.
(2) The v1 PINN campaign (`results/analysis/2026-07-04_pinn_ansatz`) is invalid — its CTNN arm had
Delta_x identically zero, so it compared a backflow-less ansatz against a real one (which is also why
"conv beat ctnn").

**Verified unaffected:** `Thesis/results_kernel.tex`. Its Q1 comparison holds the backflow identical and
fixed across the CTNN-Jastrow and DeepSet-Jastrow arms (conventional in both) and describes it as such —
a valid controlled experiment. The compression gap, mode-naming and message-ablation results are all
Jastrow-side and correctly labeled. Only backflow-side claims are retracted.

**Open / next:** v2 campaign running -> then the FIRST valid CTNN-vs-conventional *backflow* contrast.
Still open: whether v2 reaches thesis accuracy (0.01-0.08%); my VMC trainer has never been shown to
match the thesis's weak-form + residual-pretraining pipeline on energy, and if it cannot, the mechanism
analysis should move onto `run_weak_form.py`-trained states instead.

**Lesson (kept):** a network can be fully trained, have healthy weight norms, and still output
identically zero. Never infer liveness from weight norms — measure the OUTPUT.
