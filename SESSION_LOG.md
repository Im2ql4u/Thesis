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

---

## [2026-07-20] The Wigner collapse is the CONVENTIONAL backflow; multi-day N-scaling campaign launched

**Resolved (Aleksander's objection drove this).** He objected that "message passing buys kinetic
energy" cannot also explain "CTNN only helps at low omega, where there is basically no kinetic
energy". He was right, and the corrected measurement shows they are two different mechanisms:
  - dT_msg = 0.60 / 0.068 / 0.007 at omega = 1 / 0.1 / 0.01 — constant in units of omega (0.60, 0.68,
    0.70) but a SHRINKING share of total energy (3.0% -> 1.9% -> 1.0%). Kinetic is the HIGH-omega story.
  - At omega=0.01 the backflow buys dE=+0.0116, of which Coulomb +0.0167 and kinetic -0.0010. At
    Wigner the gain is entirely correlation/Coulomb; the backflow even COSTS a little kinetic energy.
  - Messages buy +0.0116 of the backflow's +0.0116: at Wigner, deleting the messages is as damaging as
    deleting the whole backflow. A backflow without inter-particle messages is worthless there.

**The inversion.** On correctly-labelled, thesis-quality states (N=6, thesis omega grid, 2 seeds,
identical training), it is the CONVENTIONAL per-particle backflow that collapses at Wigner —
BFrank 10.5 -> 1.0, d_eff 6.4 -> 1.1, error +0.58% — while the CTNN backflow holds rank ~10 and stays
at ~0.03%. The retracted 2026-07-02i claim said the opposite because backflow_arch silently defaulted
to "conv". The phenomenon was real; the attribution was inverted.

**Analysis bugs fixed before trusting any of it:**
  - ablations now RESAMPLE from their own |Psi|^2. Evaluating an ablated ansatz on un-ablated samples
    reported no-backflow energies BELOW the exact ground state (-1.62%); the corrected value is
    +1.627% — same magnitude, wrong sign. Every ablation energy before this fix was invalid.
  - ablations decomposed into kinetic and Coulomb, which is what separated the two mechanisms.
  - kernel_spectrum now uses the Gram trick (O O^T, B x B) instead of svdvals(O) (B x P workspace) —
    that SVD OOM'd the final diagnostics at N=6 and would have died at N=12/20.
  - reference energies looked up per-N from config.DMC_ENERGIES instead of hardcoded N=6 values.

**Launched (runs unattended for days):** scripts/orchestrate_scaling.py — N=6/12/20 x the THESIS omega
grid {1.0, 0.5, 0.28, 0.1, 0.01, 0.001} x {ctnn, conv} x 2 seeds = 12 chains / 60 stages, one chain per
GPU. Omegas restricted to the thesis's own grid because config._snap_omega RAISES beyond 50% from a
reference and silently SNAPS nearer values (0.3 -> 0.28), which would score a run against the wrong
reference. Resilience: OOM retry with halved sizes, 90-min stall detection on log mtime, per-chain
isolation, resume-on-restart, and an automatic analysis pass at the end.

**Open:** whether the collapse persists at N=12/20 and how the gap scales; results_kernel.tex still
describes the CTNN *Jastrow* (CTNNJastrowVCycle vs DeepSet), not the thesis ansatz — valid work, wrong
subject, NOT to be merged as-is (Aleksander asked me to hold off).

---

## 2026-08-22 — Q3 closed; thread 3 (cross-omega collapse) resolved; results banked

**Thread 3 — the deep-cascade wall, mechanism nailed.** Four transfer diagnostics on the last good
cascade rung (w0.0035.pt -> omega=0.0027, a 23% step). Scripts: warmstart_collapse_diag.py,
warmstart_shellinit_test.py, warmstart_dxscale_test.py, source_state_check.py.
  - Source rung is a GENUINE Wigner molecule at its own omega: 3.34 ell (classical 3.42), single basin
    (shell-init converges to the same 3.34 ell). Frontier is real; cascade integrity confirmed.
  - Collapse to ~1 ell' on transfer is in the ANSATZ, not the sampler: shell-initialised walkers still
    relax to 1.13 ell'. Two-basin/sampler-init hypothesis REFUTED.
  - Backflow is NOT the culprit: |dx| ~ 0.13 ell inward, preserved across transfer (0.15 -> 0.13 ell) —
    scales with ell but far too small to bridge the 2.4 ell core->shell gap.
  - Mechanism: the Wigner peak is a fine cancellation — Slater exp(-omega r^2/2) suppresses the shell
    ~1e-5 in density, the Jastrow amplifies it back ~1e5; a 23% omega change tips the balance and the
    density collapses to the non-interacting shell. From there the re-fit can't reliably climb back
    (the non-monotonic v1/v2/v3 behaviour). Coordinate-rescale warm-start is RULED OUT (ansatz relaxes
    inward from shell configs). Honest frontier: omega=0.0035, MCMC-free.

**Thesis:** folded the resolved mechanism into results_kernel.tex 4.x "The remaining wall" subsection
(replaced the earlier speculative Slater-rebuild/coordinate-rescale-as-future-work text).

**Repo hygiene:** .gitignore extended to exclude data binaries globally (*.npz/*.npy/*.zip — ~6 GB,
several npz >100MB would have broken the GitHub push) and exploratory src notebooks (per CLAUDE.md).
Figures (*.png) kept, per Aleksander's request. Everything else (tex, md/REPORTs, csv, log, json, py,
scripts, journals) committed and pushed. Q3 called DONE; Aleksander will do the prose writeup locally.

---

## 2026-08-22 (cont.) — Energy audit + editorial/coherence pass on the results chapters

**Energy audit (against real data on disk).** Traced every reported energy to source:
  - VMC+SR gold standard (old tab:energies) = config.DMC_ENERGIES; below w=0.1 (N>=6) these ARE the
    thesis's own PINN+CTNN values, not DMC (documented in config comments). Reference circularity confirmed.
  - Backflow scaling (kernel tab:bf_energies/tab:bf_rank) = 2026-07-16_scaling/master.csv + n20_wigner.log.
    Fixed rounding drifts to seed-means (N20 w1 CTNN +0.03->+0.06, rank 35.1->36.9; N12 w1; N6 w0.1 conv).
  - Cascade (tab:cascade) matches JOURNAL per-rung heavy-VMC eval exactly.
  - The "0.3%" the user distrusted = the STRONG-form catch-22 floor (N6 w0.5, Jastrow-only) and an
    arch_colloc Jastrow-only run -- NOT the weak-form REINFORCE collocation, whose genuine bests are
    +0.009% (w=1) down to +0.06-0.13% (N6 low w, per 2026-04-08 best-eval report).

**Editorial pass (3 commits).**
  - results_kernel.tex: table corrections, explicit reference hierarchy (DMC only w>=0.1), Q2 scope
    limitation (Adam~SR verified only N=2,6), [CITE NEEDED] on classical (1,5) config, warmer intro voice.
  - results.tex: intro repositioned as the physics half paired with the kernel 'why' chapter; reference
    honesty in tab:energies + tab:collocation (relabelled "Reference", noted conservative reliability
    bests); collocation strong/weak-form reconciliation fixing the 0.3% misattribution; [CITE NEEDED] on
    sample-efficiency comparison.
  - results.tex + appendix.tex: gauge caveat reconciliation (component metrics are path-dependent;
    invariants are energy/overlap/ablation); old full-rank CTNN connected to kernel's conv-collapse;
    catch-22 reframed as a strong-form negative baseline.

**Not yet done / open:** full sentence-level voice pass on section BODIES (only framing/intros/syntheses
done); a single explicit four-question spine statement; a complete [CITE NEEDED] sweep of physics claims.
Awaiting Aleksander's steer on how far to push the prose (he intends a local writeup pass).

---

## 2026-08-22 (cont. 2) — Whole-thesis coherence pass: intro/theory/method/discussion/conclusion + build fix

**Build fix (critical):** main.tex did not \input{results_kernel} -- the entire four-question kernel chapter
was absent from the thesis. Now included after results.tex (physics chapter, then the 'why' chapter).

**Introduction:** rewritten to the tangent-kernel spine. Scope now names separable-vs-message-passing and
the four questions read through the QGT/NTK; contributions replaced with message-passing-relational-channel,
optimiser+paradigm (SR conditioning, MCMC-free collocation to 0.0035), Wigner diagnostics, and a
gauge-honest representation item; 'optimizer switches backflow off' corrected to the architecture-specific
conv-collapse; N=20 added; diagnostics + thesis-structure updated (two results chapters).

**Conclusion:** added a 'method side' paragraph (relational channel, SR, MCMC-free collocation) and a
unified closing theme; corrected switch-off; gauge caveat.

**Discussion:** 'backflow switches off' retitled + qualified (net energy vs structure; conv collapses and
loses accuracy; ranks gauge-like); design bullet corrected (width vs kind); representation intro gauge
caveat; collocation bridged to the kernel Wigner-ring frontier.

**Theory:** new subsection sec:theory-tangent-kernel (score O, QGT S = Fisher metric of |Psi|^2 tied to SR,
NTK K Gram dual, d_eff participation ratio, kappa) grounding the kernel chapter. Cites BeccaSorella2017 +
StokesEtAl2020; NTK marked [CITE NEEDED] (no Jacot key in references.bib -- the ONLY remaining tag).

**Method:** (a) sec:method-collocation -- VMC vs importance-sampled collocation, weak-form REINFORCE
(Laplacian as reward, avoids the catch-22), ESS, Wigner-ring proposal; (b) sec:analysis-kernel -- tangent
kernel diagnostics (O, S/K, d_eff, kappa, common probe set, symmetric overlap^2) vs gauge-dependent ranks.

**N=20 collocation:** moved from tab:collocation to Appendix postcatch22:frontier, framed as an O(N^2 h_bf)
memory bottleneck (Jastrow-only +1.32/+2.74/+5.53%, BF +18% reversal), not a limit of the method.

**Remaining:** one [CITE NEEDED] (NTK/Jacot); two red [TODO] figure notes in results.tex (author's markers).
All energies verified against on-disk data. All commits pushed.

---

## 2026-08-22 (cont. 3) — Implemented the examiner review: submission-prep edits

**Scientific corrections (verified, not guessed):**
- Fixed a factor error I had introduced in theory tangent-kernel: S is NOT "exactly F"; since grad log|Psi|^2 = 2 grad log|Psi|, F = 4S. Stated explicitly.
- Cusp coefficients gamma_ud=1/(d-1), gamma_uu=1/(d+1) VERIFIED correct (match app:coulomb:cusp derivation: 2D -> 1 and 1/3; reduce to known 3D 1/2, 1/4). Added method->appendix cross-ref + 2D values. FLAGGED (not edited): the trap-unit slope of u carries a sqrt(omega) factor vs the physical-unit cusp slope=1; DMC-quality energies imply the computation is right, so this is a units-presentation gap for the author to reconcile against code.
- Fisher/score convention now stated once (theta = all params; F=4S).

**Consistency:**
- Correlator notation unified to W_theta (was f_net in results/discussion/conclusion, "f" in kernel). First-use identification added.
- CTNN defined at first use (results.tex arch paragraph): conventional per-particle BackflowNet vs message-passing continuous-time neural network (CTNN)/CTNNBackflowNet.
- Backflow Delta_beta identified with Delta_x (results) at method first use.
- Two \chapter{Quantum Dots} -> "Quantum dots: energies, representations, and Wigner molecules" (results) and "Discussion". Fixed Sec.->Chapter ref to ch:results.
- Duplicate LaTeX labels (6, structural-diagnostics section duplicated theory<->method) resolved by renaming the theory copies (no active refs pointed to them).

**Abstract:** rewritten to cover the actual thesis (N=20; message passing/relational channel; tangent kernel; SR/optimiser; MCMC-free collocation to 0.0035; Wigner physics; unifying insight). r_mode corrected 63.3 -> 64.3 a0 (matches tab:two_e_summary). "conditioning not depth" scoped to "at least as important as depth within the investigated regime".

**Introduction:** added a relational-architecture foreshadowing paragraph (message passing motivated by correlation being relational) without revealing conclusions.

**Citations:** NTK -> JacotEtAl2018-NTK (added to bib w/ arXiv), Deep Sets -> Zaheer2017-DeepSets (already in bib), classical (1,5) -> schweigert1994/Kong_2002. All TODO/CHECK/[kilde]/[Ref.]/CITE-NEEDED markers removed.

**FLAGGED for author (not auto-fixed):** (1) cusp trap-unit sqrt(omega) units presentation; (2) backflow naming clash method(BackflowNet=message-passing) vs results(BackflowNet=conventional, CTNNBackflowNet=message-passing) + conventional baseline not described in method; (3) content redundancy: structural-diagnostics described in both theory and method; (4) British/American spelling ~50/50 mixed, needs one careful global pass; (5) citation CONTENT verification (sources not inspected); (6) rendered figures not visually inspectable.

---

## 2026-08-22 (cont. 4) — Final targeted cleanup, verified against source code

**Cusp (Case B: implementation correct, equation wrong).** Read CuspMixin (src/jastrow_architectures.py):
u = sum gamma_ij * r_ij * exp(-r_ij/ell), ell=1/sqrt(omega), computed on RAW PHYSICAL x (forward calls
_compute_cusps(x), not x_sc). So slope du/dr|0 = gamma = the Kato condition (1 and 1/3 in 2D) -- CORRECT.
The method's eq:cusp wrongly used scaled r-tilde as the linear prefactor (physical slope gamma*sqrt(omega)).
Corrected eq:cusp to u(r)=gamma*r*exp(-r/a_ho) matching the code; trap-unit form given as (gamma/sqrt(omega))*r-tilde*exp(-r-tilde). Ansatz eqs now use u(r_ij).

**Architecture naming (code-verified).** BackflowNet AND CTNNBackflowNet are BOTH message-passing.
BackflowNet.forward: msg_in=cat[x_i,x_j,r_ij,...] over all pairs -> aggregate -> node update (single round,
pairwise). CTNNBackflowNet: "Copresheaf/graph-style", explicit node+edge features, bidirectional transport
rho_v_to_e/rho_e_to_v. DeepSetJastrow="pair-level encoder + multi-head pooling"; CTNNJastrowVCycle="Copresheaf
CTNN with V-cycle". => the baselines are NOT per-particle; all use pairwise info. Corrected the false
"per-particle / no relative channel / can only shift everyone together" claim across abstract, intro,
results, results_kernel, discussion, conclusion to the accurate "single-round pairwise messages vs iterated
copresheaf transport". Empirical results (rank collapse, d_eff, ablation) unchanged.

**CTNN expansion.** Code says "copresheaf" everywhere (CTNNJastrow/VCycle/BackflowNet docstrings), NOT
"continuous-time". Removed the "continuous-time" misnomer (was inferred from a stray results.tex phrase);
introduced CTNN as the copresheaf message-passing model WITHOUT asserting an acronym expansion (letters'
meaning not established in code/repo).

**Fisher/QGT.** Confirmed F=4S is stated and no F=S survives anywhere.

**British spelling.** Protected conversion (96 prose lines) skipping any \label/\ref/\cite/\url/\texttt line
and references.bib; verified all labels (sec:optimization etc.) intact. "centre" not converted (60 occ,
ambiguous) -- flagged.

**Structural-diagnostics redundancy.** Label collision already fixed; added a theory->Analysis cross-ref
noting the division (theory=concept, method=computation). Content NOT deleted -- left as author decision.

**Could NOT do:** no LaTeX toolchain installed -> no compile, no rendered-figure inspection, no overfull-box
check. Citation CONTENT not inspected (Pederiva2000/Mazars2008 support of specific values = source
verification required; Jacot2018/Zaheer2017 are standard landmark refs, metadata present).

---

## 2026-08-22 (cont. 5) — Built the thesis PDF (112 pp) and fixed the LaTeX errors that blocked it

No LaTeX on the box and the EasyBuild texlive was stripped (no perl infra, couldn't build a format); used
the tectonic single-binary engine (downloaded to scratch, bundle cached) instead.
Compile errors fixed:
  - graphicspath pointed at ../results/figures/results/ but the figures live in ../results/figures/;
    added that dir to all three graphicspaths. 9 figures recovered.
  - 4 figures were never committed (all_activations_shared_legend, N2/N6/N12_all_densities) -> created
    clearly-labelled PLACEHOLDER pdfs in results/figures/ so the doc compiles; MUST be replaced with the
    real figures. (These placeholders are *.pdf => gitignored, local only.)
  - results_kernel: '\dot\btheta' and unbraced subscript '\nabla_\btheta' broke under bm/physics
    (\btheta=\bm{\theta}); the unbraced subscript _\btheta made \bm's \begingroup scan fail. Fixed to
    _{\btheta} and rewrote the SR line as a discrete update \btheta <- \btheta - eta S^{-1} grad E.
  - references.bib: added the missing Vehtari2024 (PSIS) entry that was \nocite'd.
Result: exit 0, 112 pages, NO undefined refs/citations, NO errors, NO multiply-defined labels; 105 overfull
hboxes (cosmetic). NOTE: *.pdf is gitignored, so main.pdf and all figure pdfs are not normally tracked;
force-added main.pdf per request. The figure .pdf source assets remain gitignored (pre-existing), so a clean
clone will not rebuild without them.

---

## 2026-08-23 — Regenerated the 4 missing figures + fixed real errors from external review

**Figures (the "PLACEHOLDER" boxes):** the 4 were never committed. Data DID exist:
N{2,6,12}_all_densities from the saved |Psi|^2 sample bundles (results/tables/**/gr_N*.{npz,pt};
N=2 higher-omega were .pt, which my earlier .npz-only glob missed). Wrote scripts/make_missing_figures.py
to compute P(r) across omega from samples_X_bohr, and all_activations_shared_legend from the activation
functions directly. Verified N=6 renders correctly (peak marches 2->114 Bohr as omega 1->1e-3, matches
Table 6.4). Real figures now embedded (main.pdf 112 pp).

**Real errors fixed (from ChatGPT + Claude reviews, each verified against source/code):**
- CUSP: theory.tex gave 2D like-spin cusp = 1/2 (a 3D "halving" heuristic); appendix, method, and CODE
  (gamma_para=1/(d+1)) all use 1/3. Corrected theory to the p-wave value 1/(D+1)=1/3, made appendix canonical.
- "no Monte-Carlo sampling" -> "without Markov-chain Monte Carlo" (abstract/intro/conclusion): collocation
  IS Monte Carlo (importance sampling + resampling), just not a Markov chain.
- Added a precise 3-way distinction (MCMC-free / reference-use / reference-free) in the discussion: the
  higher-omega collocation uses E_DMC as a stage-I target + rollback signal, so it is MCMC-free but not
  reference-free; only the deep-Wigner cascade (no DMC below 0.1) is both.
- "Laplacian's zero variance" corrected: zero-variance is a property of the TOTAL local energy, not the
  Laplacian term alone (results_kernel Q3).
- "rigid common shift" -> "single collective displacement mode" (x3): COM projection removes literal
  translation, so rank-1 collapse is a collective mode, not a common shift.
- SR nuance added to Q2: SR helps when the metric is ill-conditioned AND reliably estimable; once sampling
  collapses the empirical S is noise/rank-deficient and inverting it harms -- reconciles Q2 with the CG-SR
  ultra-low-omega instability (Ch6).
- theory opening grammar ("preceding"->rest; unfinished "focus will then shift" sentence).

**Abstract:** rewritten tighter (Hawking-clear), MCMC-precise, with the collocation N<=12 vs VMC N=20 scope
caveat; fixed r_mode (N=6 ring approx 114 a0; the prior approx 64 was N=2's value mis-attached).

**NOT done (recommend, flagged to user):** bibliography is thin (34 refs) vs the novelty claims -- add
Carleo&Troyer 2017, Luo&Clark backflow-NQS, Hermann PauliNet, GNN-NQS, non-MCMC VMC; consider a more
specific title. Figure .pdf assets remain gitignored (*.pdf); force-added main.pdf + the 4 regenerated
figures only.
