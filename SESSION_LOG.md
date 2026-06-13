# Session Log

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
