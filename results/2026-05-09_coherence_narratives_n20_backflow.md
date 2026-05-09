# Research coherence, narratives, and why N=20 stayed Jastrow-first

**Date:** 2026-05-09  
**Scope:** Synthesis of repo state (plans, journal, best-final tables, decisions) — not a new experiment.

---

## 1. What “best” means here

Three energy numbers coexist in training logs:

1. **Collocation batch energy** — cheap, local to the current point set; good for optimization direction, dangerous as a scientific headline alone.
2. **Periodic VMC probes** — more honest than collocation energy, but can be optimistic if you cherry-pick minima across probes.
3. **Final heavy VMC** — the metric that matters for claims comparable to DMC.

The flagship **N=6, ω=1** weak-form story (`EXPERIMENT_REPORT.md`, `bf_hardfocus_v1b`) is explicitly a **continuation chain** judged by **heavy evaluation**, not by a single collocation print.

---

## 2. Coherent training style across winners

**Collocation + weak form** is the backbone: directed signal without full VMC inner loops at every step.

**REINFORCE-first** (`direct-weight` toward zero) appears repeatedly in the runs that survived scrutiny for BF+Jastrow at moderate N: mixed direct-gradient paths were easier to destabilize in the documented BF search.

**Continuation in ω** (and long low-LR polish) is the default success pattern: resume from a checkpoint that already has the right qualitative physics, then refine. One-shot from random init is repeatedly a bad bet at harder (N, ω).

**Optimizer and sampling are paired:**

- When Fisher estimates are stable and proposals overlap well, **SR / diagonal Fisher / CG-SR** can extract steep improvements (N=6 cascade and tournament literature in-repo).
- When importance weights are heavy-tailed (low ω, large N×d), **Adam + adaptive oversampling + ESS floors + tempered/clipped resampling + rollback** is the recurring survival kit. Natural-gradient noise in bad-ESS regimes is a recurring failure mode in the journal, not an accident.

**Ansatz tracks difficulty:** BF+Jastrow owns the **N=6** (and much **N=12** at ω not too small) “excellent / strong” cells. **Jastrow-only** is what the grid uses for **N=20** at accessible ω because that is where the project deliberately put compute after empirical comparison (see §4).

---

## 3. Narratives worth telling in a thesis

### A. Optimization and measurement beat architecture roulette

The best N=6 ω=1 result is framed as **process** (objective, batch design, continuation, evaluation discipline), not as “we invented a new antisymmetric architecture.” Exotic lines (e.g. neural Pfaffian) remain worse under comparable evaluation pressure in the selective history summarized in `EXPERIMENT_REPORT.md`.

### B. A real bug reframed a year of low-ω / high-N pain

The **mixture log_q** mistake (component density vs mixture density) biased importance weights by huge factors as N×d grew or ω shrank (`JOURNAL.md`, 2026-03-24). That single finding **reorders** which historical conclusions are trustworthy and motivates **post-fix** re-baselines.

### C. Scaling: solved → excellent → strong → wall

- **N=2:** essentially solved on DMC-backed ω; surrogate-backed ultra-low ω aligned after reference cleanup (`results/2026-04-19_collocation_status_update.md`).
- **N=6:** the cleanest non-MCMC showcase; BF+Jastrow + continuation.
- **N=12:** strong for moderate ω; low ω needs more care (sampling stress even when energy moves).
- **N=20:** high ω partially workable with **patient Jastrow polish**; low ω remains the frontier. After ShellFlow stabilization, **proposal variants cluster** in energy — evidence that **sampling-only** tuning stopped being the sole bottleneck (`plans/2026-04-26_grand_plan_lowomega_largeN.md`, Part 2.1).

### D. Negative results are load-bearing

- Short-chain **Langevin** proposal refinement: harmful (`DECISIONS.md`, `JOURNAL.md`).
- **ShellFlow** ablations at N=20 ω=0.01: ESS recoverable, energies still ~2× surrogate band — shifts attention to **ansatz / geometry / nodes** and training budget.

### E. “Catch-22” for backflow under collocation (conceptual)

The thesis appendix argues that **screened collocation** underweights near-node regions where backflow gradients would be largest, while **including** them risks pole structure in local-energy-based training — a structural tension between **stability** and **nodal learning** under this training distribution (`Thesis/appendix.tex`). This is not the only reason N=20 dropped BF (resource and empirical loss dominated the *operational* decision), but it explains why backflow is never “free” even when code exists.

---

## 4. Why N=20 has not been the *canonical* ansatz with backflow

**Clarification:** Backflow at N=20 was **attempted** in the early campaign; the current “best grid” is **Jastrow-first** by an explicit **engineering decision**, not because the codebase cannot run `--mode bf --n-elec 20`.

Recorded rationale (`DECISIONS.md`, 2026-03-19):

1. **Memory:** `bf-hidden=128` **OOM** on the target GPUs; `bf-hidden=64` **trained but plateaued around ~+18%** error at ω=1.0 while **Jastrow-only** reached **~+1.3%** with the same broad effort budget — a decisive empirical loss per unit wall time.
2. **Gradient budget tradeoff:** BF’s **O(N²)** message-passing structure at N=20 consumes memory that would otherwise fund **larger `n-coll` and `oversample`**, which the same decision notes as **first-order** drivers of gradient quality at that N.
3. **Noise:** Under the **then-current** (partly bug-affected) sampling, the extra BF parameters were judged **under-utilized** — capacity did not convert into lower energy.
4. **Policy:** The decision explicitly **constrained** N=20 production runs to **Jastrow-only**, with **medium-high** confidence and an explicit escape hatch: **revisit BF after a strong Jastrow warm-start**.

The **grand plan** (2026-04-26) adds that the **old BF failure is confounded**: importance sampling was wrong, BF was not warm-started from a good Jastrow, and **from-scratch** N=20 BF is intrinsically hard. **None of those confounds were removed in one controlled rerun** at the time of that document — so the grid stayed Jastrow-first while ShellFlow and reference hygiene advanced.

**Forward path (already written in-repo):** Experiment C in the grand plan proposes **tiny BF** on top of `n20x2_adam_w1_best.pt` (freeze-then-unfreeze), i.e. exactly the warm-started ablation the 2026-03-19 decision said would be the condition for a serious BF return.

---

## 5. Caveats for this note

- This file is **interpretive**; numbers for claims should still be taken from the cited logs and `src/config.py` references.
- “Best” tables mix **DMC** and **surrogate** references for some low-ω cells; percentage errors there are **relative to the chosen reference**, not absolute truth against unknown DMC.
