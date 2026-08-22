# Collocation conditioning — diagnosis and plan (2026-07-06)

Written after reviewing the four question threads, the journal back to March, the newest
results (overnight campaign, N=12 scaling, the colloc-SR fix `cc57ceb`), the thesis chapter
`Thesis/results_kernel.tex`, and the collocation training code on both stacks
(`src/run_weak_form.py` = March "thesis" recipe; `src/analysis/train.py` = new minimal probe).

---

## 0. The four questions, current state

**Q1 — Architecture (CTNN vs FFNN/DeepSet).** Strongest thread, mostly closed and seeded.
- ω=1 compression gap CONFIRMED, independent basins: d_eff 1.40±0.17 (CTNN) vs 3.25±0.03
  (DeepSet), ~11σ. Gap CLOSES at Wigner (3.70 vs 3.84 — the "inversion" was retracted under
  seeding). Survivors everywhere: var(E_L) discriminator (1.8–7×), orthogonal subspaces at the
  crystal, mode naming (CTNN leading mode stays physical at Wigner, DeepSet goes non-physical).
- Init N-scaling: gap GROWS with N (1.7×→2.6× to N=20).
- Overnight Group A surprise: **no-backflow arms reach near-DMC energy at N=6** — backflow is
  variance/quality + Wigner lattice work, not energy.
- OPEN: from-scratch 2×2 Jastrow-MP × BF-MP confound-killer (T1.1); V-cycle **bottleneck
  ablation** (decode was an honest negative — ablation is the decisive test, still unrun);
  trained-state N=12 (VMC at N=12 is itself struggling: 2026-07-04 DeepSet +4.4% raw).

**Q2 — Optimizer (SR vs Adam), VMC side.** Defensible publishable-null: with a physics-informed
ansatz under |Ψ|² sampling the optimizer is largely immaterial. SR edge resolved only ~ω=0.1
(~1.5σ), consistent ~10–20% closer in distance-to-exact + more seed-reproducible. Mechanism
(SR = NTK whitening) exact-validated at N=2. OPEN: ω=0.01 warm-started; N=6 low-ω; the
optimizer×paradigm quadrant (SR predicted to matter MORE under collocation's bad measure —
untested at N=6, entangled with the recipe problem below).

**Q3 — Paradigm (collocation vs VMC).** The confused one. See §2.

**Q4 — What the networks learn.** Three DISTINCT low-dimensionalities: readout feature-rank ~1,
tangent d_eff 1.2–3.7 (tracks physical NO count), TwoNN intrinsic dim ~6–8. Mechanism: messages
≈10% energy 100% kinetic; backflow kinetic role vanishes at Wigner → rank-1 collective lattice
corrector (now firmly an ARCHITECTURE signature: message-passing vs conventional backflow, NOT a
paradigm one — that thread resolved). OPEN: same triple on DeepSet; edge-scalar decode beyond
distance.

**Writing:** `results_kernel.tex` (822 lines), retractions folded in honestly. Its Q3 section is
the one most at risk of wrong-by-framing given §2.

---

## 2. Collocation diagnosis — recipe regression, not a bug, not (mostly) instability

Line-by-line review of `src/analysis/train.py`: **no bug found.** Self-normalised IS gradient
algebra is correct (direct + score term with E baseline); weak-form residual keeps the parameter
graph (`create_graph=True`); the Woodbury natural-gradient solve is algebraically right. It works
where noise is benign: N=2 dual-track hit overlap² 0.9992.

The regression is visible by putting March next to the overnight:

| | March (`run_weak_form.py`) | Overnight (`train_collocation_weak`) |
|---|---|---|
| N=6 ω=1   | **+0.002%** | +1.0 / +1.2% (2 seeds) |
| N=6 ω=0.1 | +0.091%     | +1.5 / +1.7% |
| N=6 ω=0.01| +0.193%     | **diverged** (−303%, +1e6%) |
| N=12 ω=1  | +0.018%     | +242% |

March recipe had, and the minimal trainer dropped:
- **(a)** strong-form E_L reward with forward-only Laplacian → the **zero-variance property**
  (gradient noise anneals to 0 near the GS). The new trainer uses the weak Rayleigh form, whose
  var(e_w)≈3 EVEN AT THE EXACT GS (2026-06-22: 400–11000× the E_L variance) → noise never
  anneals → Adam equilibrates at a finite radius ≈ order 1%.
- **(b)** importance **resampling** (points redrawn ∝ w ⇒ batch ≈ |Ψ|²-distributed, weights≈1)
  from an **adaptive auto-widened mixture**, plus ESS-floor oversampling, min-ESS skip,
  tempering, rollback. New trainer: fixed σ=1.3ℓ single Gaussian, clamped weighted batch.
- **(c)** pretrained canonical checkpoints + continuation chains as the DEFAULT (the 20.161
  N=6 ω=1 result is on record as a 4-stage chain — DECISIONS 2026-03-14).
- **(d)** probe-based checkpoint selection + heavy final eval.

Two findings from the actual overnight logs:
- **Floor, not lottery, at ω≥0.1:** the two seeds AGREE (+1.18/+1.03 at ω=1; +1.66/+1.47 at
  ω=0.1). Broad seed scatter is the instability signature; tight agreement is a floor.
- **ω=0.01 divergence was NOT training-time ESS collapse:** logs show ESS≈0.31–0.47 THROUGHOUT,
  including the diverging runs. Training-time ESS is self-referential (a from-scratch net stays
  near the proposal so w stays flat while the state is garbage). ESS collapse (3%→0.11%) is what
  happens holding a GOOD Wigner state against a fixed Gaussian. **Amend the 2026-07-03 journal
  interpretation on this point.** Two distinct failure modes: from scratch → coverage-biased
  unstable fixed point at Wigner (proposal never covers 22ℓ tails; weight clamp biases);
  warm-started under a FIXED Gaussian → genuine instant ESS collapse. ⇒ **pre-training and
  proposal adaptation are a package**; either alone fails at low ω. March worked because it had
  both.

---

## 3. The conditioning fix — brainstorm, ordered by leverage

**P0 — Split the Q3 axis first (cheap, clarifies the science).** "Collocation" conflates the
MEASURE (fixed q vs |Ψ|²-MCMC) and the ESTIMATOR (weak Rayleigh vs strong E_L). Quadrants:
VMC+E_L (great), q+E_L (March — great), q+weak (current — mediocre). Run the missing
**|Ψ|²-MCMC + weak form** (~1h): if also ~1% at N=6 ω=1 → weak form is the culprit; if fine →
the measure is. Likely estimator-dominated at ω=1, measure-dominated at low ω. Also fixes the
thesis framing: the March recipe IS the collocation paradigm (never samples |Ψ|², forward-only
Laplacian, REINFORCE) — "collocation" ≠ "weak form".

**P1 — The pre-training package (historically validated).** Three nested layers:
1. *Supervised pre-fit*: MSE-fit logΨ_θ to a cheap reference (Slater × short-range cusp Jastrow,
   or non-interacting/HF) on proposal samples. No Laplacian, seconds–minutes; right basin +
   sane w from step 0. (FermiNet-style HF pretrain; repo equivalent = the canonical-checkpoint
   init that `run_weak_form.py` loads BY DEFAULT.)
2. *ω-continuation* (chain checkpoints — but a +1.5% parent into an uncovering proposal can't
   work; needs a good parent AND layer 3).
3. *Proposal adaptation as prerequisite*: refit q to the current state every k steps —
   moment-matched from WEIGHTED RAW candidates (DECISIONS 2026-04-15: never refit from the
   resampled batch; skip on collapse), keep a defensive broad component q = α·broad +
   (1−α)·fit so weights stay bounded and tails covered. Gate steps on ESS.

**P2 — Restore the zero-variance reward.** Strong-form E_L (forward-only, chunked Laplacian) as
the TRAINING signal under q; keep the weak form as the DIAGNOSTIC contrast it was designed to be.
N=12 hybrid: weak-form early, switch to / anneal in strong-form for the endgame
(`rayleigh_hybrid_loss` already expresses this mix).

**P3 — Optimizer, in its place.** colloc-SR converges (0.91→0.97 at N=2) but is slow + needs a
stable Fisher → needs ESS → depends on P1.3. DiagFisher (`NATURAL_GRADIENT_COLLOCATION.md`) is
the cheaper purpose-built option. March evidence: Adam+ESS beat natural gradient at low ω because
Fisher noise is brittle at low ESS ⇒ optimizer is the THIRD lever, after estimator and proposal.

**P4 — Discipline.** Pre-register: match the March table within ~2× at N=6, 3 seeds/rung before
promoting; judge by MCMC-evaluated E_L and (N=2) overlap — never E_weak (cc57ceb: misleading).

**Cheapest first experiment:** 3-seed fan of the current trainer at N=6 ω=1 + the P0 missing
quadrant + one arm with a 200-step supervised pre-fit. Separates floor-vs-instability,
estimator-vs-measure, and tests the pre-training hypothesis in isolation.

---

## 4. Threads to queue alongside

- **T1.1** honest CTNN-vs-FFNN 2×2 (from scratch, seeded, kinetic/potential decomposition) —
  kills the last "energies tie" confound; most thesis-load-bearing architecture item left.
- **V-cycle** bottleneck ablation on existing checkpoints (free) + one flat-CTNN matched-param
  control — settles T1.5 (the decode was the wrong test).
- **Intrinsic dimension** triple (feature-rank / ID(Z) / d_eff) on DeepSet — generic vs
  architecture claim.
- **SR vs Adam** ω=0.01 warm-started (N=2 → N=6); the collocation side of the 2×2 waits until
  P1/P2 give collocation a fair trainer.

**Through-line:** conditioning lives in the ansatz (cusp/backflow), the measure (|Ψ|² vs q), and
the estimator (zero-variance vs not); the optimizer is downstream of all three. The collocation
fix is the Q3 chapter writing itself, not a detour.

---

## 5. IMMEDIATE next step (user directive, 2026-07-06)

Before any of §3, **reproduce the proper thesis-style collocation training** (`run_weak_form.py`,
canonical recipe: pretrained init + continuation + ESS-adaptive resampling) and report the
energies. Check they land in the reported range (COLLOCATION_BEST_RESULTS.md: N=6 ω=1 ≈ +0.002%,
ω=0.1 ≈ +0.09%, ω=0.01 ≈ +0.19%). THEN decide what to build.

### 5.1 RESULT (2026-07-06) — recipe is sound; regression confirmed

Ran `run_weak_form.py` at N=6 ω=1, 800 ep, REINFORCE + ESS-adaptive, heavy 30k VMC eval:
- **from scratch: +0.391%** (E=20.2381); **BF-warm: +0.436%** (E=20.2472). var(E_L)≈0.3, ESS≈62%.
- Correct energy scale, sub-1%, no divergence. ~10–15× worse than the +0.002% headline — that gap
  is the MISSING continuation chain + canonical init (`bf_ctnn_vcycle.pt`, wiped Jun-13) + best-probe
  + natgrad polish, NOT anything broken.
- Confirms the diagnosis empirically: the overnight "awfully far" is the NEW minimal
  `train_collocation_weak`, not the paradigm. See JOURNAL 2026-07-06.
- Blocker for bit-exact repro: Jun-13 re-sync drifted `build_jastrow_model()` (25,562 params) away
  from the thesis `f_netCTNN.pt` PINN (53,933) — only the 182,403 backflow still matches.

**Decided next:** (1) push one N=6 ω=1 run to the headline (natgrad CG-SR + longer + best-probe);
(2) cascade ω=1→0.1→0.01 to confirm low-ω range + stability under the real recipe; (3) port
resampling + ESS-adaptation + strong-form reward into `src/analysis/train.py` for a fair Q3 dual-track.
