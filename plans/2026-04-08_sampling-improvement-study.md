# Plan: Sampling Improvement Study — Learned Proposals, MinSR, and Diagnostic Infrastructure

Date: 2026-04-08
Status: in_progress

## Project objective
Produce defensible near-DMC weak-form collocation results across the (N, ω) grid, with honest understanding of what limits accuracy in each regime.

## Objective
Determine whether sampling quality — not optimizer choice — is the dominant bottleneck at low omega and high N, and implement concrete sampling improvements that unlock regimes currently blocked by ESS collapse.

Success condition: At the hardest currently-blocked regime (N=20, ω=0.1, best: +5.5%), demonstrate measurable improvement from sampling changes alone (target: <+3%), and produce diagnostic evidence sufficient to separate sampling contribution from optimizer contribution.

## Context

### Why this plan exists

The brainstorming session (2026-04-08) reached a key insight: **we cannot draw conclusions about whether SR/CG/natural-gradient are effective at low omega and high N because these optimizers have never been tested with adequate samples.** The current fixed Gaussian mixture proposal has catastrophic overlap with |ψ|² when N×d is large or ω is small:

- N=20, ω=0.1: ESS routinely 1–3 out of 8×n_coll candidates
- N=12, ω=0.01: ESS similarly collapsed, +25% error after long training
- N=20, ω=1.0: ESS 2–5, error still +1.3% — better but not converged

Meanwhile, at N=6, ω=0.1, where ESS is healthy (~100–900), both Adam and DiagFisher+REINFORCE reach <0.1% error easily. The gradient signal is there; the samples are not.

### What has already been tried and failed

From JOURNAL.md and DECISIONS.md:

1. **Langevin proposal refinement** (2026-03-19): K=10–20 Langevin steps from Gaussian proposal. Result: catastrophic (+152% at N=20, ω=0.1). Cause: too few steps for equilibrium in 40D; biased distribution creates feedback loop. **Decision: abandoned.**

2. **Adaptive sigma_fs** (2026-03-28): Auto-widen proposal widths at low ω. Result: helped initialization overlap but did not fix transfer ESS collapse at N=12, ω=0.001. Still useful but insufficient alone.

3. **Importance weight tempering** (`--resample-weight-temp`): Available in code but not systematically tested. Flattens weights, increases effective diversity but biases the distribution.

4. **Log-weight clipping** (`--resample-logw-clip-q`): Available in code but not systematically tested.

5. **ESS gating removal** (Consistency Campaign Phase 2): Removing the ESS gate allowed training to proceed, but the resulting quality was still far from target (+0.236% best at N=6, ω=0.1; +2.16% at ω=0.001).

6. **CG-SR with high CG iterations** (up to 50 in tournament): CG-SR at ω=0.1 oscillated and diverged despite tight trust regions. Root cause likely: Fisher matrix itself is noisy when ESS is poor, so a precise CG solve of a garbage system helps nothing.

### What has NOT been tried

1. **Learned/adaptive proposal distribution** — fitting a proposal to |ψ|² samples
2. **MinSR** (Chen & Heyl, Nature Physics 2024) — switching from $S^{-1}F$ to $O^+(Oe)$ when P >> N_samples
3. **PSIS-style tail diagnostics** — detecting when importance weighting is fundamentally unreliable
4. **Control variates** for energy estimator variance reduction
5. **Decoupled Fisher sampling** — using a larger or differently-drawn batch for Fisher estimation

### Literature grounding

- **MinSR** (arXiv:2302.01941, Nature Physics 2024): Instead of solving $(S + λI)^{-1}F$ in parameter space (P×P), MinSR solves in sample space (M×M where M << P). Key formula: $δθ = -O^+(Oe)$ where $O$ is the centered log-derivative matrix and $e$ is the centered local energy vector. This is numerically identical to SR when M > P but dramatically more stable and memory-efficient when P >> M (our regime: ~25K–49K params, ~500–2048 samples).

- **Neural Importance Resampling (NIR)** (arXiv:2507.20510, 2025): Trains a separate autoregressive network as proposal for NQS. Shows stable training where MCMC fails. The core idea (learned proposal → importance resample → train NQS) is directly applicable, though we can use simpler models than a full Transformer.

- **Pareto-Smoothed IS (PSIS)** (Vehtari et al.): Fits a generalized Pareto distribution to the tail of importance weights. The shape parameter $\hat{k}$ diagnoses weight reliability: $\hat{k} < 0.5$ is reliable, $0.5 < \hat{k} < 0.7$ is marginal, $\hat{k} > 0.7$ means the central limit theorem doesn't hold for the weighted estimator. This gives a principled answer to "is my ESS-based training trustworthy?"

## Approach

We attack the problem in layers, from lightweight diagnostics to structural sampling changes:

1. **Phase 1 (diagnostics):** Add PSIS tail index computation alongside ESS. This costs nothing computationally and immediately tells us *where* sampling is fundamentally broken vs merely inefficient. This is the floor.

2. **Phase 2 (adaptive proposal):** Implement a GMM-fitted proposal that periodically refits a Gaussian mixture to the best |ψ|²-weighted samples seen during training. This is the simplest possible "learned proposal" — no neural network, just sklearn-style GMM fitting every K epochs. If this works, we know the problem is proposal overlap, not something deeper.

3. **Phase 3 (MinSR):** Implement the MinSR formula as a new `sr_mode="minsr"`. This addresses the other half of the SR problem: even with perfect samples, standard SR/CG may be unstable when P >> M. MinSR is the principled fix.

4. **Phase 4 (controlled ablation):** Run the 2×2 experiment (old-proposal vs adaptive-proposal) × (Adam vs SR/MinSR) at the target hard regime (N=20, ω=0.1). This separates sampling contribution from optimizer contribution and is the scientific deliverable.

Why this order: Phase 1 is pure instrumentation (no risk). Phase 2 tests the highest-impact hypothesis (proposal overlap is the bottleneck). Phase 3 is independent of Phase 2 and can be developed in parallel. Phase 4 needs both.

## Foundation checks (must pass before new code)
- [x] Data pipeline known-input check — verified via DMC smoke matrix (2026-03-14)
- [x] Split/leakage validity check — N/A, no train/test split in VMC
- [x] Baseline existence — current best results documented in results/2026-04-08_best-final-eval-report.md
- [x] Relevant existing implementation read and understood — ran `find . -name '*.py' | grep -v __pycache__ | sort`; inspected sampling and training-loop code before edits

## Scope
**In scope:**
- PSIS diagnostic computation in `importance_resample`
- Adaptive GMM proposal as new sampling mode in `importance_resample`
- MinSR as new `sr_mode` in `src/sr_preconditioner.py`
- Controlled ablation script for 2×2 experiment
- Modifications to `src/functions/Neural_Networks.py`, `src/sr_preconditioner.py`, `src/run_weak_form.py`

**Out of scope:**
- Full normalizing flow or autoregressive proposal network (too complex for thesis timeline)
- Architecture changes (Jastrow vs BF is settled per regime)
- New loss functions
- Changes to evaluation protocol
- MCMC sampling (decided against 2026-03-19: would negate the collocation advantage)

---

## Phase 1 — PSIS Tail Diagnostics (1 session, ~2 files)

**Goal:** Every epoch reports a Pareto tail index $\hat{k}$ alongside ESS, so we can distinguish "inefficient but valid sampling" ($\hat{k} < 0.7$) from "fundamentally broken weighting" ($\hat{k} > 0.7$). This is pure instrumentation — no behavior changes.

**Why this matters:** ESS tells you how many effective samples you have but not whether the weighted estimator converges to the right answer. With heavy tails ($\hat{k} > 0.7$), the importance-weighted mean has infinite variance — more samples won't help. This diagnostic tells us exactly which (N, ω) regimes need a better proposal vs just more oversampling.

**Estimated scope:** ~60 lines of new code across 2 files, 0 behavior changes.

### Step 1.1 — Implement `psis_diagnostic` function
**What:** Add a standalone function that takes raw log-importance-weights and returns the Pareto $\hat{k}$ estimate. The algorithm: sort the M largest weights (M = min(ceil(0.2 × n), 3√n)), fit a generalized Pareto to their log-exceedances via the Zhang & Stephens (2009) estimator (a simple 1D optimization, no external library needed).
**Files:** `src/functions/Neural_Networks.py`, add after `importance_resample` (~line 248)
**Implementation detail:**
```python
def psis_diagnostic(log_w: torch.Tensor) -> float:
    """Pareto tail index k-hat for importance weight diagnostics.
    
    k < 0.5: reliable IS
    0.5 < k < 0.7: marginal, consider increasing samples
    k > 0.7: IS unreliable, need better proposal
    
    Uses the empirical Pareto fit on the largest 20% of weights.
    """
```
- Input: `log_w` tensor (B,) of raw log-importance-weights (before tempering/clipping)
- Output: float $\hat{k}$
- Method: fit GPD to upper tail via method-of-moments or maximum-spacing (simplest reliable approach)
**Acceptance check:** `python -c "import torch; from src.functions.Neural_Networks import psis_diagnostic; lw = torch.randn(1000); k = psis_diagnostic(lw); print(f'k_hat={k:.3f}'); assert -0.5 < k < 2.0"` → expected: prints k_hat value without error
**Risk:** Numerical edge cases with very small batches. Mitigation: return NaN when n < 20 with a warning.

### Step 1.2 — Integrate PSIS into `importance_resample` stats
**What:** Call `psis_diagnostic` on the raw log-weights inside `importance_resample` (before any tempering or clipping) and include `psis_khat` in the returned `stats_dict`. This is a 3-line addition.
**Files:** `src/functions/Neural_Networks.py`, inside `importance_resample` function, around line 232 (after log_w_raw is computed, before optional clipping)
**Acceptance check:** `python -c "import torch; from src.functions.Neural_Networks import importance_resample; x, ess, stats = importance_resample(lambda x: torch.zeros(x.shape[0]), 64, 2, 2, 1.0, device='cpu', dtype=torch.float32, return_stats=True); print(f'khat={stats[\"psis_khat\"]:.3f}')"` → expected: prints khat value
**Risk:** Slows down epoch slightly. Mitigation: psis_diagnostic is O(n log n) sort + O(M) fit where M ≈ 0.2n; negligible vs the Ψ evaluations.

### Step 1.3 — Log PSIS k-hat in training output
**What:** In the training loop, include `khat` in the per-epoch log line when `return_stats=True`. Format: `ESS=X khat=Y.YY` so it appears alongside existing diagnostics.
**Files:** `src/run_weak_form.py`, in the epoch logging block (around lines 920–940 where ESS is already printed)
**Acceptance check:** Run a 3-epoch smoke test at N=6, ω=1.0 and grep for `khat`:
```bash
python src/run_weak_form.py --n-particles 6 --omega 1.0 --n-coll 256 --epochs 3 --tag psis_smoke --no-save 2>&1 | grep -o 'khat=[0-9.]*'
```
→ expected: 3 lines with `khat=<value>`
**Risk:** The `--no-save` flag might not exist. Mitigation: use `--tag psis_smoke` and delete checkpoint after.

---

## Phase 2 — Adaptive GMM Proposal (1–2 sessions, ~3 files)

**Depends on:** Phase 1 complete (we need PSIS diagnostics to evaluate proposal quality)

**Goal:** Replace the fixed Gaussian mixture proposal with a periodically-refitted GMM that learns where |ψ|² has mass. The key insight: we already produce importance-resampled points every epoch that are approximately |ψ|²-distributed. We just need to fit a mixture to them.

**Why this is the highest-impact change:** The fixed proposal has widths tuned for ω ≈ 1, N ≤ 6. At N=20, the configuration space is 40-dimensional. A 3-component isotropic Gaussian centered at the origin has essentially zero overlap with a wavefunction that concentrates on a complex multi-particle configuration. An adaptive proposal that tracks where |ψ|² actually lives should dramatically improve ESS.

**Why GMM and not a neural network:** A GMM with K=8–16 components in Nd dimensions has O(K × Nd²) parameters ≈ O(25,000) for N=20, d=2, K=16. This is fast to fit (EM algorithm, ~10 iterations), fast to sample from (sample component, then sample Gaussian), and fast to evaluate density (logsumexp over K components). A neural network proposal would be more expressive but adds training instability, hyperparameters, and a second optimization loop. GMM is the minimal viable learned proposal.

**Important design decision for the implementer:** The GMM is NOT a replacement neural network. It is a simple statistical model fitted to observed samples. It lives alongside the wavefunction, not inside it. It has no gradients flowing through it. It is refitted every K epochs by calling `sklearn.mixture.GaussianMixture.fit()` or a pure-torch equivalent on the flattened resampled points.

**Estimated scope:** ~150 lines of new code, 1 new class, modifications to 2 existing functions.

### Step 2.1 — Implement `AdaptiveGMMProposal` class
**What:** A class that maintains a fitted GMM and provides `sample()` and `log_prob()` methods compatible with the existing `importance_resample` interface.

**Files:** `src/functions/Neural_Networks.py`, new class after the existing sampling functions (~line 160)

**Interface:**
```python
class AdaptiveGMMProposal:
    """Adaptive Gaussian Mixture Model proposal for importance resampling.
    
    Periodically refits a K-component full-covariance GMM to approximate |ψ|².
    Falls back to the fixed isotropic mixture when no fit is available.
    """
    def __init__(self, n_elec: int, dim: int, omega: float, 
                 n_components: int = 8, 
                 sigma_fs: tuple = (0.8, 1.3, 2.0),
                 refit_every: int = 50,
                 refit_min_samples: int = 500,
                 device='cpu', dtype=torch.float32):
        """
        Args:
            n_components: GMM components (8-16 recommended for N>=12)
            sigma_fs: fallback fixed mixture widths (used before first fit)
            refit_every: epochs between refits (50 = refit 40x in 2000 epochs)
            refit_min_samples: minimum accumulated samples before first fit
        """
    
    def accumulate(self, x_resampled: torch.Tensor):
        """Store importance-resampled points for next GMM fit.
        
        Called every epoch with the resampled collocation points.
        Maintains a rolling buffer of the last ~5000 points.
        """
    
    def maybe_refit(self, epoch: int) -> bool:
        """Refit GMM if epoch is a multiple of refit_every and enough samples exist.
        
        Returns True if a refit happened.
        Uses torch-based GMM EM to avoid CPU<->GPU transfers:
        1. Initialize means via k-means++ on buffer
        2. Run 20 EM iterations with full covariance
        3. Validate: check that no component has degenerate covariance
        """
    
    def sample(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample n points and return (x, log_q).
        
        Before first fit: delegates to sample_mixture (fixed proposal).
        After fit: samples from fitted GMM, evaluates mixture log-density.
        """
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate log q(x) at arbitrary points.
        
        Before first fit: delegates to eval_mixture_logq.
        After fit: evaluates fitted GMM log-density via logsumexp.
        """
```

**Key implementation details the implementer must follow:**
- Covariance regularization: add `1e-6 * I` to each component's covariance to prevent singular matrices
- The rolling buffer stores points as `(n_points, n_elec * dim)` flattened tensors on CPU to save GPU memory
- Buffer size: keep last `max(5000, 10 * n_components * n_elec * dim)` points — enough for stable EM
- After refit, log one line: `[GMM refit] epoch={e} components={K} min_weight={w:.3f} max_std={s:.3f}`
- If EM fails (singular covariance, NaN), keep the previous fit and log a warning. Never crash.

**Acceptance check:**
```bash
python -c "
import torch
from src.functions.Neural_Networks import AdaptiveGMMProposal

prop = AdaptiveGMMProposal(n_elec=6, dim=2, omega=1.0, n_components=4, refit_every=2, refit_min_samples=100, device='cpu', dtype=torch.float32)

# Before fit: should use fallback
x1, lq1 = prop.sample(200)
print(f'pre-fit: x={x1.shape}, lq={lq1.shape}')

# Accumulate fake points and refit
for i in range(5):
    prop.accumulate(torch.randn(100, 6, 2))
fitted = prop.maybe_refit(epoch=2)
print(f'fitted={fitted}')

# After fit: should use GMM
x2, lq2 = prop.sample(200)
lp = prop.log_prob(x2)
print(f'post-fit: x={x2.shape}, lq={lq2.shape}, lp_check={torch.allclose(lq2, lp)}')
print('OK')
"
```
→ expected: prints shapes, `fitted=True`, `lp_check=True`, `OK`

**Risk:** GMM fitting in pure PyTorch may be slow or numerically unstable at high dimensions (Nd=40 for N=20). Mitigation: Use diagonal or tied covariance as fallback if full covariance fails; log when this happens. Alternative: use `sklearn.mixture.GaussianMixture` on CPU numpy arrays (simpler, well-tested, acceptable since refit is infrequent).

### Step 2.2 — Wire `AdaptiveGMMProposal` into `importance_resample`
**What:** Add a `proposal` parameter to `importance_resample` that accepts an `AdaptiveGMMProposal` instance. When provided, use `proposal.sample()` instead of `sample_mixture()` for candidate generation, and use `proposal.log_prob()` for density evaluation. When `proposal=None` (default), behavior is unchanged.

**Files:** `src/functions/Neural_Networks.py`, modify `importance_resample` signature and the sampling block (~lines 200–215)

**Changes:**
1. Add parameter: `proposal: AdaptiveGMMProposal | None = None`
2. In the sampling block:
   ```python
   if proposal is not None and proposal.is_fitted:
       x_cand, lq_all = proposal.sample(n_cand)
       x_cand = x_cand.reshape(n_cand, n_elec, dim)
   else:
       x_cand, lq_all = sample_mixture(n_cand, n_elec, dim, omega, ...)
   ```
3. Everything downstream (log-weight computation, tempering, resampling, PSIS) stays exactly the same.

**Acceptance check:**
```bash
python -c "
import torch
from src.functions.Neural_Networks import importance_resample, AdaptiveGMMProposal

psi_fn = lambda x: -0.5 * (x**2).sum(dim=(1,2))  # Gaussian psi
prop = AdaptiveGMMProposal(2, 2, 1.0, n_components=2, refit_every=1, refit_min_samples=50)
for _ in range(3):
    prop.accumulate(torch.randn(100, 2, 2))
prop.maybe_refit(epoch=1)

x, ess, stats = importance_resample(psi_fn, 64, 2, 2, 1.0, device='cpu', dtype=torch.float32, proposal=prop, return_stats=True)
print(f'ess={ess:.1f} khat={stats[\"psis_khat\"]:.3f}')
print('OK')
"
```
→ expected: prints ess and khat values, `OK`

**Risk:** Shape mismatches between GMM output and expected (n, n_elec, dim) format. Mitigation: explicit reshape and assert in the proposal's sample method.

### Step 2.3 — Integrate adaptive proposal into training loop
**What:** Add CLI flags `--adaptive-proposal`, `--gmm-components`, `--gmm-refit-every` to `src/run_weak_form.py`. When `--adaptive-proposal` is set, create an `AdaptiveGMMProposal` instance and:
1. Pass it to `importance_resample` each epoch
2. Call `proposal.accumulate(X)` with the resampled points after each epoch
3. Call `proposal.maybe_refit(epoch)` at the appropriate frequency
4. Log the refit events and component statistics

**Files:** `src/run_weak_form.py`
- Add CLI arguments (~line 1240, after existing sampling args)
- Instantiate proposal in setup block (~line 780, after model creation)
- Pass to `importance_resample` call (~line 860)
- Add accumulate + refit calls after resampling (~line 870)

**New CLI arguments:**
```
--adaptive-proposal       Enable adaptive GMM proposal (default: off)
--gmm-components INT      Number of GMM components (default: 8)
--gmm-refit-every INT     Epochs between GMM refits (default: 50)
--gmm-covariance STR      Covariance type: 'full', 'diag', 'tied' (default: 'diag')
```

**Why default to 'diag' covariance:** Full covariance in Nd=40 means fitting 40×41/2 = 820 parameters per component. With K=8 components, that's 6,560 covariance parameters — feasible but risky with only ~5000 buffer samples. Diagonal covariance needs only 40 per component (320 total), is always well-conditioned, and captures the dominant effect (per-coordinate spread). The user can opt into 'full' when they have enough data.

**Acceptance check:**
```bash
python src/run_weak_form.py --n-particles 6 --omega 1.0 --n-coll 256 --epochs 5 --oversample 4 --adaptive-proposal --gmm-components 4 --gmm-refit-every 2 --tag gmm_smoke --lr 1e-3 2>&1 | grep -E '(GMM refit|khat=|ESS=)'
```
→ expected: see `ESS=` and `khat=` every epoch, `[GMM refit]` around epoch 2 and 4

**Risk:** GMM refit adds wall-clock time. Mitigation: with 5000 points in 12D (N=6), sklearn GMM.fit takes <0.1s. Even at 40D with 5000 points, EM is <1s. This is negligible vs the 6–15s/epoch for Ψ evaluation.

### Step 2.4 — Validate: ESS comparison on known-hard regime
**What:** Run a short (200-epoch) side-by-side comparison at N=20, ω=0.1 with and without adaptive proposal. Log ESS and PSIS k-hat. No long training — just confirm the proposal improves sample quality.

**Files:** No code changes. Shell commands only.

**Acceptance check:**
```bash
# Baseline
python src/run_weak_form.py --n-particles 20 --omega 0.1 --n-coll 512 --epochs 200 --oversample 8 --tag ess_baseline_n20w01 --lr 5e-4 2>&1 | tail -n 5

# Adaptive
python src/run_weak_form.py --n-particles 20 --omega 0.1 --n-coll 512 --epochs 200 --oversample 8 --adaptive-proposal --gmm-components 8 --gmm-refit-every 20 --gmm-covariance diag --tag ess_adaptive_n20w01 --lr 5e-4 2>&1 | tail -n 5
```
→ expected: adaptive run shows higher ESS and lower k-hat after the first GMM refit (~epoch 20). Specific target: ESS > 10 (vs baseline ESS ≈ 1–3).

**Decision gate:** If adaptive ESS is not meaningfully higher (>3× baseline) after 200 epochs, stop and diagnose before proceeding. The issue would then be that even |ψ|²-fitted samples don't improve diversity — suggesting the wavefunction itself is too peaked, not just the proposal.

**Risk:** N=20 runs are slow (~7s/epoch on RTX 2080 Ti). 200 epochs ≈ 25 min — acceptable for a gate.

---

## Phase 3 — MinSR Implementation (1–2 sessions, ~2 files)

**Depends on:** Phase 1 complete (PSIS diagnostics). Independent of Phase 2.

**Goal:** Implement the MinSR optimizer (Chen & Heyl 2024) as `sr_mode="minsr"`, solving the SR equation in sample space instead of parameter space. This is the correct approach when P >> M (our standard regime: P ≈ 25K–49K params, M ≈ 500–2048 samples).

**Why MinSR matters even if proposal improves:** Even with perfect sampling, standard CG-SR solves $(O^T O / M + λI) δθ = -F$ which is a P×P system regularized by λ. When P >> M, this system is massively underdetermined. CG converges to *a* solution, but it's not the minimum-norm solution — it depends on initialization and stopping point. MinSR explicitly finds the minimum-norm solution: $δθ = O^T (O O^T / M + λI)^{-1} e$ where $e$ is the centered local energy vector. This is a M×M system — cheap, stable, and unique.

**Key insight for the implementer:** MinSR is almost identical to the Woodbury formula already implemented in `WoodburySR.precondition()`. The difference is: Woodbury computes $S^{-1}F$ (preconditions the gradient), while MinSR computes $O^+ e$ (projects the energy residual directly). The mechanical steps are very similar, but the input is different ($e$ = per-sample centered local energies, not $F$ = averaged gradient).

**Estimated scope:** ~120 lines new code in 1 new class, ~30 lines wiring in 2 files.

### Step 3.1 — Implement `MinSR` class in `src/sr_preconditioner.py`
**What:** A new preconditioner class following the same interface as `WoodburySR` and `CGSR`.

**Files:** `src/sr_preconditioner.py`, new class after `CGSR` (~line 412)

**Core math:**
```
# Given: O (M × P) centered log-derivative matrix, e (M,) centered local energies
# MinSR update:
#   G = O O^T / M + λ I     (M × M matrix)
#   z = G^{-1} e             (M × 1 solve via Cholesky)
#   δθ = -lr * O^T z / M    (project back to parameter space)
```

**Interface:**
```python
class MinSR:
    """Minimum-step Stochastic Reconfiguration (Chen & Heyl 2024).
    
    Solves the SR equation in sample space (M×M) instead of parameter space (P×P).
    Produces the minimum-norm parameter update, which is more stable than CG-SR
    when the number of parameters P greatly exceeds the number of samples M.
    """
    def __init__(self, damping, damping_end, damping_anneal_epochs,
                 max_param_change, trust_region, subsample,
                 center_gradients=True):
        # Same interface as WoodburySR
    
    def update(self, psi_log_fn, x_batch, params) -> dict:
        """Build the O matrix and cache the centered log-derivatives.
        
        Same O-matrix construction as WoodburySR.update().
        ADDITIONALLY: stores per-sample local energies e_k for the MinSR formula.
        """
    
    def set_local_energies(self, E_L: torch.Tensor):
        """Store per-sample local energies from the loss computation.
        
        Called between update() and precondition() by the training loop.
        E_L shape: (M,) — the per-sample local energy from colloc_fd_loss or
        rayleigh_hybrid_loss (both return E_L as their 3rd output).
        """
    
    def precondition(self, params) -> dict:
        """Apply MinSR update: δθ = -O^T (G^{-1} e_centered).
        
        1. Center e: e_bar = e - mean(e)
        2. Form G = O O^T / M + λ I  (M × M)
        3. Solve G z = e_bar via Cholesky
        4. Compute δθ = O^T z / M
        5. Apply trust region and max-change clipping
        6. Write δθ into param.grad for each param
        """
```

**Critical detail:** The standard WoodburySR reads the gradient from `param.grad` after `loss.backward()`. MinSR needs per-sample local energies $E_L$ which are available from the loss function but not from the gradient. The training loop must pass $E_L$ to `minsr.set_local_energies(E_L)` before calling `precondition()`. Both `colloc_fd_loss` and `rayleigh_hybrid_loss` already return `E_L` as their third output — this is not new data, just a new consumer of existing data.

**Acceptance check:**
```bash
python -c "
import torch
from src.sr_preconditioner import MinSR

# Fake model: 2 params
p1 = torch.nn.Parameter(torch.randn(10))
p2 = torch.nn.Parameter(torch.randn(5))
params = [p1, p2]

# Fake psi_log_fn
psi_fn = lambda x: (x.sum(dim=-1) * p1[:x.shape[-1]].sum()).expand(x.shape[0])

minsr = MinSR(damping=1e-3, damping_end=0, damping_anneal_epochs=0,
              max_param_change=0.1, trust_region=1.0, subsample=32,
              center_gradients=True)

x = torch.randn(32, 6, 2)
stats = minsr.update(psi_fn, x, params)
print(f'O shape in stats: {stats.get(\"O_shape\", \"missing\")}')

# Fake local energies
E_L = torch.randn(32)
minsr.set_local_energies(E_L)

# Precondition
pstats = minsr.precondition(params)
print(f'update norm: {pstats.get(\"update_norm\", \"missing\"):.6f}')
print('OK')
"
```
→ expected: prints O_shape, update_norm, `OK`

**Risk:** The O-matrix construction (per-sample backward) is the expensive part and is shared with Woodbury/CG. MinSR does not make this cheaper — it makes the *solve* cheaper and more principled. For P=49K, M=1024, the Woodbury solve is O(M³) ≈ 10⁹ flops either way. The improvement is numerical stability and minimum-norm guarantee, not speed.

### Step 3.2 — Wire MinSR into training loop
**What:** Add `sr_mode="minsr"` option. In the training loop, after computing the loss (which returns E_L), pass E_L to the MinSR preconditioner before calling `precondition()`.

**Files:** `src/run_weak_form.py`
- Add `"minsr"` to sr_mode choices (~line 1303)
- Instantiate MinSR in preconditioner creation block (~line 670)
- After loss computation (~line 890), call `fisher_precond.set_local_energies(E_L_detached)`
- The existing `fisher_precond.precondition(all_trainable)` call handles the rest

**Key: backward() is still needed.** MinSR replaces what goes into `param.grad`, but the training loop still calls `loss.backward()` to populate initial gradients (which MinSR overwrites). This is wasteful but safe. An optimization would be to skip `loss.backward()` when using MinSR, but that's a follow-up — correctness first.

**Acceptance check:**
```bash
python src/run_weak_form.py --n-particles 6 --omega 1.0 --n-coll 256 --epochs 5 --natural-grad --sr-mode minsr --fisher-damping 1e-3 --fisher-subsample 128 --tag minsr_smoke --lr 1e-3 2>&1 | grep -E '(E=|minsr|update_norm)'
```
→ expected: 5 epoch lines with energy values, no errors/NaN

**Risk:** The `set_local_energies` call must happen between `update()` and `precondition()` in the training loop. If the ordering is wrong, MinSR will use stale or missing E_L. Mitigation: add an assertion in `precondition()` that checks E_L was set this epoch.

### Step 3.3 — Validate: MinSR vs CG-SR on N=6, ω=1.0
**What:** Short (500-epoch) comparison at the regime where CG-SR is known to work well (N=6, ω=1.0), to verify MinSR is at least as good. Resume from the same checkpoint.

**Files:** No code changes. Shell commands only.

**Acceptance check:**
```bash
# CG-SR baseline
python src/run_weak_form.py --n-particles 6 --omega 1.0 --n-coll 2048 --epochs 500 --natural-grad --sr-mode cg --sr-cg-iters 20 --fisher-damping 1e-3 --fisher-subsample 512 --tag minsr_gate_cg --lr 5e-3 2>&1 | tail -n 3

# MinSR
python src/run_weak_form.py --n-particles 6 --omega 1.0 --n-coll 2048 --epochs 500 --natural-grad --sr-mode minsr --fisher-damping 1e-3 --fisher-subsample 512 --tag minsr_gate_minsr --lr 5e-3 2>&1 | tail -n 3
```
→ expected: MinSR final energy within 0.1% of CG-SR. If MinSR is dramatically worse, it means the implementation has a bug (the math guarantees equivalence when M < P).

**Decision gate:** If MinSR is >2× worse than CG-SR on this easy regime, debug before proceeding. Check: E_L centering, O-matrix centering, Cholesky stability, gradient-writing logic.

---

## Phase 4 — Controlled 2×2 Ablation (1 session, scripts only)

**Depends on:** Phase 2 and Phase 3 both complete.

**Goal:** Run the decisive experiment: (old-proposal vs adaptive-proposal) × (Adam vs MinSR) at the target hard regime (N=20, ω=0.1). This separates the sampling contribution from the optimizer contribution and produces the scientific deliverable for the thesis.

**Why this is the capstone:** Everything before this is infrastructure. This phase answers the question: "Is the low-omega high-N bottleneck primarily a sampling problem, an optimization problem, or both?"

**Expected outcome matrix:**

| | Fixed proposal | Adaptive proposal |
|---|---|---|
| **Adam** | Baseline (+5.5%) | Sampling effect |
| **MinSR** | Optimizer effect | Combined effect |

- If (adaptive, Adam) >> (fixed, Adam): sampling is dominant → proposal work pays off
- If (fixed, MinSR) >> (fixed, Adam): optimizer is dominant → preconditioner work pays off  
- If both help and combine: both matter (best case for thesis narrative)
- If neither helps: the problem is elsewhere (architecture or loss, Layer 3–4)

**Estimated scope:** 1 launcher script, ~4 GPU-hours per cell (16 GPU-hours total across 4 GPUs).

### Step 4.1 — Create ablation launcher script
**What:** A script that runs 4 jobs in parallel, one per GPU, with identical everything except (proposal, optimizer).

**Files:** `scripts/launch_sampling_ablation.sh` (new)

**Design:**
```bash
# Cell A: fixed proposal + Adam (control)
GPU=0, --lr 5e-4, standard sampling, 2000 epochs

# Cell B: adaptive proposal + Adam
GPU=1, --lr 5e-4, --adaptive-proposal --gmm-components 8 --gmm-refit-every 30, 2000 epochs

# Cell C: fixed proposal + MinSR
GPU=2, --lr 5e-3, --natural-grad --sr-mode minsr --fisher-damping 1e-3, 2000 epochs

# Cell D: adaptive proposal + MinSR
GPU=3, --lr 5e-3, --natural-grad --sr-mode minsr --fisher-damping 1e-3, --adaptive-proposal --gmm-components 8 --gmm-refit-every 30, 2000 epochs
```

All cells: N=20, ω=0.1, n-coll=512, oversample=8, REINFORCE-hybrid loss, same init checkpoint, same seed.

**Acceptance check:**
```bash
bash scripts/launch_sampling_ablation.sh && sleep 60 && for f in outputs/sampling_ablation/*.log; do head -n 3 "$f"; done
```
→ expected: 4 logs showing training started with correct configurations

**Risk:** N=20 MinSR with fisher-subsample=512 means building a 512×25562 O-matrix every epoch (~50MB), then solving a 512×512 system. This should be fine memory-wise (fits in 11GB) but verify s/epoch is acceptable (<20s).

### Step 4.2 — Collect and compare results
**What:** After runs complete (~4–8 hours), extract final eval lines and ESS/PSIS trajectories.

**Files:** No code changes. Analysis commands only.

**Acceptance check:**
```bash
for f in outputs/sampling_ablation/*.log; do
    echo "=== $(basename $f) ==="
    grep -E '(E = .*err =|khat=)' "$f" | tail -n 3
done
```
→ expected: 4 cells with final energy and error percentages. The scientific conclusion follows from comparing the 4 numbers.

**Decision gate after Phase 4:**
- If adaptive proposal is the dominant factor: the thesis narrative is "sampling quality limits collocation methods at high N, and adaptive proposals unlock these regimes."
- If MinSR is the dominant factor: the thesis narrative is "the optimizer must respect the underdetermined geometry, and minimum-norm SR is the right approach."
- If both matter: the thesis narrative is "both proposal and preconditioner must be matched to the sampling regime."
- If neither matters: re-examine architecture (Layer 3) — the Jastrow ansatz may lack capacity at N=20 ω=0.1.

---

## Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| GMM fitting fails in 40D (degenerate covariance) | Medium | Blocks Phase 2 | Use diagonal covariance as default; full covariance only when explicitly requested and enough data exists |
| MinSR Cholesky fails (ill-conditioned G matrix) | Low | Blocks Phase 3 | Damping λ prevents this; same Cholesky used in WoodburySR already works |
| Adaptive proposal doesn't improve ESS meaningfully | Medium | Undermines thesis narrative | Phase 2 Step 2.4 is an explicit gate — detect this early at 200 epochs, not after the full ablation |
| N=20 runs OOM with MinSR + adaptive proposal | Low | Blocks Phase 4 | O-matrix is 512×25K = 50MB; GMM is <1MB. Total <200MB on top of baseline. 11GB GPUs have headroom |
| PSIS k-hat shows sampling is fundamentally broken even with adaptive proposal | Medium | Changes conclusion | This is a valid scientific finding, not a failure. Report it honestly. |
| Phase 4 results are noisy (single seed) | High | Weakens claims | If Phase 4 shows clear signal, follow up with 3-seed replication. If noisy, the single-seed result itself shows the regime is fragile. |

## Success criteria

1. **Phase 1:** Every training run reports PSIS k-hat alongside ESS. k-hat values are consistent with known regime difficulty (low k-hat at N=6 ω=1.0, high k-hat at N=20 ω=0.1).

2. **Phase 2:** Adaptive GMM proposal produces ESS > 3× baseline at N=20, ω=0.1 after 200 epochs. PSIS k-hat drops below 0.7 (from expected >1.0 baseline).

3. **Phase 3:** MinSR produces final energy within 0.1% of CG-SR at N=6, ω=1.0 in a 500-epoch validation run.

4. **Phase 4:** The 2×2 ablation produces a clear ranking with at least one cell showing >30% relative improvement over the control (fixed + Adam) at N=20, ω=0.1.

5. **Overall:** The results are sufficient to write a thesis section distinguishing sampling limitations from optimizer limitations in weak-form collocation.

## Current State
**Active phase:** Phase 1 — PSIS Tail Diagnostics (completed)
**Active step:** Phase boundary reached; waiting for confirmation before Phase 2
**Last evidence:** `CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python3.11 src/run_weak_form.py --n-elec 6 --omega 1.0 --n-coll 256 --epochs 3 --tag psis_smoke 2>&1 | grep -o 'khat=[0-9.]*'` -> `khat=1.09`
**Current risk:** k-hat estimator is a lightweight Hill-style approximation, not full PSIS smoothing
**Next action:** If confirmed, start Phase 2 Step 2.1 (adaptive proposal class)
**Blockers:** none
