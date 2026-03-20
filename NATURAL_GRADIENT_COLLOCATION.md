# Natural Gradient for Collocation Learning

Design rationale for the diagonal Fisher preconditioner integrated into the
weak-form collocation trainer.

---

## 1. The Problem

The collocation training pipeline uses a REINFORCE gradient estimator:

$$\nabla_\theta L = 2\,\mathbb{E}\big[(E_L - R)\,\nabla_\theta \log\Psi\big]$$

optimized with Adam. This combination has a fundamental conditioning problem.

### What Adam does wrong here

Adam normalizes each parameter update by the running second moment of the
gradient:

$$v_i \approx \mathbb{E}\big[((E_L - R) \cdot \partial_i \log\Psi)^2\big]$$

This **conflates two independent noise sources**:

1. **Physics noise**: $\text{Var}(E_L)$ — dominated by near-node and
   near-coalescence configurations where the Laplacian and $\text{SD}^{-1}$
   blow up.
2. **Model geometry**: the curvature of the parameter space, captured by the
   Fisher information $\mathbb{E}[(\partial_i \log\Psi)^2]$.

When $E_L$ has heavy tails (which it always does near nodes), Adam's denominator
$v_i$ is inflated by outliers. This suppresses the effective step size for ALL
updates — including the informative ones. The `clip_el` clipping helps, but it
is a blunt instrument that discards signal along with noise.

### Evidence from our runs

- **VMC/SR works with the same ansatz.** SR uses the Fisher (overlap) matrix to
  precondition — it separates physics signal from model geometry.
- **Continuation chains are needed.** The 4-stage chain to 20.161 is manual
  annealing to compensate for the optimizer's inability to navigate the
  landscape on its own. A well-conditioned optimizer should converge in one run.
- **Low ω and large N fail.** These are precisely the regimes where
  $\text{Var}(E_L)$ grows and importance sampling quality degrades — making
  Adam's noise conflation maximally destructive.

---

## 2. The Fix: Diagonal Natural Gradient

### What SR does

Stochastic Reconfiguration (SR) separates the two noise sources:

- **Overlap matrix**: $S_{ij} = \text{Cov}(\partial_i \log\Psi,\, \partial_j \log\Psi)$
  — pure model geometry (Fisher information)
- **Force vector**: $f_i = \text{Cov}(E_L,\, \partial_i \log\Psi)$
  — pure physics signal
- **Update**: $\delta\theta = S^{-1} f$ — physics signal preconditioned by
  model geometry

### Why diagonal

For ~50K parameters, the full $S$ matrix is $50\text{K} \times 50\text{K}$
(2.5 billion entries) — too large. We use the diagonal approximation:

$$F_{ii} = \mathbb{E}\big[(\partial_i \log\Psi)^2\big]$$

This gives each parameter its own curvature scaling, independent of $E_L$ noise.
It does not capture cross-parameter correlations (which full SR or KFAC would),
but it already fixes the core issue: **Adam's denominator no longer includes
$\text{Var}(E_L)$.**

### Comparison

| Optimizer | Denominator for param $i$ | What it conflates |
|-----------|--------------------------|-------------------|
| Adam | $\mathbb{E}[(E_L - R)^2 \cdot O_i^2]$ | Physics noise × Fisher |
| Diagonal Fisher + SGD | $\mathbb{E}[O_i^2] + \lambda$ | Fisher only |
| Full SR | $(S + \lambda I)^{-1}$ row $i$ | Nothing (exact natural gradient) |

where $O_i = \partial_i \log\Psi$.

---

## 3. How the Fisher Diagonal is Estimated

Computing $F_{ii} = \mathbb{E}[O_i^2]$ exactly requires per-sample parameter
gradients — expensive with standard backprop.

We use **Hutchinson's trace estimator**: for a Rademacher random vector
$v \in \{-1, +1\}^B$,

$$\mathbb{E}_v\Big[\Big(\sum_k v_k \frac{\partial \log\Psi(x_k)}{\partial \theta_i}\Big)^2\Big] = \sum_k \Big(\frac{\partial \log\Psi(x_k)}{\partial \theta_i}\Big)^2$$

So each probe (one backward pass of $\sum_k v_k \log\Psi(x_k)$ w.r.t. $\theta$)
gives an **unbiased estimate** of $\sum_k O_{k,i}^2$ for ALL parameters
simultaneously.

### Cost

- **Per probe**: 1 forward pass + 1 backward pass through $\log\Psi$
- **Default**: 4 probes on a subsample of 256 points
- **Relative to existing cost**: the Laplacian computation in `rayleigh_hybrid_loss`
  already does $N \times d$ backward passes (12 for N=6, 40 for N=20).
  Fisher estimation adds 4 backward passes on a smaller batch — negligible.

### Smoothing

The Fisher estimate is smoothed across epochs with an exponential moving average
(EMA decay = 0.95, ≈20 epoch memory). This prevents noisy single-epoch
estimates from destabilizing the preconditioner. Bias correction (as in Adam)
is applied during the first epochs.

---

## 4. Implementation

### Files

- **`src/fisher_preconditioner.py`**: `DiagonalFisherPreconditioner` class.
  Standalone module with no dependencies beyond PyTorch.
- **`src/run_weak_form.py`**: Integration via new CLI flags. When
  `--natural-grad` is set, Adam is replaced with SGD+momentum and the Fisher
  preconditioner is inserted between `loss.backward()` and `optimizer.step()`.

### Training loop integration

```
for each epoch:
    # 1. Importance-resample collocation points from Gaussian mixture (unchanged)
    X = importance_resample(psi_log_fn, ...)

    # 2. Accumulate REINFORCE gradient over micro-batches (unchanged)
    for xb in microbatches(X):
        L = rayleigh_hybrid_loss(psi_log_fn, xb, ...)
        (L / n_microbatch).backward()

    # 3. NEW: Update Fisher estimate and precondition gradients
    fisher.update(psi_log_fn, X, all_params)     # Hutchinson probes
    fisher.precondition(all_params)               # .grad /= (F + λ)

    # 4. Clip and step (unchanged except SGD replaces Adam)
    clip_grad_norm_(all_params, max_norm)
    optimizer.step()
```

### CLI flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--natural-grad` | off | Enable diagonal Fisher preconditioning |
| `--fisher-damping` | 1e-3 | Tikhonov regularization λ |
| `--fisher-ema` | 0.95 | EMA decay for running Fisher |
| `--fisher-probes` | 4 | Hutchinson probes per update |
| `--fisher-subsample` | 256 | Points subsampled for Fisher |
| `--fisher-max` | 1e6 | Upper clamp on Fisher entries |
| `--nat-momentum` | 0.9 | SGD momentum |

### Suggested first experiment

```bash
python src/run_weak_form.py \
  --mode bf \
  --n-elec 6 --omega 1.0 \
  --epochs 500 \
  --n-coll 4096 --oversample 8 --micro-batch 512 \
  --natural-grad \
  --lr 3e-2 --lr-jas 3e-3 \
  --fisher-damping 1e-3 \
  --grad-clip 1.0 \
  --clip-el 5.0 --direct-weight 0.0 \
  --vmc-every 50 --seed 42 \
  --resume results/arch_colloc/bf_ctnn_vcycle.pt \
  --tag natgrad_v1
```

Note the higher learning rates (3e-2 vs 5e-4). The Fisher preconditioning
already normalizes the update magnitude per parameter, so the optimizer needs
a larger base step size. The right range is typically 1e-2 to 1e-1 for natural
gradient, compared to 1e-4 to 1e-3 for Adam.

---

## 5. What This is NOT

This is **not VMC/SR**. The critical differences:

| | VMC/SR | Natural gradient collocation |
|-|--------|------------------------------|
| **Sampling** | MCMC from $\|\Psi\|^2$ | Importance resampling from Gaussian mixture |
| **Loss** | $\delta\theta = S^{-1} \text{Cov}(E_L, O)$ | REINFORCE loss with Fisher-preconditioned SGD |
| **Laplacian** | Forward-only, exact | Forward-only in REINFORCE, not backpropped |
| **Fisher** | Full overlap matrix $S$ via CG | Diagonal approximation via Hutchinson |
| **Optimizer** | Direct SR step (no momentum/schedule) | SGD + momentum + cosine schedule |

The sampling distribution is the key difference. VMC/SR samples from $\|\Psi\|^2$
using MCMC — that IS variational Monte Carlo. Our collocation approach samples
from a fixed Gaussian mixture proposal and importance-reweights. This keeps
the method in the collocation/PINN family.

---

## 6. Expected Impact

### What should improve

- **Single-run convergence for N=6, ω=1.0.** The continuation chain was needed
  because Adam couldn't navigate the landscape. Fisher preconditioning gives
  each parameter an appropriately-scaled update, reducing the need for manual
  LR staging.
- **Fewer rollbacks.** Adam's noise conflation causes spurious large updates
  that trigger rollbacks. Natural gradient updates are geometrically meaningful.
- **Better N=12 behavior.** The 12→40D configuration space makes gradient
  conditioning worse, but the Fisher is dimension-agnostic — it preconditions
  each parameter individually regardless of N.

### What this does NOT fix

- **Sampling quality at low ω or large N.** The importance resampling from
  Gaussian mixture still degrades exponentially with system size. Fisher
  preconditioning helps the optimizer use whatever gradient signal exists,
  but it cannot create signal where sampling provides none.
- **Nodal surface issues.** Near nodes, both $E_L$ and $O_i$ blow up. Fisher
  preconditioning reduces the damage (by not letting $\text{Var}(E_L)$ pollute
  the denominator), but the fundamental problem of ill-conditioned gradients
  through $\text{slogdet}^{-1}$ near nodes remains.

### Path to further improvement

If diagonal Fisher proves insufficient, the natural next steps are:

1. **Block-diagonal Fisher (KFAC)**: Kronecker-factored approximation, per
   layer. Standard in FermiNet/PauliNet. More accurate than diagonal, moderate
   implementation cost.
2. **Low-rank Fisher**: Keep top-$k$ eigenvectors of $S$ via randomized SVD.
   Captures the dominant curvature directions that diagonal misses.
3. **Adaptive damping**: Levenberg-Marquardt style — increase $\lambda$ when
   step quality is poor, decrease when good. Currently $\lambda$ is fixed.

---

## 7. Why This Should Work (Theoretical Grounding)

The REINFORCE gradient $g = 2\mathbb{E}[(E_L - R) O]$ is the **score function
estimator** of the energy gradient. Its variance is:

$$\text{Var}(g_i) = 4\,\mathbb{E}[(E_L - R)^2 O_i^2] - 4\,(\mathbb{E}[(E_L - R) O_i])^2$$

The first term dominates. Expanding:

$$\mathbb{E}[(E_L - R)^2 O_i^2] = \mathbb{E}[(E_L - R)^2] \cdot \mathbb{E}[O_i^2] + \text{Cov}((E_L - R)^2, O_i^2)$$

Adam divides by the running estimate of $\mathbb{E}[g_i^2] \approx 4\mathbb{E}[(E_L-R)^2] \cdot F_{ii}$.
Natural gradient divides by $F_{ii}$ only. The ratio of effective step sizes is:

$$\frac{\text{Adam step}_i}{\text{NatGrad step}_i} \sim \frac{1}{\sqrt{\mathbb{E}[(E_L - R)^2]}}$$

When $\text{Var}(E_L)$ is large (near nodes, low ω, large N), Adam's step is
suppressed by $1/\sqrt{\text{Var}(E_L)}$ relative to natural gradient. This is
exactly the regime where we struggle.
