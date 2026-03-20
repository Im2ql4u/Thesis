# Experiment Report: Weak-Form Search From Scratch to 20.161314

**System:** N=6 electrons, d=2 dimensions, harmonic trap $\omega = 1.0$  
**Reference:** $E_{\mathrm{DMC}} = 20.15932$  
**Best archived result:** $E = 20.161314 \pm 0.002342$  
**Checkpoint:** `results/arch_colloc/bf_hardfocus_v1b.pt`

---

## 1. Executive Summary

This report is meant to answer two questions at once:

1. What was the **training philosophy** behind the weak-form program in this repo?
2. What sequence of **concrete experiments** produced the best N=6, $\omega=1.0$ result $20.161314$?

The short answer is that the best result did **not** come from a single architectural jump. It came from a process:

- define a collocation objective that is cheaper and more directed than full VMC-only optimization
- learn which $x$ regions are worth spending collocation budget on
- discover that some richer ansatz families were harder to optimize than the simpler BF+Jastrow family
- find that pure REINFORCE-style weak-form training was more reliable than mixed direct-gradient variants for the good BF runs
- and then refine an already-good BF+Jastrow state through a continuation chain rather than trying to reproduce the result in one shot

The canonical lineage is:

```text
ctnn_vcycle.pt
  -> bf_ctnn_vcycle.pt
  -> bf_joint_reinf_v3.pt
  -> bf_resume_lr_v1.pt
  -> bf_hardfocus_v1b.pt
```

The best reported number,

$$
E = 20.161314 \pm 0.002342,
$$

comes from the **heavy 30k-sample exact VMC re-evaluation** of `bf_hardfocus_v1b.pt`. That final re-evaluation, not the noisy collocation batch estimate and not the sparse online VMC probe, is the authoritative metric.

---

## 2. Problem Setting and Evaluation Philosophy

The task is to optimize fermionic trial wavefunctions for a trapped Coulomb system. The trial state is built from a Slater core plus learnable correlation structure:

$$
\Psi(x) = \Psi_{\mathrm{Slater}}(x; \text{orbitals/backflow}) \; e^{J(x)}.
$$

There are three distinct numbers that appear during training, and confusing them is the fastest way to misunderstand the history:

1. **Collocation batch energy** `E=...` printed during training. This is a noisy local estimate over the current collocation set.
2. **Sparse online VMC probe** `vmc=...` done every fixed number of epochs. This is used for cheap checkpoint selection.
3. **Final heavy exact VMC** at 30k samples. This is the number that should be quoted in the report.

Several experiments looked excellent on cheap probes and then regressed under heavy evaluation. That happened even for runs that ended up important. So the final heavy VMC is not a cosmetic post-processing step. It changes the ranking.

---

## 3. Weak-Form Philosophy

The core idea was not to chase DMC directly with a black-box optimizer. The idea was to use the structure of the Schrödinger equation to create a cheaper, more local signal for improvement.

### 3.1 Why weak-form collocation

Direct VMC-only optimization is expensive, noisy, and can spend many updates just rediscovering obvious local structure. Weak-form collocation tries to make training more surgical:

- evaluate the wavefunction and its derivatives at chosen configurations $x$
- compute a local-energy-like signal
- penalize inconsistency of the ansatz with the target eigenstate structure
- and only occasionally pay for a more expensive VMC measurement

The weak-form energy-like quantity in the shared utilities is

$$
\widetilde{E}(x) = \tfrac12 \lVert \nabla \log \Psi(x) \rVert^2 + V(x),
$$

while the exact local energy used in the hybrid objective is

$$
E_L(x) = -\tfrac12 \big(\Delta \log \Psi(x) + \lVert \nabla \log \Psi(x) \rVert^2\big) + V(x).
$$

The point of the weak-form program was never that these local objectives are themselves the final scientific answer. The point was that they give a dense optimization signal that can move the ansatz into a better variational basin before the expensive exact VMC judgment step.

### 3.2 Loss families that were tried

Three loss styles matter historically.

#### A. Finite-difference residual loss

This computes $E_L$ through graph-safe finite differences and penalizes its spread around the batch mean:

$$
L_{\mathrm{FD}} \approx \mathbb E\big[\rho(E_L - \bar E_L)\big],
$$

where $\rho$ is either squared residual or Huber. This was appealing because it looks close to a direct PDE residual minimization. In practice it was often brittle, especially for larger or richer ansätze.

#### B. Hybrid REINFORCE + direct weak-form loss

The most important objective family was

$$
L = 2\,\mathbb E\big[(E_L - \bar E_L)\log |\Psi|\big] + \beta\,\mathbb E[\tilde E].
$$

In code, `direct_weight = β`. The historically successful BF chain set

$$
\beta = 0,
$$

so in practice the good runs were **pure REINFORCE-style weak-form** runs with robust local-energy clipping.

#### C. Variants around batch size, ESS, or variance shaping

The `bf_micro_*`, `bf_resume_ess_v1`, and `bf_var_v1` branches were all attempts to keep the same basic BF+Jastrow ansatz but alter the optimization geometry: smaller batches, ESS-aware behavior, variance emphasis, or more focused continuation schedules. These were useful near-misses and in a few cases landed very close to the final best energy.

### 3.3 Why clipping mattered

The hybrid loss used `clip_el = 5.0` in the successful chain. This clips local-energy outliers relative to robust median and MAD statistics. That was important because raw collocation batches can include rare pathological points where the derivative-based signal is numerically valid but optimization-wise destructive.

The successful recipe is therefore philosophically conservative:

- keep the ansatz expressive but not exotic
- use a stochastic estimator with some robustness
- and avoid letting a few bad collocation points dominate the gradient

---

## 4. How We Chose the $x$ Points

The choice of sampling strategy mattered almost as much as the loss.

### 4.1 Plain Gaussian sampling

The simplest option was isotropic Gaussian sampling with width proportional to $1 / \sqrt{\omega}$. This is cheap and trap-aware, but it wastes samples in low-weight regions once the model becomes structured.

### 4.2 Mixture-of-Gaussians screening

The main historical scripts moved to a mixture sampler with scale factors like

$$
\sigma_f \in \{0.8, 1.3, 2.0\}, \qquad x \sim \mathcal N(0, \sigma_f^2 / \omega).
$$

This gives three useful regimes in one batch:

- tighter center samples
- moderate bulk samples
- broader tail samples

### 4.3 Screened collocation

The next improvement was not merely to sample from the mixture, but to **oversample** and then keep the points that look relevant under the current ansatz. In the old architecture-screening and BF scripts, this meant:

- draw roughly `oversample × n_coll` candidates
- score them with an approximate log-importance ratio $2\log|\Psi(x)| - \log q(x)$
- keep the highest-scoring subset
- reserve a small exploration fraction so the process does not collapse too quickly

That is a practical compromise between two bad extremes:

- sampling too broadly and wasting most of the batch
- sampling only near current high-density regions and becoming myopic

### 4.4 Importance resampling and later generalized samplers

The shared utilities now expose a cleaner `importance_resample` path that approximates $|\Psi|^2$ via multinomial resampling from a Gaussian mixture. Later infrastructure also introduced more structured stratified samplers with shells, dimers, clusters, and replayed hard points.

Historically, though, the best N=6 BF path was built on the simpler mixture-plus-screening idea. That matters because the winning recipe was not the most elaborate sampler in the repo. It was the one whose bias matched the optimization problem well enough to stabilize the good basin.

---

## 5. Ansatz Families We Tried

The repo explored several different families. The search was not only “deeper network = better result”. In fact, one of the main lessons was that moderate expressivity plus good optimization often beat more ambitious ansätze with weaker training behavior.

### 5.1 Jastrow-only architectures

The Jastrow-only search was the starting point for the later BF work.

The architecture-screening script compared several families:

- `CTNNBackflowStyleJastrow`
- `CTNNJastrow`
- `CTNNJastrowVCycle`
- `CTNNJastrowAttnGlobal`
- `DeepSetJastrow`
- `TriadicDeepSetJastrow`

The quick screen stored in `results/arch_colloc/summary.json` selected `CTNNJastrowVCycle` as the most promising new Jastrow candidate with:

$$
E = 20.214443 \pm 0.007443.
$$

That result by itself was not spectacular, but it identified the architecture family that later mattered most. The script header also records that the already-known PINN-style baseline was around

$$
20.210 \pm 0.003,
$$

so the architectural screen was not a dramatic breakthrough in energy. Its real contribution was selecting a good backbone with a manageable parameter count and stable behavior.

### 5.2 Jastrow-only weak-form continuation

The main pure Jastrow continuation that matters historically is `jas_reinf_v2`.

Configuration:

- `CTNNJastrowVCycle`
- 25,562 parameters
- `n_coll = 4096`
- pure REINFORCE-style hybrid loss with `direct_weight = 0.0`
- `clip_el = 5.0`

Final heavy exact VMC:

$$
E = 20.220938 \pm 0.003192.
$$

This is one of the most instructive failures in the archive. The best sparse probe during training was much lower, around `20.16880`, but the final heavy evaluation came out substantially worse. That taught two things:

- the Jastrow-only family could get online probes very close to DMC
- but those cheap probes were not reliable enough to justify claiming a true $20.16x$ Jastrow-only result

In other words, Jastrow-only training was useful, but it did not ultimately explain the best N=6 number.

### 5.3 Coordinate backflow plus Jastrow

This is the family that won.

The historical BF architecture is not enormous:

- Jastrow: 25,562 parameters
- backflow: 23,811 parameters
- total trainable in the main BF runs: 49,373

That scale matters. It was expressive enough to capture the improvement beyond Jastrow-only models, but still trainable enough that continuation strategy, collocation budget, and clipping were able to dominate the outcome.

### 5.4 Orbital backflow

There is code for orbital backflow training and it reflects a real line of exploration: perturb orbital matrices directly rather than only shifting coordinates. Conceptually it is attractive because it changes the nodal structure more explicitly.

But the archived N=6 result story is not an orbital-backflow success story. The decisive documented improvements in `results/arch_colloc/` came from the simpler coordinate-backflow plus V-cycle Jastrow family, not from orbital backflow.

### 5.5 Pfaffian / neural Pfaffian branch

The Pfaffian branch was the boldest architectural attempt. In principle it should offer more expressive antisymmetry than the BF+Jastrow ansatz. In practice it underperformed in this training regime.

Representative results:

- `npf_joint_reinf_v1`: $20.260770 \pm 0.004063$
- `npf_fd_v6`: $20.245231 \pm 0.004566$
- `npf_fd_v5`: $20.360910 \pm 0.015774$

This is a useful negative result. The more expressive ansatz did not win, because the loss and sampling pipeline were not yet good enough to exploit that expressivity reliably. The easier BF+Jastrow family ended up being the scientifically productive one.

---

## 6. Representative Results Across the Search

The table below is intentionally selective. It is not every single run in `results/arch_colloc/`, but it covers the major branches and the runs that changed the direction of the project.

| Branch | What it tested | Final heavy exact VMC |
|---|---|---:|
| `ctnn_vcycle` quick screen | Best new Jastrow architecture from architecture sweep | `20.214443 ± 0.007443` |
| `jas_reinf_v2` | Long pure-Jastrow REINFORCE continuation | `20.220938 ± 0.003192` |
| `hybrid_bf_v1` | Early BF+Jastrow hybrid run | `20.185454 ± 0.002671` |
| `hybrid_bf_v2` | Improved BF+Jastrow hybrid run | `20.176684 ± 0.002363` |
| `bf_fd_v2` | BF trained with FD-style collocation residual | `20.203739 ± 0.002272` |
| `bf_more_v1` | More BF continuation without the final successful schedule | `20.178438 ± 0.002418` |
| `bf_var_v1` | Variance-oriented continuation | `20.167766 ± 0.005974` |
| `bf_resume_ess_v1` | ESS-focused continuation | `20.171354 ± 0.002481` |
| `bf_micro_v1` | Micro-batch focused BF run | `20.162272 ± 0.006871` |
| `bf_micro_v4` | Best near-miss micro branch | `20.161549 ± 0.003958` |
| `bf_joint_reinf_v3` | Stage A of the winning continuation chain | `20.175464 ± 0.002490` |
| `bf_resume_lr_v1` | Stage B low-LR continuation | `20.170112 ± 0.002574` |
| `bf_hardfocus_v1b` | Stage C hardfocus continuation | `20.161314 ± 0.002342` |
| `npf_joint_reinf_v1` | Joint neural Pfaffian continuation | `20.260770 ± 0.004063` |
| `npf_fd_v6` | FD-trained neural Pfaffian | `20.245231 ± 0.004566` |

Three patterns stand out.

1. Jastrow-only runs got close enough to be misleading, but not truly competitive under heavy evaluation.
2. BF+Jastrow was clearly the right ansatz family.
3. Within BF+Jastrow, optimization path mattered more than adding a more exotic model class.

---

## 7. What We Learned From the Failed Branches

### 7.1 The richer ansatz did not automatically win

The Pfaffian branch is the clearest example. More expressive antisymmetry was available, but optimization quality got worse, not better. This is a strong argument that for this problem, training geometry was the real bottleneck.

### 7.2 Finite-difference residual training was usually too brittle

The BF and Pfaffian FD runs were useful diagnostic tools, but they were not the route to the best energies. They tended to be more sensitive to noise, more vulnerable to unstable local-energy behavior, and less consistently reproducible than the REINFORCE-style hybrid runs.

### 7.3 Cheap VMC probes can flatter bad checkpoints

`jas_reinf_v2` is the strongest warning case, but it is not the only one. Probe numbers are useful for online selection, not for scientific claims. The archive only makes sense once heavy 30k-sample re-evaluation is treated as the final judge.

### 7.4 Small-batch and focused branches found the right basin before the canonical chain was recognized

The `bf_micro_*` family, especially `bf_micro_v1` and `bf_micro_v4`, showed that the BF+Jastrow ansatz could indeed reach the $20.16x$ basin. What they did not provide was a clean, reproducible explanation of how to stay there and improve reliably. The final chain supplied that missing structure.

---

## 8. How the Best Result Was Actually Obtained

This is the part that must be reproducible.

The best result was **not** a one-shot BF run from `bf_ctnn_vcycle.pt`. It was a continuation method.

### 8.1 Pre-history: why `bf_ctnn_vcycle.pt` matters

The BF chain starts from `bf_ctnn_vcycle.pt`, which already contains a coordinated BF+Jastrow state. That means the best result is not “what happens when we initialize the historical architecture and train once”. It is “what happens when we keep training a specific, already-useful state in the right way”.

### 8.2 Stage A: `bf_joint_reinf_v3`

This stage resumed the BF+Jastrow state and trained both parts jointly with the pure REINFORCE-style hybrid objective.

Settings:

- `epochs = 900`
- `n_coll = 4096`
- `lr = 5e-4`
- `lr_jas = 5e-5`
- `direct_weight = 0.0`
- `clip_el = 5.0`

Final heavy VMC:

$$
20.175464 \pm 0.002490.
$$

This is not yet the best result, but it is the first decisive move into the right basin.

### 8.3 Stage B: `bf_resume_lr_v1`

This stage kept the same ansatz and almost the same recipe, but lowered the learning rates while preserving the continuation trajectory.

Settings:

- resume from `bf_joint_reinf_v3.pt`
- `epochs = 500`
- `n_coll = 4096`
- `lr = 2e-4`
- `lr_jas = 2e-5`
- `direct_weight = 0.0`
- `clip_el = 5.0`

Final heavy VMC:

$$
20.170112 \pm 0.002574.
$$

This is the stage where the method stopped being merely “promising BF” and became “continuation into a refined basin”.

### 8.4 Stage C: `bf_hardfocus_v1b`

This is the final step. It resumed from `bf_resume_lr_v1.pt`, kept the lower learning rates, kept pure REINFORCE-style hybrid training, and increased the collocation budget.

Settings:

- resume from `bf_resume_lr_v1.pt`
- `epochs = 500`
- `n_coll = 6144`
- `lr = 2e-4`
- `lr_jas = 2e-5`
- `direct_weight = 0.0`
- `clip_el = 5.0`

Best sparse online VMC probe during training:

$$
20.16533.
$$

Final heavy exact VMC of the restored checkpoint:

$$
20.161314 \pm 0.002342.
$$

This is the best archived result.

### 8.5 Why this worked

The result is best understood as the combination of four ingredients.

1. **Right ansatz family.** BF+V-cycle-Jastrow was expressive enough but still trainable.
2. **Right loss.** Pure REINFORCE-style hybrid weak-form training with outlier clipping was more robust than the seemingly more direct alternatives.
3. **Right sampling pressure.** Oversampled, screened collocation points spent budget where the wavefunction mattered instead of diffusing effort everywhere.
4. **Right continuation path.** The optimizer needed Stage A to find the basin, Stage B to settle into it, and Stage C to sharpen it with a larger collocation set.

This is why the result cannot be honestly summarized as “BF + Jastrow reaches 20.161”. The real statement is more specific:

> BF + V-cycle Jastrow, trained with pure REINFORCE-style weak-form continuation and a staged low-LR hardfocus schedule, reaches $20.161314 \pm 0.002342$ for N=6, $\omega=1.0$.

---

## 9. What Did Not Reproduce It

The most important failed modern approximation was the one-shot rerun from `bf_ctnn_vcycle.pt` that used the wrong loss weighting and clipping:

- `direct_weight = 0.1`
- `clip_el = 4.0`
- no faithful Stage A $\to$ Stage B $\to$ Stage C continuation

That route landed around

$$
20.189580 \pm 0.002825,
$$

which is not close enough to count as a reproduction.

This matters because it proves that the success was not just due to architecture identity. The continuation path itself is part of the method.

---

## 10. Practical Reproduction Notes

If the historical checkpoints already exist, the minimal faithful reconstruction is to rerun the final hardfocus stage from `bf_resume_lr_v1.pt` using the current thin runner `src/run_collocation.py`.

If the goal is to reproduce the full history, rerun the chain in order:

```text
bf_ctnn_vcycle.pt
  -> bf_joint_reinf_v3
  -> bf_resume_lr_v1
  -> bf_hardfocus_v1b
```

The critical hyperparameters to preserve are:

- `direct_weight = 0.0`
- `clip_el = 5.0`
- `lr = 5e-4`, `lr_jas = 5e-5` in Stage A
- `lr = 2e-4`, `lr_jas = 2e-5` in Stages B and C
- `n_coll = 4096` in Stages A and B
- `n_coll = 6144` in Stage C
- heavy final exact VMC evaluation for the reported number

---

## 11. Bottom Line

From scratch, the project tried better Jastrows, multiple BF schedules, finite-difference residual training, hybrid REINFORCE training, micro-batch variants, ESS and variance-focused resumptions, and a more ambitious Pfaffian branch.

The winning lesson was unexpectedly simple:

- do not over-read cheap probes
- do not assume the fanciest ansatz will win
- spend effort on sampling and optimization geometry
- and treat continuation lineage as part of the scientific recipe

That is how the repo got from a broad weak-form exploration program to the concrete archived result

$$
20.161314 \pm 0.002342.
$$

#### Stage A

```bash
python src/run_collocation.py \
  --mode bf \
  --n-elec 6 \
  --omega 1.0 \
  --epochs 900 \
  --n-coll 4096 \
  --oversample 8 \
  --micro-batch 512 \
  --lr 5e-4 \
  --lr-jas 5e-5 \
  --grad-clip 1.0 \
  --clip-el 5.0 \
  --direct-weight 0.0 \
  --vmc-every 50 \
  --seed 42 \
  --resume results/arch_colloc/bf_ctnn_vcycle.pt \
  --tag bf_joint_reinf_v3_rebuild
```

#### Stage B

```bash
python src/run_collocation.py \
  --mode bf \
  --n-elec 6 \
  --omega 1.0 \
  --epochs 500 \
  --n-coll 4096 \
  --oversample 8 \
  --micro-batch 512 \
  --lr 2e-4 \
  --lr-jas 2e-5 \
  --grad-clip 1.0 \
  --clip-el 5.0 \
  --direct-weight 0.0 \
  --vmc-every 50 \
  --seed 42 \
  --resume results/arch_colloc/bf_joint_reinf_v3_rebuild.pt \
  --tag bf_resume_lr_v1_rebuild
```

#### Stage C

```bash
python src/run_collocation.py \
  --mode bf \
  --n-elec 6 \
  --omega 1.0 \
  --epochs 500 \
  --n-coll 6144 \
  --oversample 8 \
  --micro-batch 512 \
  --lr 2e-4 \
  --lr-jas 2e-5 \
  --grad-clip 1.0 \
  --clip-el 5.0 \
  --direct-weight 0.0 \
  --vmc-every 50 \
  --seed 42 \
  --resume results/arch_colloc/bf_resume_lr_v1_rebuild.pt \
  --tag bf_hardfocus_v1b_rebuild
```

Then use the heavy final exact VMC evaluation from the runner output as the headline metric.

---

## 9. Expected Landmarks During Rebuild

If a rebuild is following the correct path, the logs should look qualitatively like this:

- Stage A: long noisy continuation, early stop around the high hundreds, best probe around `20.164`
- Stage B: lower-LR continuation, early stop around `300`, best probe around `20.158`
- Stage C: same low LR but `n_coll = 6144`, final heavy eval near `20.161`

If instead the run settles near `20.189`, that usually means one of these happened:

- the chain started directly from `bf_ctnn_vcycle.pt` and skipped Stages A/B
- `direct_weight` was left at `0.1`
- `clip_el` was left at `4.0` or `0.0`
- the wrong checkpoint was resumed
- BF architecture reconstruction failed and the run was not truly using the historical BF weights

---

## 10. Final Bottom Line

The historical best N=6, $\omega=1.0$ BF result in this repo was obtained by:

- starting from the existing CTNN BF+Jastrow checkpoint
- performing a long pure-REINFORCE continuation
- lowering LR and continuing again
- then doing one more continuation with a larger collocation set

The final authoritative number is:

$$
E = 20.161314 \pm 0.002342
$$

from:

```text
results/arch_colloc/bf_hardfocus_v1b.pt
```

Anyone trying to reproduce that number should treat the checkpoint chain itself as part of the method. It was not an incidental detail.
