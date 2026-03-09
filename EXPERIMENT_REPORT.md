# Experiment Report: Variational Quantum-Dot Wavefunction Optimization

**System:** N=6 electrons, d=2 dimensions, ω=1.0 harmonic trap  
**Target:** E_DMC = 20.15932 (diffusion Monte Carlo reference)  
**Method:** Collocation/residual training with Huber loss on local energy residuals (NOT VMC energy minimization)  
**Date:** March 2026

---

## 1. Goal

Achieve a variational energy E < 20.16 for the 6-electron quantum dot at ω=1.0, 
matching or beating the DMC energy E_DMC = 20.15932. The wavefunction ansatz is:

    ψ(x) = sign · exp( log|det Slater| + f(x) )

where the Slater determinant uses non-interacting harmonic oscillator orbitals 
(not Hartree-Fock), and f(x) is a learned Jastrow factor. All training uses 
collocation-based residual optimization: minimize Huber(E_L - E_eff) over 
screened collocation points, with exact Laplacians. PyTorch CPU, float64.

---

## 2. What Has Been Tested

### Phase 1: Jastrow Architecture Comparison

Tested 7 Jastrow architectures via collocation training (all starting from scratch):

| Architecture          | Params  | E (colloc) | Notes                     |
|-----------------------|---------|------------|---------------------------|
| PINN (large MLP)      | ~80k    | ~20.30     | Baseline, poor             |
| DeepSet                | ~15k    | ~20.25     | Set-equivariant            |
| CTNN (flat)            | ~20k    | ~20.22     | Copresheaf GNN, 1-level    |
| CTNN BF-style          | ~22k    | ~20.21     | BF-borrowed architecture   |
| CTNN Attention          | ~25k    | ~20.23     | Global attention variant   |
| **CTNNJastrowVCycle**  | **25,562** | **20.196** | **V-cycle multigrid GNN — WINNER** |
| Triadic DeepSet        | ~18k    | ~20.24     | Channel-triadic variant    |

**Winner: CTNNJastrowVCycle** — a multigrid V-cycle copresheaf GNN with 
coarse→fine→coarse message passing. 25,562 parameters.

Heavy VMC re-evaluation (3 chains × 16,000 samples):
- **E = 20.1881 ± 0.0019** (err = +0.143%)

### Phase 2: Coordinate Backflow + Jastrow

Added backflow transformation x_eff = x + Δx(x; θ) before the Slater determinant.
The Jastrow was loaded from the pre-trained V-cycle checkpoint and fine-tuned jointly.

| Backflow Type        | BF Params | Total | Best VMC Probe | Final VMC E        |
|----------------------|-----------|-------|----------------|--------------------|
| MLP BackflowNet      | 9,155     | 34,717| 20.167 (0.04%) | 20.179 ± 0.003 (+0.097%) |
| **CTNN BackflowNet** | **41,987**| **67,549** | **20.166 (0.03%)** | **20.173 ± 0.003 (+0.067%)** |

**Best result: CTNN BF + V-cycle Jastrow → E = 20.173 ± 0.003 (+0.067% above DMC)**

Key observation: the backflow scale parameter (bf_scale) barely moves during training
(0.10 → 0.097), suggesting the coordinate-backflow mechanism is underperforming.
The backflow provides only ~0.015 improvement over Jastrow-alone.

### Phase 3: Orbital Backflow (Current Experiment)

**Hypothesis:** The remaining gap to DMC is in the **nodal surface position** of the
Slater determinant. Coordinate backflow shifts electron positions before orbital 
evaluation, which changes nodes indirectly. Orbital backflow directly modifies the 
orbital matrix values, giving more direct control over nodal surfaces.

Instead of x_eff = x + Δx → Slater(x_eff), orbital backflow computes:

    Ψ̃[i,k] = Ψ[i,k] · (1 + scale · δ[i,k](x; θ))

where δ is the output of an OrbitalBackflowNet (copresheaf GNN, 42,052 params).

#### What worked:
- **Multiplicative perturbation** (Ψ → Ψ·(1+δ)) is stable because near nodal 
  surfaces where Ψ→0, the perturbation also vanishes.
- **Scheduled bf_scale ramp** (0.001 → 0.10 over 200 epochs) prevents early divergence.
- VMC probe at epoch 50: **E = 20.190** (0.15% above DMC) — competitive with Jastrow baseline.
- VMC probe at epoch 100: **E = 20.189** (0.15%) — stable and slightly better.

#### What didn't work:
- **Additive perturbation** (Ψ → Ψ + δ) diverges catastrophically. Near nodal 
  surfaces, even small additive δ creates huge log|Ψ| changes → NaN gradients.
  The training appeared to produce good VMC probes (20.19 at epoch 80) but the 
  final heavy evaluation failed badly (E = 22.3), confirming the instability.
- **Large initial bf_scale** (0.10) with additive perturbation diverges within 10 epochs.
- **bf_scale = 0.01 additive** is too conservative → barely perturbs the wavefunction,
  trivially recovers Jastrow-only performance, early stops.

#### Current status:
Training of multiplicative orbital backflow was in progress with encouraging early 
results. The bf_scale ramp was at ~0.05 (epoch 100 of 200 ramp), with the network 
still learning. Full 900-epoch run was interrupted for this cleanup/commit.

---

## 3. Hypotheses

### H1: The gap is in nodal surface position (NOT topology)
The non-interacting Slater determinant has the correct nodal topology for a 
closed-shell N=6 system but the wrong nodal surface positions. Orbital backflow 
can continuously deform these surfaces to improve the energy. If this hypothesis 
is correct, orbital backflow should close the remaining +0.067% gap.

### H2: Multiplicative orbital perturbation is the right formulation
Additive perturbation destroys log-domain stability near nodes. Multiplicative 
perturbation Ψ·(1+δ) maintains the sign structure and scales naturally with 
orbital values. The scheduled ramp prevents early instability.

### H3: The Jastrow is essentially converged
Heavy re-evaluation shows V-cycle Jastrow at E = 20.188 (+0.143%). Adding 
coordinate backflow only brings it to 20.173 (+0.067%). The remaining error is 
in the **determinantal** part of the wavefunction, not the Jastrow correlation factor.

### H4 (Fallback): Pfaffian wavefunction may be needed
If orbital backflow cannot close the gap, the nodal topology itself may need 
to change. A Pfaffian wavefunction (which subsumes the Slater determinant) 
allows different nodal topology and could reach DMC accuracy.

---

## 4. Summary of Results

| Method                           | Params  | E (VMC)          | Error vs DMC |
|----------------------------------|---------|------------------|--------------|
| Jastrow only (V-cycle)           | 25,562  | 20.188 ± 0.002   | +0.143%      |
| + MLP coord backflow             | 34,717  | 20.179 ± 0.003   | +0.097%      |
| + CTNN coord backflow            | 67,549  | 20.173 ± 0.003   | +0.067%      |
| + Orbital BF (multiplicative)    | 67,614  | 20.189 ± ? (ep100)| +0.15% (early)|
| **DMC reference**                | —       | **20.15932**     | **0.000%**   |

---

## 5. Architecture Overview

### Core files:
- `src/PINN.py` — All network architectures (BackflowNet, CTNNBackflowNet, OrbitalBackflowNet, UnifiedCTNN)
- `src/jastrow_architectures.py` — Jastrow architectures (CTNNJastrowVCycle, etc.)
- `src/functions/Neural_Networks.py` — `psi_fn()` dispatcher (unified, orbital BF, legacy paths)
- `src/functions/Slater_Determinant.py` — Basis functions, orbital matrix, slogdet
- `src/functions/Energy.py` — VMC energy evaluation

### Training scripts:
- `src/run_colloc_archs.py` — Architecture comparison (Phase 1)
- `src/run_colloc_bf_jastrow.py` — Coordinate BF + Jastrow (Phase 2)
- `src/run_colloc_orbital_bf.py` — Orbital backflow + Jastrow (Phase 3)

### Evaluation:
- `src/reeval_checkpoint.py` — Heavy multi-chain VMC re-evaluation
- `src/reeval_orbital_bf.py` — Re-evaluation for orbital BF checkpoints

### Checkpoints (results/arch_colloc/):
- `ctnn_vcycle.pt` — Best Jastrow-only (E=20.196 train, 20.188 reeval)
- `bf_mlp_vcycle.pt` — MLP BF + Jastrow (E=20.179)
- `bf_ctnn_vcycle2.pt` — CTNN BF + Jastrow (E=20.173)
- `orb_bf_vcycle.pt` — Orbital BF additive (failed, E=22.3)

---

## 6. Next Steps

1. **Complete multiplicative orbital backflow run** — restart the training with 
   the full 900-epoch schedule and evaluate.
2. **If orbital BF reaches <20.18**: Fine-tune jointly with Jastrow unfreezing.
3. **If orbital BF plateaus >20.17**: Implement Pfaffian wavefunction (H4 fallback).
4. **Consider node-targeted collocation**: Weight collocation points near the 
   nodal surface to drive improvements where they matter most.
