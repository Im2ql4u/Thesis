# Energy Results and Training Method Summary (N=2,6,12,20)

Date: 2026-04-08  
Scope: consolidated **final-evaluation VMC** results (not probe snapshots), compared to DMC references.

---

## 1) Best acquired energies (final-eval VMC) by \(N,\omega\)

**Legend**
- \(E_{\text{VMC}}\): final evaluation energy
- \(\sigma_E\): reported evaluation uncertainty (standard error from final eval)
- Error %: \(\frac{E_{\text{VMC}}-E_{\text{DMC}}}{|E_{\text{DMC}}|}\times 100\)

> Where uncertainty was not yet extracted from older historical logs, it is marked `not yet parsed` (no fabrication).

| N | \(\omega\) | \(E_{\text{VMC}}\) | \(\sigma_E\) | Error vs DMC | Source |
|---|---:|---:|---:|---:|---|
| 2 | 1.0 | 3.000055 | 0.000354 | +0.002% | `outputs/higher_n/phase5_overnight_n2_n20_n12/n2ovr2_w1_20260407_233933.log` |
| 2 | 0.5 | 1.659755 | 0.000224 | -0.001% | `.../n2ovr2_w05_20260407_233933.log` |
| 2 | 0.1 | 0.440750 | 0.000112 | -0.009% | `.../n2ovr2_w01_20260407_233933.log` |
| 2 | 0.01 | 0.073830 | 0.000001 | -0.014% | `.../n2ovr2_w001_20260407_233933.log` |
| 2 | 0.001 | 0.013774 | 0.000001 | +88.689% | `.../n2ovr2_w0001_20260407_233933.log` |
| 6 | 1.0 | not consolidated | — | — | pending canonical extraction |
| 6 | 0.5 | not consolidated | — | — | pending canonical extraction |
| 6 | 0.1 | 3.556558 | not yet parsed | +0.076% | `outputs/consistency_campaign/phase4/eval_matched_20260401_075211.json` |
| 6 | 0.01 | (best found) | not yet parsed | +0.134% | `outputs/consistency_campaign/phase5_probe/p5probe_ng_n6w001_s42_20260401_080353.log` |
| 6 | 0.001 | 0.140956 | 0.000005 | +0.088% | `outputs/higher_n/phase4_n20_lowomega_escalation/n6x2_adam_w0001_20260407_083247.log` |
| 12 | 1.0 | 65.716233 *(best historical err was +0.009% in older run)* | 0.004378 | +0.025% | `outputs/higher_n/phase2/n12_adam_w1_20260406_120525.log` |
| 12 | 0.5 | 39.171354 *(best historical err was +0.024% in older run)* | 0.002644 | +0.030% | `outputs/higher_n/phase2/n12_df_w05_20260406_120525.log` |
| 12 | 0.1 | 12.282442 | 0.000817 | +0.103% | `outputs/higher_n/phase2/n12_adam_w01_20260406_120525.log` |
| 12 | 0.01 | 2.733083 | 0.001902 | +36.654% | `outputs/higher_n/phase4_n20_lowomega_escalation/n12x2_adam_w001_20260407_083247.log` |
| 12 | 0.001 | not run in current validated set | — | — | — |
| 20 | 1.0 | **157.283168** | **0.014853** | **+0.899%** | `outputs/higher_n/phase5_overnight_n2_n20_n12/n20ovr2_w1_20260407_233933.log` |
| 20 | 0.5 | **96.119158** | **0.014549** | **+2.390%** | `.../n20ovr2_w05_20260407_233933.log` |
| 20 | 0.1 | 31.753996 | 0.008050 | +5.925% | `outputs/higher_n/phase4_n20_lowomega_escalation/n20x2_adam_w01_20260407_083247.log` |
| 20 | 0.01 | not supported in current N=20 config | — | — | — |
| 20 | 0.001 | not supported in current N=20 config | — | — | — |

---

## 2) Best picks (current practical frontier)

- **N=20, \(\omega=1.0\): +0.899%** (new best in this campaign family)
- **N=20, \(\omega=0.5\): +2.390%**
- **N=20, \(\omega=0.1\): +5.925%**
- **N=12, \(\omega=0.1\): +0.103%**
- **N=6, \(\omega=0.001\): +0.088%**
- **N=2:** near-DMC at \(\omega\in\{1.0,0.5,0.1,0.01\}\), but failed badly at \(\omega=0.001\)

---

## 3) How we trained these models (what actually made this work)

## 3.1 Ansatz

We used a variational wavefunction \(\psi_\theta(\mathbf{x})\) with:

- **Backflow + Jastrow** for regimes where backflow was enabled (notably \(N\le 12\) in these campaigns)
- **Jastrow-only** for current \(N=20\) runs (as configured in this repo branch)

Intuition: Jastrow captures pairwise correlation cheaply; backflow adds flexible coordinate transforms for nodal/correlation structure when affordable.

## 3.2 Sampling of positions \(x\)

Training points \(\mathbf{x}\) are sampled from a Markov chain targeting \(|\psi_\theta(\mathbf{x})|^2\) (importance-aware workflow in the codebase).  
Practical controls:
- burn-in / decorrelation
- oversampling
- ESS monitoring (critical at higher \(N\), especially low \(\omega\))

Why this matters: if ESS collapses, gradients become high-variance and optimizer improvements can be fake/fragile.

## 3.3 Residual-based learning objective

Core idea: optimize by minimizing a Schrödinger residual signal (not supervised labels).  
Two estimators are used in this codebase:

- **FD-collocation residual** (finite-difference style weak-form/collocation residual), used more in harder/stiffer regimes where it behaved better in prior phases
- **REINFORCE-style residual estimator** (score-function form), used where FD became unstable/noisy

Both are residual-based VMC training, but their gradient estimators have different variance/computation tradeoffs.

## 3.4 Gradients and optimization

- Gradients are computed by autodiff through \(\log\psi_\theta\)-based quantities and residual estimators.
- Optimizers tested: Adam, DiagFisher (and some SR-family runs in prior phases).
- For higher \(N\), **Adam continuation from strong checkpoints** became the most reliable path in practice.

## 3.5 Evaluation protocol

- “VMC result” here means **final end-of-run evaluation** \(E \pm \sigma\), then error vs DMC.
- We explicitly separate this from intermediate probe evaluations.
- Checkpoints are compared by final-eval energy and uncertainty, not training loss alone.

---

## 4) Was this luck, or real?

Short answer: **partly real progress, not yet full robustness**.

Why it is likely real:
- Improvements at \(N=20\) (\(\omega=1.0\): +1.316% \(\rightarrow\) +0.899%) were reproduced by continued training under the same eval protocol.
- Multiple long runs converged in the same direction for \(N=20\) with Adam.

Why luck/noise is still possible:
- Most regimes are still effectively **single-seed** evidence.
- Some “best checkpoint” gains can be sample-noise sensitive without heavier multi-seed confirmation.
- Low-ESS regimes (especially very low \(\omega\)) can make optimistic snapshots unreliable.

Conservative interpretation:
- Treat these as **credible directional gains**, not final consistency claims, until multi-seed and stricter heavy-eval confirmation are completed.

---

## 5) What changed the outcome most

1. Long continuation in tmux with stable resume/checkpoint flow  
2. Regime-specific optimizer/loss choices rather than one global recipe  
3. Strict separation of probe monitoring vs final-eval reporting  
4. Aggressive pruning of underperforming branches (e.g., dropping unhelpful regimes/recipes)