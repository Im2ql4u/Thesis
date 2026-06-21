"""Phase 0 of the collocation-conditioning program: measure kappa(A).

For each converged checkpoint we build, on collocation points drawn from |Psi|^2:
  * S      = O^T O / B   (Fisher / QGT  -- what SR inverts; the VMC operator)
  * A_weak = J_w^T J_w / B   J_w[k,i] = d(1/2|grad logPsi|^2 + V)/dtheta_i   (Rayleigh)
  * A_strong = J_s^T J_s / B J_s[k,i] = d E_L/dtheta_i                       (De Ryck, k^4)
and report each spectrum, its condition number, effective rank, and the log-log power-law
slope of the eigenvalues vs mode index.

Pre-registered: strong-form spectrum decays FASTER (worse kappa) than weak-form (the k^4 vs k^2
claim from the thesis appendix); both far worse than the Fisher S. No training -- load-time only.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analysis import diagnostics as dg  # noqa: E402
from analysis.system import load_system  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# (label, checkpoint) -- quickest (N=2) first so the probe is validated before the heavy N=6 runs.
DEFAULT_CKPTS = [
    ("N2_w1", "results/analysis/2026-06-15_N2_w1_ctnn_acc/checkpoint.pt"),
    ("N6_w1", "results/analysis/2026-06-15_N6_w1_ctnn_big_bf_acc/checkpoint.pt"),
    ("N6_w05", "results/analysis/2026-06-15_N6_w05_ctnn_big_bf_casc/checkpoint.pt"),
    ("N6_w01", "results/analysis/2026-06-15_N6_w01_ctnn_big_bf_casc/checkpoint.pt"),
    ("N6_w001", "results/analysis/2026-06-15_N6_w001_ctnn_big_bf_casc/checkpoint.pt"),
]


def _powerlaw_slope(lam: np.ndarray, rel_tol: float = 1e-8) -> float:
    """Slope of log(lambda) vs log(mode index) over the resolved spectrum (steeper = worse cond).
    This is the floor-independent, theory-faithful metric (lambda_a ~ a^{-s})."""
    lam = np.sort(lam)[::-1]
    supp = lam[lam > lam[0] * rel_tol]
    if supp.size < 8:
        return float("nan")
    k = np.arange(1, supp.size + 1)
    a = slice(1, supp.size - 1)  # drop the very ends (edge effects)
    p = np.polyfit(np.log(k[a]), np.log(supp[a]), 1)
    return float(p[0])


# Collocation proposal widths sigma_f / sqrt(omega) (mirrors run_weak_form.adapt_sigma_fs tiers):
# the mixture deliberately covers low-density / tail / near-node regions the trainer evaluates on.
def _mixture_sigma_fs(omega: float) -> tuple[float, ...]:
    if omega <= 0.01:
        return (0.2, 0.4, 0.6, 1.0, 1.5, 2.5, 4.0, 6.0, 10.0)
    if omega <= 0.05:
        return (0.3, 0.5, 0.8, 1.2, 2.0, 3.5, 6.0)
    if omega <= 0.15:
        return (0.4, 0.7, 1.0, 1.5, 2.5, 4.0)
    return (0.8, 1.3, 2.0)


def _sample_mixture(system, n: int) -> torch.Tensor:
    """Draw n collocation points from the Gaussian mixture q (the measure the colloc trainer uses)."""
    import math
    sfs = _mixture_sigma_fs(float(system.omega))
    nc = len(sfs)
    xs = []
    for i, sf in enumerate(sfs):
        ni = n // nc if i < nc - 1 else n - (n // nc) * (nc - 1)
        s = sf / math.sqrt(float(system.omega))
        xs.append(torch.randn(ni, system.N, system.d, device=system.device, dtype=system.dtype) * s)
    x = torch.cat(xs)
    return x[torch.randperm(x.shape[0], device=x.device)]


def _mixture_logq(system, x: torch.Tensor) -> torch.Tensor:
    """log q(x) of the Gaussian mixture (full mixture density, for importance weights)."""
    import math
    sfs = _mixture_sigma_fs(float(system.omega))
    Nd = system.N * system.d
    xf = x.reshape(x.shape[0], -1)
    comps = []
    for sf in sfs:
        s = sf / math.sqrt(float(system.omega))
        comps.append(-0.5 * Nd * math.log(2 * math.pi * s**2) - xf.pow(2).sum(-1) / (2 * s**2))
    return torch.logsumexp(torch.stack(comps, -1), -1) - math.log(len(sfs))


@torch.no_grad()
def _iw_sqrt_weights(system, x: torch.Tensor) -> torch.Tensor:
    """sqrt of self-normalised importance weights w = |Psi|^2/q (the operator the trainer actually
    sees: down-weights the low-density tail/near-node points). Row-scaling J by these gives the
    weighted Gauss-Newton operator."""
    logw = 2.0 * system.log_psi(x).double() - _mixture_logq(system, x).double()
    logw = logw - logw.max()
    w = torch.exp(logw)
    w = w / w.sum()
    return w.sqrt()


def _kappa(lam: np.ndarray, rel_tol: float = 1e-8) -> float:
    """Condition number over the RESOLVED spectrum (rel_tol above float64 noise, not the 1e-12
    machine floor that pins every kappa at ~1e12)."""
    lam = np.sort(lam)[::-1]
    supp = lam[lam > lam[0] * rel_tol]
    return float(supp[0] / supp[-1]) if supp.size else float("inf")


def analyse(label: str, ckpt: str, n_samples: int, chunk: int, measure: str, store: dict, out: Path) -> None:
    system = load_system(ckpt)
    system.eval()
    if measure == "psi2":
        x = system.sample(n_samples, steps=120, burn_in=400)  # |Psi|^2 (VMC control)
    else:
        x = _sample_mixture(system, n_samples)                # collocation mixture q
    # mixture_iw: scale rows by sqrt(importance weights) -> the weighted operator the trainer sees
    sw = _iw_sqrt_weights(system, x) if measure == "mixture_iw" else None

    O = dg.build_O(system.log_psi, x, system.modules(), center=True)
    Jw = dg.residual_jacobian(system, x, form="weak", chunk=chunk)
    Js = dg.residual_jacobian(system, x, form="strong", chunk=chunk)
    if sw is not None:
        scale = (sw * np.sqrt(x.shape[0])).reshape(-1, 1).to(O)  # keep ~unit scale for readability
        O, Jw, Js = O * scale, Jw * scale, Js * scale

    spectra, slopes, kap = {}, {}, {}
    for name, M in (("S", O), ("A_weak", Jw), ("A_strong", Js)):
        sp = dg.kernel_spectrum(M)
        spectra[name] = sp["eigenvalues"]
        slopes[name] = _powerlaw_slope(sp["eigenvalues"])
        kap[name] = _kappa(sp["eigenvalues"])
        store.setdefault(label, {})[name] = {
            "kappa_resolved": kap[name],          # condition number over the resolved spectrum
            "powerlaw_slope": slopes[name],       # lambda_a ~ a^{-s}: the floor-independent metric
            "effective_rank": sp["effective_rank"],
            "numerical_rank": sp["numerical_rank"],
            "n_params": sp["n_params"],
            "n_samples": sp["n_samples"],
        }
    np.savez(out / f"spectra_{label}.npz", **{k: v for k, v in spectra.items()})

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, col in (("S", "C2"), ("A_weak", "C0"), ("A_strong", "C1")):
        lam = np.sort(spectra[name])[::-1]
        lam = lam[lam > lam[0] * 1e-12] / lam[0]
        ax.loglog(np.arange(1, lam.size + 1), lam, "-", color=col,
                  label=f"{name}  (slope {slopes[name]:+.2f}, kappa {kap[name]:.1e})")
    ax.set_xlabel("mode index"); ax.set_ylabel("eigenvalue / max")
    ax.set_title(f"{label}: residual-operator conditioning"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / f"fig_cond_{label}.png", dpi=140); plt.close(fig)

    print(f"[{label}] kappa(resolved): S={kap['S']:.2e} A_weak={kap['A_weak']:.2e} "
          f"A_strong={kap['A_strong']:.2e}  ratio strong/weak={kap['A_strong']/kap['A_weak']:.2f}  "
          f"slopes weak={slopes['A_weak']:+.2f} strong={slopes['A_strong']:+.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=192)
    ap.add_argument("--chunk", type=int, default=24)
    ap.add_argument("--measure", type=str, default="psi2", choices=["psi2", "mixture", "mixture_iw"])
    ap.add_argument("--only", type=str, default=None, help="comma-separated labels to run")
    a = ap.parse_args()

    ckpts = DEFAULT_CKPTS
    if a.only:
        keep = set(a.only.split(","))
        ckpts = [(l, c) for (l, c) in DEFAULT_CKPTS if l in keep]

    out = Path(f"results/analysis/{date.today().isoformat()}_conditioning_A")
    out.mkdir(parents=True, exist_ok=True)
    store: dict = {}
    print(f"[conditioning_A] measure={a.measure}")
    for label, ckpt in ckpts:
        if not Path(ckpt).exists():
            print(f"[{label}] MISSING {ckpt}"); continue
        analyse(f"{label}_{a.measure}", ckpt, a.samples, a.chunk, a.measure, store, out)
        (out / f"summary_{a.measure}.json").write_text(json.dumps(store, indent=2) + "\n")
    print(f"[conditioning_A] wrote {out}")


if __name__ == "__main__":
    main()
