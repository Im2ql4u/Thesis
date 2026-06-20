"""The decisive SR-vs-Adam mechanism experiment (N=2, exact ground truth).

Controlled: a short common Adam warm-up, then branch into Adam-only vs SR-only from the IDENTICAL
checkpoint; 5 seeds; tuned SR. We log, on a fixed probe set, vs step:
  * energy error and distance-to-exact ||delta|| (delta = log|Psi_exact| - log|Psi|, centred)
  * the SOFT vs STIFF decomposition of delta in the NTK (S) eigenbasis (modes split at the median
    eigenvalue): does SR remove the stiff (low-eigenvalue) error that Adam stalls on?

Pre-registered outcomes (both publishable):
  (+) SR reaches lower ||delta||/energy and specifically collapses the STIFF-mode error -> proves
      what SR does to the gradients and where.
  (0) SR ~ Adam when both tuned -> SR's advantage is regime-confined (report the null).
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
from analysis.fast_sr import train_sr  # noqa: E402
from analysis.reference import TwoElectronExact  # noqa: E402
from analysis.system import System  # noqa: E402
from analysis.train import train_vmc_adam  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

AKW = dict(node_hidden=16, edge_hidden=16, bottleneck_hidden=8, n_down=1, n_up=1,
           msg_layers=1, node_layers=1, readout_hidden=32, readout_layers=2, act="silu")


def make_diag(system, exact, x_fix, logex, Vfix, mu_supp_idx, store, label, seed):
    """Cheap closure: log distance-to-exact and its soft/stiff NTK-mode split, using a FIXED
    precomputed eigenbasis Vfix (one forward pass per call; no per-step O rebuild)."""
    lo, hi = mu_supp_idx  # index ranges (low-mu stiff half, high-mu soft half) into Vfix columns

    @torch.no_grad()
    def diag(step):
        logp = system.log_psi(x_fix).double()
        delta = logex - logp
        delta = delta - delta.mean()
        c = Vfix.t() @ delta                     # delta in the fixed NTK eigenbasis (ascending mu)
        stiff = float((c[lo[0]:lo[1]] ** 2).sum().sqrt())
        soft = float((c[hi[0]:hi[1]] ** 2).sum().sqrt())
        x = system.sample(512, steps=100, burn_in=200)   # light variational energy
        E = dg.local_energy(system.log_psi, x, system.omega, system.params, lap_mode="exact")
        E = E[torch.isfinite(E)]
        store.append({"opt": label, "seed": seed, "step": step,
                      "energy_err_pct": float((E.mean() - exact.energy) / exact.energy * 100),
                      "dist_exact": float(delta.norm() / np.sqrt(delta.numel())),
                      "soft_err": soft, "stiff_err": stiff})
    return diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--warm", type=int, default=80)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--every", type=int, default=40)
    a = ap.parse_args()
    from analysis.system import load_system
    exact = TwoElectronExact(omega=1.0)
    # Fixed probe set + fixed NTK eigenbasis from the CONVERGED N=2 checkpoint (consistent reference)
    ref = load_system("results/analysis/2026-06-15_N2_w1_ctnn_acc/checkpoint.pt")
    x_fix = torch.randn(512, 2, 2, device=ref.device, dtype=ref.dtype)
    logex = torch.tensor(exact.log_psi(x_fix.detach().cpu().double().numpy()),
                         device=ref.device, dtype=torch.float64)
    Oref = dg.build_O(ref.log_psi, x_fix, ref.modules(), center=True).double()
    mu, Vfix = torch.linalg.eigh(Oref @ Oref.t())   # ascending mu; columns = NTK eigenfunctions
    nsupp = int((mu > float(mu.max()) * 1e-10).sum())
    P = mu.shape[0]
    half = nsupp // 2
    idx = ((P - nsupp, P - nsupp + half), (P - half, P))  # (stiff low-mu range, soft high-mu range)
    print(f"[SRmech] fixed NTK basis: supported rank={nsupp}, stiff/soft split at median mu")

    store = []
    for seed in range(a.seeds):
        torch.manual_seed(seed); np.random.seed(seed)
        s = System(N=2, omega=1.0, arch="ctnn_vcycle", arch_kwargs=AKW, seed=seed)
        train_vmc_adam(s, steps=a.warm, lr=5e-3, batch=2048, log_every=a.warm)  # common warm-up
        init = {k: v.detach().cpu().clone() for k, v in s.f_net.state_dict().items()}

        diag = make_diag(s, exact, x_fix, logex, Vfix, idx, store, "adam", seed)
        train_vmc_adam(s, steps=a.steps, lr=5e-3, batch=2048, log_every=a.steps,
                       ckpt_every=a.every, ckpt_fn=diag)
        s.f_net.load_state_dict({k: v.to(s.device) for k, v in init.items()})  # identical branch point
        diag = make_diag(s, exact, x_fix, logex, Vfix, idx, store, "sr", seed)
        train_sr(s, steps=a.steps, batch=1024, lr=0.2, lr_final=0.01, damping=1e-2,
                 damping_final=1e-4, max_step=0.05, max_step_final=0.005, sr_samples=256,
                 log_every=a.steps, diag_every=a.every, diag_fn=diag, ref_energy=3.0)
        print(f"[seed {seed}] done")

    out = Path(f"results/analysis/{date.today().isoformat()}_SRmech_N2")
    out.mkdir(parents=True, exist_ok=True)
    (out / "raw.json").write_text(json.dumps(store, indent=2) + "\n")
    _summarize(out, store)


def _summarize(out, store):
    import collections
    steps = sorted({r["step"] for r in store})
    agg = {}
    for opt in ("adam", "sr"):
        agg[opt] = {}
        for key in ("energy_err_pct", "dist_exact", "soft_err", "stiff_err"):
            m, lo, hi = [], [], []
            for st in steps:
                vals = np.array([r[key] for r in store if r["opt"] == opt and r["step"] == st])
                m.append(float(np.mean(vals))); lo.append(float(np.mean(vals) - np.std(vals)))
                hi.append(float(np.mean(vals) + np.std(vals)))
            agg[opt][key] = {"mean": m, "lo": lo, "hi": hi}
    (out / "summary.json").write_text(json.dumps({"steps": steps, "agg": agg}, indent=2) + "\n")

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for opt, col in (("adam", "C1"), ("sr", "C0")):
        for ax, key, ttl in zip(axs, ("energy_err_pct", "dist_exact", "stiff_err"),
                                ("energy err %", "||delta|| to exact", "STIFF-mode error")):
            m = np.array(agg[opt][key]["mean"]); lo = np.array(agg[opt][key]["lo"]); hi = np.array(agg[opt][key]["hi"])
            ax.plot(steps, m, "-o", color=col, label=opt)
            ax.fill_between(steps, lo, hi, color=col, alpha=0.2)
            ax.set_title(ttl); ax.set_xlabel("step"); ax.legend()
    axs[2].set_yscale("log")
    fig.tight_layout(); fig.savefig(out / "fig_sr_mechanism.png", dpi=140); plt.close(fig)
    print("[SRmech] energy err (final): adam=%.3f%% sr=%.3f%%" %
          (agg["adam"]["energy_err_pct"]["mean"][-1], agg["sr"]["energy_err_pct"]["mean"][-1]))
    print("[SRmech] stiff-mode error (final): adam=%.3e sr=%.3e" %
          (agg["adam"]["stiff_err"]["mean"][-1], agg["sr"]["stiff_err"]["mean"][-1]))


if __name__ == "__main__":
    main()
