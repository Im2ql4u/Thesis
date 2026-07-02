"""Q3 dual-track at the N=2 exact anchor: same ansatz, two paradigms.

Trains an identical CTNN (same init) two ways at N=2, omega=1:
  VMC     : train_vmc_adam        -- |Psi|^2-MCMC, strong-form E_L (the VMC paradigm)
  colloc  : train_collocation_weak -- fixed Gaussian proposal + importance weights, weak-form residual
                                      (the collocation paradigm; never samples |Psi|^2)
and asks the roadmap's dual-track question against exact truth: do the two routes reach the SAME
wavefunction, or just the same energy? Reports overlap^2 with exact, overlap^2 between the two states,
var(E_L), fair d_eff, and the learned pair-Jastrow u(r) vs exact.

Run: CUDA_VISIBLE_DEVICES=0 python3 -u scripts/run_dual_track_N2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from analysis.system import System  # noqa: E402
from analysis import diagnostics as dg  # noqa: E402
from analysis.reference import TwoElectronExact  # noqa: E402
from analysis.train import train_vmc_adam, train_collocation_weak  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/analysis/2026-07-02_dual_track_N2"
OMEGA = 1.0
AKW = dict(node_hidden=16, edge_hidden=16, bottleneck_hidden=8, n_down=1, n_up=1,
           msg_layers=1, node_layers=1, readout_hidden=32, readout_layers=2, act="silu")
STEPS = 900
SEED = 0


def build():
    return System(N=2, omega=OMEGA, d=2, arch="ctnn_vcycle", arch_kwargs=AKW, use_backflow=False, seed=SEED)


def overlap2_to_exact(sysm, exact, n=4096):
    x = sysm.sample(n, steps=400, burn_in=800)
    lp = sysm.log_psi(x).detach().cpu().double().numpy()
    le = exact.log_psi(x.cpu().double().numpy())
    d = le - lp; d -= d.mean(); w = np.exp(d)
    return float((w.mean() ** 2) / (w ** 2).mean()), x


def overlap2_pair(sysA, sysB, x):
    lpA = sysA.log_psi(x).detach().cpu().double().numpy()
    lpB = sysB.log_psi(x).detach().cpu().double().numpy()
    d = lpA - lpB; d -= d.mean(); w = np.exp(d)
    return float((w.mean() ** 2) / (w ** 2).mean())


def var_el(sysm, x):
    return float(dg.local_energy(sysm.log_psi, x, sysm.omega, sysm.params).detach().double().var())


def deff(sysm, x):
    O = dg.build_O(sysm.log_psi, x, [sysm.f_net], center=True)
    return float(dg.kernel_spectrum(O.cpu())["effective_rank"])


def jastrow_curve(sysm, exact):
    r = np.linspace(0.05, 4.0, 40)
    x = np.zeros((r.size, 2, 2)); x[:, 1, 0] = r  # electron 1 slid along x
    xt = torch.tensor(x, device=sysm.device, dtype=sysm.dtype)
    J = (sysm.log_psi(xt) - sysm.log_slater(xt)).detach().cpu().double().numpy()
    Jex = exact.jastrow_log(r)
    J -= J.mean() - Jex.mean()  # align additive constant
    return r.tolist(), J.tolist(), Jex.tolist()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    exact = TwoElectronExact(omega=OMEGA)
    print(f"[dual] exact E={exact.energy:.5f}")

    vmc = build()
    print("[dual] training VMC ...")
    train_vmc_adam(vmc, steps=STEPS, lr=5e-3, batch=2048, log_every=STEPS // 3)
    vmc.eval()

    col = build()
    print("[dual] training collocation (weak form) ...")
    train_collocation_weak(col, steps=STEPS, lr=5e-3, batch=2048, sigma_q=1.3, log_every=STEPS // 3)
    col.eval()

    ov_vmc, xv = overlap2_to_exact(vmc, exact)
    ov_col, xc = overlap2_to_exact(col, exact)
    ov_pair = overlap2_pair(vmc, col, xv)  # on VMC samples
    probe = torch.cat([xv[:768], xc[:768]], dim=0)
    out = {
        "exact_E": exact.energy,
        "vmc":    {"overlap2_exact": ov_vmc, "var_EL": var_el(vmc, xv), "deff": deff(vmc, probe)},
        "colloc": {"overlap2_exact": ov_col, "var_EL": var_el(col, xc), "deff": deff(col, probe)},
        "overlap2_vmc_colloc": ov_pair,
    }
    r, Jv, Jex = jastrow_curve(vmc, exact); _, Jc, _ = jastrow_curve(col, exact)
    out["jastrow"] = {"r": r, "vmc": Jv, "colloc": Jc, "exact": Jex}
    json.dump(out, open(OUT / "summary.json", "w"), indent=2)

    print(f"[dual] overlap^2 with exact:  VMC={ov_vmc:.5f}   colloc={ov_col:.5f}")
    print(f"[dual] overlap^2 VMC<->colloc: {ov_pair:.5f}")
    print(f"[dual] var(E_L): VMC={out['vmc']['var_EL']:.3e}  colloc={out['colloc']['var_EL']:.3e}")
    print(f"[dual] d_eff:    VMC={out['vmc']['deff']:.2f}   colloc={out['colloc']['deff']:.2f}")
    _figure(out)
    print(f"-> {OUT}")


def _figure(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    j = out["jastrow"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(j["r"], j["exact"], "k-", lw=2, label="exact")
    ax.plot(j["r"], j["vmc"], "C0--", label=f"VMC (ov$^2$={out['vmc']['overlap2_exact']:.4f})")
    ax.plot(j["r"], j["colloc"], "C3:", label=f"colloc (ov$^2$={out['colloc']['overlap2_exact']:.4f})")
    ax.set_xlabel("pair distance r"); ax.set_ylabel("Jastrow log u(r)")
    ax.set_title("Dual-track N=2: learned correlation, VMC vs collocation vs exact")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "fig_dual_track.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
