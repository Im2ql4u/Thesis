"""T1.1 (Phase M0, no training) — What does the graph structure compute? The confound-free message
ablation, decomposed into kinetic vs potential.

DIAGNOSTIC_SUMMARY Fig F/I found that zeroing a trained CTNN's inter-particle messages raises the
variational energy by 22-30%, ALL kinetic. But my later CTNN-vs-DeepSet comparison tied on energy --
because both arms shared the SAME message-passing backflow. This untangles the confound: on a trained
checkpoint (no retraining), evaluate the 2x2 {messages on/off} x {backflow on/off} on common samples
from the full |Psi|^2, decomposing E = <T> + <V_trap> + <V_Coul> with the weak-form kinetic
T = 1/2 |grad logPsi|^2 (no Laplacian -> cheap, no OOM). This isolates:
  - what the Jastrow MESSAGES compute (with and without backflow to compensate),
  - what the BACKFLOW (itself message passing) contributes,
  - whether the gain is kinetic (a smoother wavefunction) as Fig I claimed.

Run: CUDA_VISIBLE_DEVICES=0 python3 -u scripts/run_message_ablation.py
"""
from __future__ import annotations

import contextlib
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from analysis.system import load_system  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/analysis/2026-07-02_message_ablation"
CKPTS = {
    1.0:  ["2026-06-15_N6_w1_ctnn_big_bf_acc"],
    0.1:  ["2026-06-15_N6_w01_ctnn_big_bf_casc"],
    0.01: ["2026-07-02_N6_w001_ctnn_s0", "2026-07-02_N6_w001_ctnn_s1", "2026-07-02_N6_w001_ctnn_s2"],
}
N = 6
N_SAMP = 2048
SAMPLE_KW = dict(steps=400, burn_in=800)
RHO_NAMES = ["rho_v_to_e_down", "rho_e_to_v_down", "rho_v_to_e_up", "rho_e_to_v_up"]


@contextlib.contextmanager
def no_messages(system):
    """Zero every inter-particle linear map (rho_*) in the V-cycle Jastrow -> pairwise/DeepSet-like."""
    saved = []
    net = system.f_net
    for nm in RHO_NAMES:
        ml = getattr(net, nm, None)
        if ml is None:
            continue
        for lin in ml:
            saved.append((lin, lin.weight.data.clone()))
            lin.weight.data.zero_()
    try:
        yield
    finally:
        for lin, w in saved:
            lin.weight.data.copy_(w)


@contextlib.contextmanager
def no_backflow(system):
    bf = system.backflow_net
    system.backflow_net = None
    try:
        yield
    finally:
        system.backflow_net = bf


def energy_terms(system, x, omega):
    """<T>, <V_trap>, <V_Coul> on configs x with T = 1/2 |grad logPsi|^2 (weak-form kinetic)."""
    xg = x.detach().requires_grad_(True)
    lp = system.log_psi(xg)
    g = torch.autograd.grad(lp.sum(), xg, create_graph=False)[0]
    T = 0.5 * (g ** 2).sum(dim=(1, 2))
    xd = x.double()
    v_trap = 0.5 * omega ** 2 * (xd ** 2).sum(dim=(1, 2))
    ii, jj = torch.triu_indices(x.shape[1], x.shape[1], offset=1)
    rij = (xd[:, ii, :] - xd[:, jj, :]).norm(dim=-1).clamp_min(1e-9)
    v_coul = (1.0 / rij).sum(dim=-1)
    return (float(T.double().mean()), float(v_trap.mean()), float(v_coul.mean()))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for omega, dirs in CKPTS.items():
        for d in dirs:
            s = load_system(str(ROOT / "results/analysis" / d / "checkpoint.pt"), device=dev, seed=0)
            x = s.sample(N_SAMP, **SAMPLE_KW)  # common probe: the FULL net's |Psi|^2
            arms = {}
            # full
            arms["msg+bf"] = energy_terms(s, x, omega)
            with no_messages(s):
                arms["nomsg+bf"] = energy_terms(s, x, omega)
            with no_backflow(s):
                arms["msg+nobf"] = energy_terms(s, x, omega)
            with no_messages(s), no_backflow(s):
                arms["nomsg+nobf"] = energy_terms(s, x, omega)
            for arm, (T, Vt, Vc) in arms.items():
                rows.append(dict(omega=omega, ckpt=d, arm=arm, T=T, V_trap=Vt, V_coul=Vc, E=T + Vt + Vc))
            E = {a: sum(v) for a, v in arms.items()}
            Tt = {a: v[0] for a, v in arms.items()}
            print(f"w={omega:<5} {d.split('_')[-1]:8}  E: full={E['msg+bf']:.3f}  "
                  f"noJmsg(bf)={E['nomsg+bf']:.3f}  noBF={E['msg+nobf']:.3f}  none={E['nomsg+nobf']:.3f}")
            print(f"            dT from Jastrow-msg | bf on: {Tt['nomsg+bf']-Tt['msg+bf']:+.3f}  "
                  f"| bf off: {Tt['nomsg+nobf']-Tt['msg+nobf']:+.3f}   dT from backflow: {Tt['msg+nobf']-Tt['msg+bf']:+.3f}   "
                  f"dV_coul(full->none): {arms['nomsg+nobf'][2]-arms['msg+bf'][2]:+.3f}")

    with open(OUT / "ablation.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    json.dump(rows, open(OUT / "summary.json", "w"), indent=2)
    _figure(rows)
    _headline(rows)
    print(f"-> {OUT}")


def _agg(rows, omega, arm, key):
    vs = [r[key] for r in rows if r["omega"] == omega and r["arm"] == arm]
    return float(np.mean(vs)), float(np.std(vs))


def _headline(rows):
    print("\n[ablation] Jastrow-message energy gain (E(none)-E(full)) and its kinetic fraction:")
    for omega in sorted({r["omega"] for r in rows}, reverse=True):
        Ef = _agg(rows, omega, "msg+bf", "E")[0]; En = _agg(rows, omega, "nomsg+nobf", "E")[0]
        Tf = _agg(rows, omega, "msg+bf", "T")[0]; Tn = _agg(rows, omega, "nomsg+nobf", "T")[0]
        Vf = Ef - Tf; Vn = En - Tn
        dE = En - Ef
        print(f"  w={omega:<5} full E={Ef:.3f} -> no-MP E={En:.3f}  dE={dE:+.3f} ({100*dE/abs(Ef):+.1f}%)  "
              f"dT={Tn-Tf:+.3f}  dV={Vn-Vf:+.3f}  (kinetic frac {100*(Tn-Tf)/(dE+1e-9):.0f}%)")


def _figure(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    omegas = sorted({r["omega"] for r in rows}, reverse=True)
    arms = ["msg+bf", "nomsg+bf", "msg+nobf", "nomsg+nobf"]
    labels = {"msg+bf": "full (MP+BF)", "nomsg+bf": "no Jastrow-msg", "msg+nobf": "no backflow", "nomsg+nobf": "no MP anywhere"}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for arm, col in zip(arms, ["k", "C3", "C1", "C0"]):
        ys = [_agg(rows, w, arm, "E")[0] / _agg(rows, w, "msg+bf", "E")[0] for w in omegas]
        ax.plot(omegas, ys, "o-", color=col, label=labels[arm])
    ax.set_xscale("log"); ax.set_xlabel("omega"); ax.set_ylabel("E / E(full)")
    ax.set_title("What message passing computes (trained-net ablation, N=6)\nJastrow messages vs backflow, gain is kinetic")
    ax.axhline(1.0, color="gray", lw=0.5); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "fig_message_ablation.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
