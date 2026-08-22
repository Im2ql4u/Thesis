"""T1.5 (Phase M0) — Is the V-cycle a scale separator? Does the coarse bottleneck encode GLOBAL
collective coordinates while the fine passes encode LOCAL structure?

The production Jastrow is a multigrid: fine message passing -> coarsen to a 16-dim bottleneck
(node_down/edge_down) -> refine. The physicist's hypothesis for WHY the graph structure helps is that
the bottleneck carries the global/collective coordinate (breathing, shell reorganisation) while the
fine node/edge features carry local pair structure -- i.e. the net separates scales the way the physics
does. We test it directly by hooking the fine (node_embed/edge_embed) and coarse (node_down/edge_down)
layers and linear-probing each against GLOBAL vs LOCAL observables (R^2), across omega, on existing
CTNN checkpoints (no training).

Run: CUDA_VISIBLE_DEVICES=0 python3 -u scripts/run_scale_separation.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from analysis.system import load_system  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/analysis/2026-07-02_scale_separation"
CKPTS = {
    1.0:  "2026-06-15_N6_w1_ctnn_big_bf_acc",
    0.1:  "2026-06-15_N6_w01_ctnn_big_bf_casc",
    0.01: "2026-07-02_N6_w001_ctnn_s0",
}
N_PROBE = 1024
SAMPLE_KW = dict(steps=400, burn_in=800)
FINE = ["node_embed", "edge_embed"]
COARSE = ["node_down", "edge_down"]


def _r2(feats: np.ndarray, target: np.ndarray) -> float:
    """Linear-probe R^2 of target from feats (ridge, centered)."""
    X = feats - feats.mean(0, keepdims=True)
    y = target - target.mean()
    lam = 1e-3 * (X.shape[0])
    XtX = X.T @ X + lam * np.eye(X.shape[1])
    beta = np.linalg.solve(XtX, X.T @ y)
    pred = X @ beta
    ss_res = float(((y - pred) ** 2).sum()); ss_tot = float((y ** 2).sum()) + 1e-30
    return max(0.0, 1.0 - ss_res / ss_tot)


def global_targets(x: torch.Tensor):
    """Per-config GLOBAL collective scalars (B,)."""
    xf = x.double()
    r2 = (xf ** 2).sum(-1)              # (B,N)
    ii, jj = torch.triu_indices(x.shape[1], x.shape[1], offset=1)
    rij = (xf[:, ii, :] - xf[:, jj, :]).norm(dim=-1)  # (B,P)
    return {
        "breathing_totr2": r2.sum(-1).cpu().numpy(),       # sum_i r_i^2 (monopole/breathing)
        "radial_spread": r2.std(-1).cpu().numpy(),          # shell spread
        "total_coulomb": (1.0 / rij.clamp_min(1e-6)).sum(-1).cpu().numpy(),
        "mean_pairdist": rij.mean(-1).cpu().numpy(),
        "max_radius": xf.norm(dim=-1).max(-1).values.cpu().numpy(),
    }


def local_targets_node(x: torch.Tensor, omega: float):
    """Per-particle LOCAL scalars (B*N,)."""
    from analysis.representation import physical_local_targets
    B, N, _ = x.shape
    spin = torch.cat([torch.zeros(N // 2, dtype=torch.long), torch.ones(N - N // 2, dtype=torch.long)])
    t = physical_local_targets(x, spin, omega)
    return {k: v.reshape(-1).cpu().numpy() for k, v in t.items()}  # (B*N,)


def hook_layers(system, x, names):
    net = system.f_net
    cap = {}
    hooks = []
    for nm in names:
        m = getattr(net, nm, None)
        if m is not None:
            hooks.append(m.register_forward_hook(lambda _m, _i, o, nm=nm: cap.__setitem__(nm, o.detach())))
    _ = system.log_psi(x)
    for h in hooks:
        h.remove()
    return cap


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for omega, d in CKPTS.items():
        s = load_system(str(ROOT / "results/analysis" / d / "checkpoint.pt"), device=dev, seed=0)
        x = s.sample(N_PROBE, **SAMPLE_KW)
        cap = hook_layers(s, x, FINE + COARSE)
        G = global_targets(x)
        Lnode = local_targets_node(x, omega)
        B, N = x.shape[0], x.shape[1]
        for nm, act in cap.items():
            H = act.shape[-1]
            per_cfg = act.reshape(B, -1, H).mean(1).cpu().double().numpy()   # (B,H) config-level
            eff = _effrank(per_cfg)
            r2_glob = {g: _r2(per_cfg, G[g]) for g in G}
            # node layers also get a per-particle local decode
            r2_loc = {}
            if act.dim() == 3 and act.shape[1] == N:  # (B,N,H) node features
                per_part = act.reshape(B * N, H).cpu().double().numpy()
                r2_loc = {k: _r2(per_part, Lnode[k]) for k in Lnode}
            stage = "coarse" if nm in COARSE else "fine"
            rows.append(dict(omega=omega, layer=nm, stage=stage, eff_rank=eff, n_channels=int(H),
                             global_r2=r2_glob, local_r2=r2_loc,
                             global_best=max(r2_glob.values()), global_best_target=max(r2_glob, key=r2_glob.get),
                             local_best=(max(r2_loc.values()) if r2_loc else None)))
            gb = rows[-1]["global_best"]; lb = rows[-1]["local_best"]
            print(f"  w={omega:<5} {stage:6} {nm:11} eff_rank={eff:.2f}  "
                  f"global_R2(best)={gb:.2f}[{rows[-1]['global_best_target']}]  "
                  f"local_R2(best)={('%.2f'%lb) if lb is not None else '--'}")
    json.dump(rows, open(OUT / "summary.json", "w"), indent=2)
    _figure(rows)
    # headline: coarse-vs-fine global-encoding at each omega
    print("\n[scale-sep] does the bottleneck encode the GLOBAL collective better than the fine layers?")
    for omega in CKPTS:
        cg = np.mean([r["global_best"] for r in rows if r["omega"] == omega and r["stage"] == "coarse"])
        fg = np.mean([r["global_best"] for r in rows if r["omega"] == omega and r["stage"] == "fine"])
        print(f"  w={omega:<5} coarse global_R2={cg:.2f}  vs  fine global_R2={fg:.2f}  "
              f"({'coarse>fine (scale sep.)' if cg > fg else 'no separation'})")
    print(f"-> {OUT}")


def _effrank(feats):
    X = feats - feats.mean(0, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False); lam = s ** 2
    return float((lam.sum() ** 2) / (lam ** 2).sum()) if lam.sum() > 0 else 0.0


def _figure(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    omegas = sorted({r["omega"] for r in rows}, reverse=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for stage, col in (("coarse", "C3"), ("fine", "C0")):
        gb = [np.mean([r["global_best"] for r in rows if r["omega"] == w and r["stage"] == stage]) for w in omegas]
        a1.plot(omegas, gb, "o-", color=col, label=f"{stage} (bottleneck)" if stage == "coarse" else f"{stage} (embed)")
    a1.set_xscale("log"); a1.set_xlabel("omega"); a1.set_ylabel("global collective R^2 (best)")
    a1.set_title("T1.5: does the bottleneck encode the global coordinate?"); a1.legend(); a1.set_ylim(0, 1.02)
    for stage, col in (("coarse", "C3"), ("fine", "C0")):
        er = [np.mean([r["eff_rank"] for r in rows if r["omega"] == w and r["stage"] == stage]) for w in omegas]
        a2.plot(omegas, er, "s-", color=col, label=stage)
    a2.set_xscale("log"); a2.set_xlabel("omega"); a2.set_ylabel("effective rank")
    a2.set_title("intrinsic dimension of coarse vs fine features"); a2.legend()
    fig.tight_layout(); fig.savefig(OUT / "fig_scale_separation.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
