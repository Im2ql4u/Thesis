"""Mechanistic depth analysis (vision doc s2.5: D1, D3, D4, D6).

Trains a closed-shell NQS (Adam-VMC, single optimizer with mid-flight checkpoints), then opens the
box on the trained ground state:

  D1  NTK eigenmodes in real space + delta_psi_SR vs delta_psi_plain (where the natural-gradient
      update lands)
  D2  cusp decomposition  J = u_cusp(fixed prior) + f_net(learned)
  D3  effective variational coordinate (perturb the dominant NTK direction) + circuit decode
  D4  lazy vs rich: NTK CKA across training checkpoints
  D6  decode the message: linear-probe node features to physical local quantities (meaningful N>=6)

Generic in N and omega; saves every plot's arrays (npz + CSVs) + figures + REPORT.md.

Usage:
  python scripts/run_depth_analysis.py --N 2 --omega 1.0 --arch ctnn_vcycle
  python scripts/run_depth_analysis.py --N 6 --omega 1.0 --arch ctnn_vcycle
  python scripts/run_depth_analysis.py --N 6 --omega 1.0 --arch deepset     # FFNN baseline
  python scripts/run_depth_analysis.py --N 2 --omega 1.0 --no-cusp          # D2 ablation arm
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from datetime import date
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import config  # noqa: E402
from analysis import diagnostics as dg  # noqa: E402
from analysis import representation as rp  # noqa: E402
from analysis.reference import TwoElectronExact  # noqa: E402
from analysis.system import System  # noqa: E402
from analysis.train import train_vmc_adam  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ARCH_KWARGS = {
    "ctnn_vcycle": dict(node_hidden=16, edge_hidden=16, bottleneck_hidden=8, n_down=1, n_up=1,
                        msg_layers=1, node_layers=1, readout_hidden=32, readout_layers=2, act="silu"),
    "deepset": dict(hidden=32, layers=2, readout_hidden=32, readout_layers=2, act="silu"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=2)
    ap.add_argument("--omega", type=float, default=1.0)
    ap.add_argument("--arch", type=str, default="ctnn_vcycle", choices=list(ARCH_KWARGS))
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--eval-samples", type=int, default=512, help="samples for per-ckpt NTK (CKA)")
    ap.add_argument("--align-samples", type=int, default=2048, help="samples for final O / eigenmodes")
    ap.add_argument("--no-cusp", action="store_true", help="disable analytic cusp prior (D2 arm)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="")
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    wtag = f"{a.omega:.4f}".rstrip("0").rstrip(".").replace(".", "p")
    cusptag = "_nocusp" if a.no_cusp else ""
    out = Path(a.outdir) if a.outdir else Path(
        f"results/analysis/{date.today().isoformat()}_depth_N{a.N}_w{wtag}_{a.arch}{cusptag}")
    out.mkdir(parents=True, exist_ok=True)

    akw = dict(ARCH_KWARGS[a.arch])
    if a.arch == "ctnn_vcycle":
        akw["use_analytic_cusp"] = not a.no_cusp
    sysm = System(N=a.N, omega=a.omega, d=2, arch=a.arch, arch_kwargs=akw, seed=a.seed)
    ref = config.get().E if np.isfinite(config.get().E) else None
    exact = TwoElectronExact(omega=a.omega) if a.N == 2 else None
    print(f"[depth] N={a.N} omega={a.omega} arch={a.arch} cusp={not a.no_cusp} "
          f"params={sysm.n_params():,} dev={sysm.device} -> {out}")

    # ---- train (single optimizer) with mid-flight checkpoints for D4/D5 ----
    ckpts = []  # list of (step, cpu state_dict)

    def snap(t):
        ckpts.append((t, {k: v.detach().cpu().clone() for k, v in sysm.f_net.state_dict().items()}))

    sysm.train()
    train_vmc_adam(sysm, steps=a.steps, lr=a.lr, batch=a.batch, lap_mode="exact",
                   log_every=max(1, a.steps // 6), ckpt_every=a.ckpt_every, ckpt_fn=snap)
    sysm.eval()

    # ---- fixed probe sets from the final |Psi|^2 ----
    x_align = sysm.sample(a.align_samples, steps=400, burn_in=800)
    E_align = dg.local_energy(sysm.log_psi, x_align, a.omega, sysm.params, lap_mode="exact")
    fin = torch.isfinite(E_align)
    x_align, E_align = x_align[fin], E_align[fin]
    x_eval = x_align[: a.eval_samples]
    pd_align = rp.pair_scalar(x_align, mode="min")
    pd_eval = rp.pair_scalar(x_eval, mode="min")

    q = dg.gs_quality(E_align, ref_energy=ref)
    err_pct = q.get("error_pct")
    print(f"[depth] trained E={q['E_mean_raw']:.6f}"
          + ("" if err_pct is None else f" ({err_pct:+.3f}% clip)"))

    # ---- D1: final O, NTK eigenmodes, update fields ----
    O_final = dg.build_O(sysm.log_psi, x_align[: a.align_samples], sysm.modules(), center=True)
    eig = rp.ntk_eigenmodes(O_final, pd_align, n_top=3, n_bottom=3)
    upd = rp.update_fields(O_final, E_align[: O_final.shape[0]], pd_align)

    # ---- D3: effective coordinate + circuit decode ----
    effc = rp.effective_coordinate(O_final, sysm, x_align[: a.align_samples], pd_align)
    hid = rp.decode_hidden(sysm, x_eval, pd_eval)

    # ---- D2: cusp decomposition (CTNN with cusp only) ----
    cusp_dec = None
    if a.arch == "ctnn_vcycle" and not a.no_cusp:
        with torch.no_grad():
            J_tot = (sysm.log_psi(x_align) - sysm.log_slater(x_align)).cpu().double().numpy()
            u_cusp = sysm.f_net._compute_cusps(x_align, sysm.spin).squeeze(-1).cpu().double().numpy()
        f_nn = J_tot - u_cusp
        rc, Jc = rp._bin(pd_align, J_tot, 30)
        _, uc = rp._bin(pd_align, u_cusp, 30)
        _, fc = rp._bin(pd_align, f_nn, 30)
        cusp_dec = {"r": rc, "J_total": Jc, "u_cusp": uc, "f_net": fc}

    # ---- D4: lazy vs rich (NTK CKA across checkpoints, fixed probe set) ----
    final_state = {k: v.detach().cpu().clone() for k, v in sysm.f_net.state_dict().items()}
    O_list, steps_ck = [], []
    for (t, st) in ckpts:
        sysm.f_net.load_state_dict({k: v.to(sysm.device) for k, v in st.items()})
        O_list.append(dg.build_O(sysm.log_psi, x_eval, sysm.modules(), center=True))
        steps_ck.append(t)
    sysm.f_net.load_state_dict({k: v.to(sysm.device) for k, v in final_state.items()})  # restore
    cka = rp.kernel_cka(O_list) if len(O_list) >= 2 else np.array([[1.0]])
    cka_vs_final = cka[-1] if cka.ndim == 2 else np.array([1.0])

    # ---- D6: decode message (physical local quantities) ----
    msg = rp.decode_message(sysm, x_align[: a.eval_samples])

    # ---- exact-truth overlap (N=2) ----
    overlap = None
    if exact is not None:
        overlap = dg.overlap_with_exact(sysm.log_psi, exact.log_psi, x_align)["overlap_sq"]

    summary = {
        "N": a.N, "omega": a.omega, "arch": a.arch, "cusp": (not a.no_cusp),
        "n_params": sysm.n_params(), "energy_raw": q["E_mean_raw"], "var_EL": q["var_EL"],
        "error_pct": q.get("error_pct"), "overlap_sq": overlap,
        "ntk_numerical_rank": eig["numerical_rank"],
        "hidden_eff_rank": {k: v["eff_rank"] for k, v in hid.items()},
        "message_r2": (msg.get("r2") if msg.get("available") else None),
        "cka_final_vs_first": float(cka_vs_final[0]) if cka_vs_final.size else None,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")

    np.savez(
        out / "depth_data.npz",
        eig_mu=eig["mu_desc"], eig_r=eig["r_centers"],
        eig_top=eig["top_modes"], eig_bottom=eig["bottom_modes"],
        upd_r=upd["r_centers"], upd_sr=upd["delta_sr"], upd_plain=upd["delta_plain"],
        upd_residual=upd["residual"],
        effc_r=effc["r_centers"], effc_dlogpsi=effc["delta_logpsi"],
        effc_singular=effc["singular_values"],
        cka_matrix=cka, cka_steps=np.array(steps_ck), cka_vs_final=cka_vs_final,
        **({f"cusp_{k}": v for k, v in cusp_dec.items()} if cusp_dec else {}),
    )
    _save_csvs(out, eig, upd, effc, cka, steps_ck, cka_vs_final, cusp_dec, hid)
    _figures(out, a, eig, upd, effc, cka_vs_final, steps_ck, cusp_dec, hid, msg, exact, pd_align)
    _report(out, a, summary, msg)
    print("[depth] summary:\n" + json.dumps(summary, indent=2, default=float))


def _save_csvs(out, eig, upd, effc, cka, steps_ck, cka_vs_final, cusp_dec, hid):
    def w(name, cols, arrs):
        n = min(len(x) for x in arrs)
        rows = [",".join(cols)] + [",".join(f"{arrs[c][i]:.8g}" for c in range(len(cols)))
                                   for i in range(n)]
        (out / name).write_text("\n".join(rows) + "\n")

    w("data_ntk_eigenmodes.csv",
      ["r"] + [f"top{i}" for i in range(len(eig["top_modes"]))]
      + [f"bot{i}" for i in range(len(eig["bottom_modes"]))],
      [eig["r_centers"], *list(eig["top_modes"]), *list(eig["bottom_modes"])])
    w("data_update_fields.csv", ["r", "delta_sr", "delta_plain", "residual"],
      [upd["r_centers"], upd["delta_sr"], upd["delta_plain"], upd["residual"]])
    w("data_effective_coordinate.csv", ["r", "delta_logpsi"], [effc["r_centers"], effc["delta_logpsi"]])
    w("data_lazy_rich.csv", ["step", "cka_vs_final"], [np.array(steps_ck), cka_vs_final])
    if cusp_dec:
        w("data_cusp_decomposition.csv", ["r", "J_total", "u_cusp", "f_net"],
          [cusp_dec["r"], cusp_dec["J_total"], cusp_dec["u_cusp"], cusp_dec["f_net"]])


def _figures(out, a, eig, upd, effc, cka_vs_final, steps_ck, cusp_dec, hid, msg, exact, pd):
    # D1a eigenmodes
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, m in enumerate(eig["top_modes"]):
        ax.plot(eig["r_centers"], m, "-", label=f"top {i} (mu={eig['top_mu'][i]:.1e})")
    for i, m in enumerate(eig["bottom_modes"]):
        ax.plot(eig["r_centers"], m, "--", alpha=0.6, label=f"stiff {i}")
    ax.set_xlabel("pair distance"); ax.set_ylabel("NTK eigenvector value")
    ax.set_title("NTK eigenmodes in real space (soft vs stiff)"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "fig_ntk_eigenmodes.png", dpi=140); plt.close(fig)

    # D1b update fields
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(upd["r_centers"], upd["delta_sr"], "o-", label="delta_psi SR")
    ax.plot(upd["r_centers"], upd["delta_plain"], "s-", label="delta_psi plain")
    ax.plot(upd["r_centers"], upd["residual"], "k:", alpha=0.6, label="residual (imag-time)")
    ax.set_xlabel("pair distance"); ax.set_ylabel("normalised change")
    ax.set_title("Where the update lands (SR vs plain)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig_update_fields.png", dpi=140); plt.close(fig)

    # D3 effective coordinate
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    axs[0].plot(effc["r_centers"], effc["delta_logpsi"], "o-")
    axs[0].set_xlabel("pair distance"); axs[0].set_ylabel("delta log|Psi|")
    axs[0].set_title("Effective coordinate (perturb dominant NTK dir)")
    s = effc["singular_values"]; s = s[s > 0]
    axs[1].semilogy(np.arange(1, s.size + 1), s, ".")
    axs[1].set_xlabel("index"); axs[1].set_ylabel("singular value of O")
    axs[1].set_title("O spectrum")
    fig.tight_layout(); fig.savefig(out / "fig_effective_coordinate.png", dpi=140); plt.close(fig)

    # D4 lazy vs rich
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps_ck, cka_vs_final, "o-")
    ax.set_xlabel("training step"); ax.set_ylabel("CKA(NTK_t, NTK_final)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Lazy vs rich (NTK drift): ~1 = lazy, falling = feature learning")
    fig.tight_layout(); fig.savefig(out / "fig_lazy_rich.png", dpi=140); plt.close(fig)

    # D2 cusp decomposition
    if cusp_dec:
        r0i = len(cusp_dec["r"]) // 2
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(cusp_dec["r"], cusp_dec["J_total"], "o-", label="J total")
        ax.plot(cusp_dec["r"], cusp_dec["u_cusp"], "--", label="u_cusp (fixed prior)")
        ax.plot(cusp_dec["r"], cusp_dec["f_net"], "-.", label="f_net (learned)")
        if exact is not None:
            Je = exact.jastrow_log(cusp_dec["r"])
            ax.plot(cusp_dec["r"], Je, "k:", alpha=0.6, label="exact J")
        ax.set_xlabel("pair distance"); ax.set_ylabel("Jastrow contribution")
        ax.set_title("Cusp prior vs learned correction"); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out / "fig_cusp_decomposition.png", dpi=140); plt.close(fig)

    # D3 circuit decode (eff rank bars)
    if hid:
        fig, ax = plt.subplots(figsize=(6, 4))
        names = list(hid.keys()); er = [hid[n]["eff_rank"] for n in names]
        nc = [hid[n]["n_channels"] for n in names]
        xb = np.arange(len(names))
        ax.bar(xb, er)
        for i, n in enumerate(nc):
            ax.text(xb[i], er[i], f"/{n}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(xb); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("effective rank"); ax.set_title("Hidden-layer effective rank (/n_channels)")
        fig.tight_layout(); fig.savefig(out / "fig_hidden_eff_rank.png", dpi=140); plt.close(fig)

    # D6 message decode
    if msg.get("available"):
        fig, ax = plt.subplots(figsize=(6, 4))
        keys = list(msg["r2"].keys()); vals = [msg["r2"][k] for k in keys]
        ax.bar(np.arange(len(keys)), vals)
        ax.set_xticks(np.arange(len(keys))); ax.set_xticklabels(keys, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("probe R^2"); ax.set_ylim(0, 1.05)
        ax.set_title(f"Message decode: node features -> physical local quantities ({a.arch})")
        fig.tight_layout(); fig.savefig(out / "fig_message_decode.png", dpi=140); plt.close(fig)


def _report(out, a, s, msg):
    L = [f"# Depth analysis: N={a.N}, omega={a.omega}, arch={a.arch}, cusp={s['cusp']}", "",
         f"- params: {s['n_params']:,}; E={s['energy_raw']:.6f} Ha"
         + ("" if s["error_pct"] is None else f" ({s['error_pct']:+.3f}%)")
         + ("" if s["overlap_sq"] is None else f"; overlap^2={s['overlap_sq']:.6f}"),
         f"- NTK numerical rank: {s['ntk_numerical_rank']}",
         f"- lazy-vs-rich CKA(first,final): {s['cka_final_vs_first']}",
         f"- hidden eff_rank: {s['hidden_eff_rank']}"]
    if s["message_r2"]:
        L.append(f"- message decode R^2: " + ", ".join(f"{k}={v:.2f}" for k, v in s["message_r2"].items()))
    L += ["", "Figures: fig_ntk_eigenmodes, fig_update_fields, fig_effective_coordinate, "
          "fig_lazy_rich, fig_cusp_decomposition, fig_hidden_eff_rank, fig_message_decode.",
          "Data: depth_data.npz + data_*.csv + summary.json."]
    (out / "REPORT.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
