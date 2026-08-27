"""Re-render the tangent-kernel chapter figures using the thesis mplstyle.

Reads the saved diagnostic data (no models are run) and re-plots so the figures
match the rest of the thesis (serif/STIX fonts, cream axes, dashed grid, thesis
colour cycle). Outputs overwrite the PNGs referenced by results_kernel.tex.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STYLE = ROOT / "src" / "Thesis_style.mplstyle"
ANALYSIS = ROOT / "results" / "analysis"
OUT = ROOT / "results" / "figures" / "results" / "kernel"

plt.style.use(str(STYLE))
# The style targets very large canvases; scale the type down for these compact,
# multi-panel diagnostic figures while keeping the thesis look (fonts, colours, grid).
plt.rcParams.update({
    "axes.titlesize": 17,
    "axes.labelsize": 15,
    "figure.titlesize": 18,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "lines.linewidth": 2.2,
    "lines.markersize": 6,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
})

CTNN_BLUE = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
DEEPSET_RED = plt.rcParams["axes.prop_cycle"].by_key()["color"][1]


def fair_dimension() -> None:
    d = ANALYSIS / "2026-06-22_fair_dimension_N6w1"
    conv = pd.read_csv(d / "convergence.csv")
    conv = conv[conv["measure"] == "pooled"]
    fair = pd.read_csv(d / "fair_table.csv")

    fig, (axc, axb) = plt.subplots(1, 2, figsize=(15, 5.6))
    for model, g in conv.groupby("model"):
        fam = g["family"].iloc[0]
        c = CTNN_BLUE if fam == "CTNN" else DEEPSET_RED
        g = g.sort_values("n_samples")
        axc.plot(g["n_samples"], g["eff_rank"], marker="o", color=c, label=model)
    axc.set_xlabel("probe samples")
    axc.set_ylabel(r"$d_{\mathrm{eff}}(S)$")
    axc.set_title(r"Sample-convergence of $d_{\mathrm{eff}}$ ($N{=}6,\ \omega{=}1$)")
    axc.legend(fontsize=9, ncol=2)

    fair = fair.sort_values(["family", "n_params"])
    colors = [CTNN_BLUE if f == "CTNN" else DEEPSET_RED for f in fair["family"]]
    axb.bar(range(len(fair)), fair["eff_rank_pooled"], color=colors)
    axb.set_xticks(range(len(fair)))
    axb.set_xticklabels(fair["model"], rotation=40, ha="right", fontsize=9)
    axb.set_ylabel(r"$d_{\mathrm{eff}}(S)$ (pooled)")
    axb.set_title("CTNN (blue) vs DeepSet (red)")

    fig.tight_layout()
    fig.savefig(OUT / "fair_dimension.png", dpi=200)
    plt.close(fig)


def ess_collapse() -> None:
    rec = json.load(open(ANALYSIS / "2026-06-21_ess_collapse" / "ess.json"))
    rec = sorted(rec, key=lambda r: r["omega"])
    w = [r["omega"] for r in rec]
    f = [100 * r["ess_frac_mean"] for r in rec]
    e = [100 * r["ess_frac_std"] for r in rec]

    fig, ax = plt.subplots(figsize=(8, 5.6))
    ax.errorbar(w, f, yerr=e, marker="o", color=DEEPSET_RED, capsize=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel("ESS fraction (\\%)")
    ax.set_title(r"Collocation ESS collapse vs $\omega$ ($N{=}6$)")
    fig.tight_layout()
    fig.savefig(OUT / "ess_collapse.png", dpi=200)
    plt.close(fig)


def message_ablation() -> None:
    df = pd.read_csv(ANALYSIS / "2026-07-02_message_ablation" / "ablation.csv")
    # mean energy per (omega, arm) across checkpoints, then ratio to the full model
    g = df.groupby(["omega", "arm"])["E"].mean().unstack("arm")
    ratio = g.div(g["msg+bf"], axis=0).sort_index()
    labels = {
        "msg+bf": ("full (MP+BF)", "black"),
        "nomsg+bf": ("no Jastrow message", DEEPSET_RED),
        "msg+nobf": ("no backflow", plt.rcParams["axes.prop_cycle"].by_key()["color"][5]),
        "nomsg+nobf": ("no message passing", CTNN_BLUE),
    }
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    for arm, (lab, c) in labels.items():
        if arm in ratio:
            ax.plot(ratio.index, ratio[arm], marker="o", color=c, label=lab)
    ax.set_xscale("log")
    ax.axhline(1.0, color="0.4", lw=1.0)
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$E / E_{\mathrm{full}}$")
    ax.set_title(r"Ablation of message passing and backflow ($N{=}6$)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "message_ablation.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fair_dimension()
    ess_collapse()
    message_ablation()
    print("re-rendered:", ", ".join(p.name for p in OUT.glob("*.png")
                                     if p.name in {"fair_dimension.png", "ess_collapse.png", "message_ablation.png"}))
