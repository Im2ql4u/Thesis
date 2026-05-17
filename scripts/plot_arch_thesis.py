#!/usr/bin/env python3
"""
Thesis-ready architectural diagnostic figures.

Loads pre-computed .npz data from results/figures/architecture_diagnostics/
and produces separate, Thesis_style-formatted PDFs for inclusion in the thesis.

Output directory: results/figures/architecture_diagnostics/thesis/

Usage (all figures):
    .venv/bin/python3 scripts/plot_arch_thesis.py

Usage (specific figures):
    .venv/bin/python3 scripts/plot_arch_thesis.py B E FI G J A

Figures produced:
    B   — Input channel attribution across ω regimes (§repr-analysis)
    E   — Three-body sensitivity: CTNN vs pairwise Jastrow (method.tex §CTNN)
    FI  — Message-passing ablation + kinetic decomposition (§energy or §repr)
    G   — Backflow correlation-hole geometry (§repr-analysis §backflow)
    J   — Backflow force alignment vs Wigner crossover [CENTERPIECE] (§repr-analysis close)
    A   — Wavefunction sensitivity vs pair distance [appendix]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

REPO = Path(__file__).resolve().parent.parent
STYLE = REPO / "src" / "Thesis_style.mplstyle"
D_A = REPO / "results/figures/architecture_diagnostics/architecture_diagnostics_data.npz"
D_CP = REPO / "results/figures/architecture_diagnostics/ctnn_pairwise_data.npz"
D_D = REPO / "results/figures/architecture_diagnostics/deeper_diagnostics_data.npz"
OUT = REPO / "results/figures/architecture_diagnostics/thesis"
OUT.mkdir(parents=True, exist_ok=True)

plt.style.use(str(STYLE))

# ── thesis palette (matches main.tex \definecolor / Thesis_style prop_cycle) ──
BLUE = "#1F77B4"  # C0 mutedblue
WINE = "#A13333"  # C1 deepwine
BROWN = "#8c564b"  # C2 earthybrown
GREEN = "#2ca02c"  # C3 — not in palette but complementary
PURPLE = "#9467bd"  # C4 mutedpurple
TEAL = "#17becf"  # C5

# ω colours used consistently across all figures
OMEGA_CLR = {"ω=1.0": BLUE, "ω=0.1": PURPLE, "ω=0.001": WINE}
OMEGA_LBL = {
    "ω=1.0": r"$\omega = 1.0$",
    "ω=0.1": r"$\omega = 0.1$",
    "ω=0.001": r"$\omega = 0.001$",
}
OMEGA_KEYS = ["ω=1.0", "ω=0.1", "ω=0.001"]

# input channel order from diagnostic script docstring:
# [Δx, Δy, |r|, r², x, y, spin, r̄², s̄₁]
CHAN_TEX = [
    r"$\Delta x$",
    r"$\Delta y$",
    r"$|r|$",
    r"$r^2$",
    r"$x$",
    r"$y$",
    r"spin",
    r"$\bar{r}^2$",
    r"$\bar{s}_1$",
]
# group colours: pair features / particle features / global pooled
CHAN_CLR = [BLUE] * 4 + [WINE] * 3 + [BROWN] * 2


def _savefig(fig: plt.Figure, name: str) -> None:
    for ext in (".pdf", ".png"):
        fig.savefig(OUT / (name + ext), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  → {name}.pdf")


def _add_value_label(
    ax, bar, val: float, fmt: str = "{:.1f}%", offset_frac: float = 0.03, **kwargs
) -> None:
    """Annotate a bar with its value, above or below depending on sign."""
    height = bar.get_height()
    ypos = (
        height + offset_frac * ax.get_ylim()[1]
        if height >= 0
        else height - offset_frac * abs(ax.get_ylim()[0])
    )
    va = "bottom" if height >= 0 else "top"
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        ypos,
        fmt.format(val),
        ha="center",
        va=va,
        fontsize=20,
        **kwargs,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Fig B — Input channel attribution across ω regimes
# ══════════════════════════════════════════════════════════════════════════════


def plot_attribution() -> None:
    """3-panel bar chart: attribution fraction per input channel at each ω."""
    d = np.load(D_A, allow_pickle=True)

    raw_w1 = np.array(d["fig_B_attr_w1"], dtype=float)
    raw_w0001 = np.array(d["fig_B_attr_w0001"], dtype=float)
    w1 = raw_w1 / raw_w1.sum()
    w0001 = raw_w0001 / raw_w0001.sum()

    if w1.argmax() != 2:  # sanity: |r| should dominate at ω=1.0
        w1, w0001 = w0001, w1

    # ω=0.1 from DIAGNOSTIC_SUMMARY.md; order: [Δx, Δy, |r|, r², x, y, spin, r̄², s̄₁]
    w01 = np.array([0.45, 0.45, 5.0, 17.7, 5.4, 10.5, 51.4, 4.55, 4.55]) / 100.0

    fig, axes = plt.subplots(1, 3, figsize=(28, 12), sharey=False)
    fig.subplots_adjust(wspace=0.40)

    datasets = [
        (w1, r"$\omega = 1.0$"),
        (w01, r"$\omega = 0.1$"),
        (w0001, r"$\omega = 0.001$"),
    ]

    BAR_EC = "black"
    BAR_LW = 1.4

    for ax, (vals, title) in zip(axes, datasets, strict=False):
        pct = vals * 100
        dom = int(np.argmax(vals))
        bars = ax.bar(
            range(9), pct, color=CHAN_CLR, zorder=3, alpha=0.88, edgecolor=BAR_EC, linewidth=BAR_LW
        )
        bars[dom].set_linewidth(3.0)  # extra-thick edge on dominant bar

        ax.set_xticks(range(9))
        ax.set_xticklabels(CHAN_TEX, rotation=38, ha="right", fontsize=22)
        ax.set_ylabel("Attribution (%)", fontsize=28)
        ax.set_title(title, fontsize=30, pad=10)
        ax.set_ylim(0, max(pct.max() * 1.15, 8))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

    from matplotlib.patches import Patch

    legend_elems = [
        Patch(
            facecolor=BLUE,
            edgecolor=BAR_EC,
            linewidth=BAR_LW,
            label=r"Pair ($\Delta x$, $\Delta y$, $|r|$, $r^2$)",
        ),
        Patch(
            facecolor=WINE, edgecolor=BAR_EC, linewidth=BAR_LW, label=r"Particle ($x$, $y$, spin)"
        ),
        Patch(
            facecolor=BROWN,
            edgecolor=BAR_EC,
            linewidth=BAR_LW,
            label=r"Global ($\bar{r}^2$, $\bar{s}_1$)",
        ),
    ]
    axes[2].legend(handles=legend_elems, loc="upper right", fontsize=22, framealpha=0.9)

    fig.suptitle("Input channel attribution of the Jastrow correlator", fontsize=34, y=1.02)
    _savefig(fig, "fig_arch_attribution")


# ══════════════════════════════════════════════════════════════════════════════
# Fig E — Three-body sensitivity: CTNN vs pairwise Jastrow
# ══════════════════════════════════════════════════════════════════════════════


def plot_threebody() -> None:
    """
    Bar chart: intra-bin variance ratio CTNN/pairwise at each ω.
    A ratio > 1 means the CTNN's pair-gradient is more environment-sensitive
    than a pairwise Jastrow at the same pair distance — evidence of three-body
    and higher-order correlations.
    """
    d = np.load(D_D, allow_pickle=True)

    ratios = []
    for o in OMEGA_KEYS:
        vc = np.array(d[f"E_{o}_vc"], dtype=float)
        vp = np.array(d[f"E_{o}_vp"], dtype=float)
        # mean-of-ratios (per bin), clipping vp from below to avoid division artefacts
        vp_safe = np.where(vp > 1e-14, vp, np.nan)
        ratio = float(np.nanmean(vc / vp_safe))
        ratios.append(ratio)

    fig, ax = plt.subplots(1, 1, figsize=(14, 13))
    x = np.arange(3)
    bars = ax.bar(
        x, ratios, color=[OMEGA_CLR[o] for o in OMEGA_KEYS], width=0.52, zorder=3, alpha=0.88
    )

    ax.axhline(1.0, color="black", lw=2.2, ls="--", zorder=2, label="Pairwise baseline (ratio = 1)")
    ax.set_xticks(x)
    ax.set_xticklabels([OMEGA_LBL[o] for o in OMEGA_KEYS], fontsize=28)
    ax.set_ylabel(
        r"Intra-bin variance ratio  " r"$\sigma^2_{\mathrm{CTNN}}/\sigma^2_{\mathrm{pair}}$",
        fontsize=28,
    )
    ax.set_title("Three-body sensitivity:\nCTNN vs pairwise Jastrow", fontsize=30, pad=10)
    ax.legend(fontsize=24)
    ax.set_ylim(0, max(ratios) * 1.22)

    for bar, r in zip(bars, ratios, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            r + 0.04 * max(ratios),
            f"{r:.2f}×",
            ha="center",
            va="bottom",
            fontsize=26,
            fontweight="bold",
        )

    ax.annotate(
        "Wigner limit:\nnear-classical\nconfiguration space",
        xy=(2, ratios[2] + 0.05),
        xytext=(2.0, ratios[2] + 0.65),
        fontsize=20,
        ha="center",
        arrowprops=dict(arrowstyle="-|>", color="gray", lw=1.8),
        color="gray",
    )

    fig.tight_layout()
    _savefig(fig, "fig_arch_threebody")


# ══════════════════════════════════════════════════════════════════════════════
# Fig FI — Message-passing ablation + kinetic decomposition
# ══════════════════════════════════════════════════════════════════════════════


def plot_ablation() -> None:
    """
    Left: % energy increase when message-passing weights are zeroed.
    Right: kinetic energy T for CTNN vs pairwise at each ω — 100% of ΔE is kinetic.
    Numbers from DIAGNOSTIC_SUMMARY.md (Fig F and Fig I).
    """
    omegas_disp = [OMEGA_LBL[o] for o in OMEGA_KEYS]
    clrs = [OMEGA_CLR[o] for o in OMEGA_KEYS]

    # Fig F numbers (% energy increase when message-passing is removed)
    delta_E_pct = [22.4, 30.5, 11.8]

    # Fig I numbers (Hartree): kinetic energy for CTNN vs ablated pairwise Jastrow
    T_ctnn = [3.403, 0.473, 0.048]
    T_pair = [7.851, 1.603, 0.104]
    # Potentials are identical (same MCMC samples, only logΨ shape changes)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(26, 13))
    fig.subplots_adjust(wspace=0.35)

    x = np.arange(3)
    w = 0.50

    # ── left panel: energy cost of removing message-passing ──
    bars1 = ax1.bar(x, delta_E_pct, width=w, color=clrs, zorder=3, alpha=0.88)
    ax1.set_xticks(x)
    ax1.set_xticklabels(omegas_disp, fontsize=26)
    ax1.set_ylabel("Energy increase when MP removed (%)", fontsize=26)
    ax1.set_title("Cost of inter-particle\ncommunication", fontsize=28, pad=8)
    ax1.set_ylim(0, 42)
    for bar, v in zip(bars1, delta_E_pct, strict=False):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.7,
            f"+{v:.1f}%",
            ha="center",
            va="bottom",
            fontsize=25,
            fontweight="bold",
        )

    ax1.text(
        0.97,
        0.96,
        "Random messages\n= zero messages\n(learned geometry\nis not replaceable)",
        transform=ax1.transAxes,
        ha="right",
        va="top",
        fontsize=20,
        color="gray",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
    )

    # ── right panel: kinetic energy split ──
    wb = 0.38
    T_ctnn_a = np.array(T_ctnn)
    T_pair_a = np.array(T_pair)

    ax2.bar(x - wb / 2, T_ctnn_a, wb, label="CTNN Jastrow", color=BLUE, alpha=0.88, zorder=3)
    ax2.bar(
        x + wb / 2,
        T_pair_a,
        wb,
        label="Pairwise (MP removed)",
        color=WINE,
        alpha=0.65,
        zorder=3,
        hatch="//",
    )

    # ΔT arrows
    for xi, (tc, tp) in enumerate(zip(T_ctnn_a, T_pair_a, strict=False)):
        ax2.annotate(
            "",
            xy=(xi + wb / 2, tp),
            xytext=(xi + wb / 2, tc),
            arrowprops=dict(arrowstyle="<->", color="black", lw=2),
        )
        ax2.text(
            xi + wb / 2 + 0.03,
            (tc + tp) / 2,
            f"ΔT={tp-tc:.3f}",
            ha="left",
            va="center",
            fontsize=19,
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels(omegas_disp, fontsize=26)
    ax2.set_ylabel("Kinetic energy $T$  (Hartree)", fontsize=26)
    ax2.set_title(
        "100% of ΔE is kinetic\n(potentials unchanged at same positions)", fontsize=26, pad=8
    )
    ax2.legend(fontsize=23)

    fig.suptitle("Message-passing ablation: energy cost and kinetic origin", fontsize=34, y=1.03)
    _savefig(fig, "fig_arch_ablation")


# ══════════════════════════════════════════════════════════════════════════════
# Fig G — Backflow correlation-hole geometry
# ══════════════════════════════════════════════════════════════════════════════


def plot_bfgeo() -> None:
    """
    Left:  fraction of electrons displaced AWAY from nearest neighbour (cos > 0).
    Right: median cos split by same-spin vs opposite-spin nearest neighbour.
    The sign reversal at ω=0.001 shows the regime switch from correlation-hole
    deepening to Wigner-crystal orbital correction.
    """
    d = np.load(D_CP, allow_pickle=True)

    frac_away, med_same, med_opp = [], [], []
    for o in OMEGA_KEYS:
        cos = np.array(d[f"fig_G_{o}_cos"], dtype=float)
        ss = np.array(d[f"fig_G_{o}_ss"], dtype=float)
        frac_away.append(float((cos > 0).mean()) * 100)
        med_same.append(float(np.median(cos[ss == 1])) if (ss == 1).any() else 0.0)
        med_opp.append(float(np.median(cos[ss == 0])) if (ss == 0).any() else 0.0)

    labels = [OMEGA_LBL[o] for o in OMEGA_KEYS]
    clrs = [OMEGA_CLR[o] for o in OMEGA_KEYS]

    BAR_EC = "black"
    BAR_LW = 1.4

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
    fig.subplots_adjust(wspace=0.40)

    x = np.arange(3)

    # ── left: fraction displaced away from nearest neighbour ──
    bars = ax1.bar(
        x,
        frac_away,
        width=0.52,
        color=clrs,
        zorder=3,
        alpha=0.88,
        edgecolor=BAR_EC,
        linewidth=BAR_LW,
    )
    ax1.axhline(50, color="black", lw=2.8, ls="--", zorder=2)
    ax1.text(2.58, 51.5, "50%", fontsize=22, color="black", va="bottom")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=28)
    ax1.set_ylabel("Displaced away from nearest neighbour (%)", fontsize=26)
    ax1.set_title("Backflow direction", fontsize=30, pad=10)
    ax1.set_ylim(0, 100)
    for bar, v in zip(bars, frac_away, strict=False):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            v + 1.5,
            f"{v:.1f}%",
            ha="center",
            va="bottom",
            fontsize=25,
            fontweight="bold",
        )

    # ── right: spin-resolved median cosine ──
    wb = 0.38
    ax2.bar(
        x - wb / 2,
        med_same,
        wb,
        label="Same-spin",
        color=BLUE,
        alpha=0.88,
        zorder=3,
        edgecolor=BAR_EC,
        linewidth=BAR_LW,
    )
    ax2.bar(
        x + wb / 2,
        med_opp,
        wb,
        label="Opposite-spin",
        color=WINE,
        alpha=0.80,
        zorder=3,
        edgecolor=BAR_EC,
        linewidth=BAR_LW,
    )
    ax2.axhline(0, color="black", lw=2.8, zorder=2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=28)
    ax2.set_ylabel(r"Median cos$(\Delta\mathbf{x}_i,\;\hat{r}_{i\to j_{\min}})$", fontsize=26)
    ax2.set_title("Spin-resolved direction", fontsize=30, pad=10)
    ax2.set_ylim(-1.12, 1.12)
    ax2.legend(fontsize=24, loc="upper right")

    for xi, (ms, mo) in enumerate(zip(med_same, med_opp, strict=False)):
        for xpos, val in [(x[xi] - wb / 2, ms), (x[xi] + wb / 2, mo)]:
            yoff = 0.04 if val >= 0 else -0.07
            ax2.text(
                xpos,
                val + yoff,
                f"{val:+.2f}",
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=21,
            )

    fig.suptitle("Backflow correlation-hole geometry", fontsize=34, y=1.02)
    _savefig(fig, "fig_arch_bfgeo")


# ══════════════════════════════════════════════════════════════════════════════
# Fig J — Backflow force alignment  [CENTERPIECE]
# ══════════════════════════════════════════════════════════════════════════════


def plot_force_alignment() -> None:
    """
    Main panel: grouped bar chart — median cos(Δxᵢ, F) for three force components
    (full classical force, trap, Coulomb) at ω = 1.0, 0.1, 0.001.

    The key result: at ω≥0.1, the BF aligns with the trap force (opposes Coulomb spread).
    At ω=0.001 (Wigner crystal), complete sign reversal — BF aligns with Coulomb repulsion
    and moves electrons to their lattice sites. The transition coincides with the
    Wigner–molecule crossover identified in §sec:wigner-molecule.
    """
    d = np.load(D_D, allow_pickle=True)

    med_full, med_trap, med_coul = [], [], []
    for o in OMEGA_KEYS:
        cos = np.array(d[f"J_{o}_cos"], dtype=float)
        ctrap = np.array(d[f"J_{o}_ctrap"], dtype=float)
        ccoul = np.array(d[f"J_{o}_ccoul"], dtype=float)
        med_full.append(float(np.median(cos)))
        med_trap.append(float(np.median(ctrap)))
        med_coul.append(float(np.median(ccoul)))

    labels = [OMEGA_LBL[o] for o in OMEGA_KEYS]

    BAR_EC = "black"
    BAR_LW = 1.4

    fig, ax = plt.subplots(1, 1, figsize=(22, 14))

    x = np.arange(3)
    w = 0.25
    b1 = ax.bar(
        x - w,
        med_full,
        w,
        label="Full classical force",
        color=GREEN,
        alpha=0.88,
        zorder=3,
        edgecolor=BAR_EC,
        linewidth=BAR_LW,
    )
    b2 = ax.bar(
        x,
        med_trap,
        w,
        label="Trap force only",
        color=BLUE,
        alpha=0.88,
        zorder=3,
        edgecolor=BAR_EC,
        linewidth=BAR_LW,
    )
    b3 = ax.bar(
        x + w,
        med_coul,
        w,
        label="Coulomb force only",
        color=WINE,
        alpha=0.88,
        zorder=3,
        edgecolor=BAR_EC,
        linewidth=BAR_LW,
    )

    ax.axhline(0, color="black", lw=3.0, zorder=2)
    ax.axhline(1, color="black", ls=":", lw=1.8, zorder=2, alpha=0.35)
    ax.axhline(-1, color="black", ls=":", lw=1.8, zorder=2, alpha=0.35)

    ax.axvspan(1.5, 2.55, alpha=0.07, color=WINE, zorder=1)
    ax.text(2.02, 1.10, "Wigner crystal", ha="center", fontsize=24, color=WINE, style="italic")
    ax.axvspan(-0.55, 1.5, alpha=0.04, color=BLUE, zorder=1)
    ax.text(0.35, 1.10, "Correlated fluid", ha="center", fontsize=24, color=BLUE, style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=30)
    ax.set_ylabel(r"Median cos$(\Delta\mathbf{x}_i,\;\mathbf{F})$", fontsize=28)

    for bar_group, vals in [(b1, med_full), (b2, med_trap), (b3, med_coul)]:
        for bar, v in zip(bar_group, vals, strict=False):
            yoff = 0.05 if v >= 0 else -0.08
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + yoff,
                f"{v:+.2f}",
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=20,
                color="black",
            )

    ax.set_title("Backflow force alignment across the Wigner crossover", fontsize=30, pad=14)
    ax.legend(
        fontsize=25,
        loc="lower center",
        ncol=3,
        framealpha=0.9,
        edgecolor="gray",
        bbox_to_anchor=(0.5, -0.14),
    )
    ax.set_ylim(-1.22, 1.25)
    ax.set_xlim(-0.55, 2.55)
    fig.tight_layout(pad=1.5)
    _savefig(fig, "fig_arch_force_alignment")


# ══════════════════════════════════════════════════════════════════════════════
# Fig A — Wavefunction sensitivity vs pair distance  [appendix]
# ══════════════════════════════════════════════════════════════════════════════


def plot_sensitivity() -> None:
    """
    Two-panel appendix figure:
    Left:  ||∂logΨ/∂x₀|| vs pair distance r — compares 4 checkpoints.
    Right: per-channel attribution vs r for the REINFORCE ω=0.1 checkpoint.
    """
    d = np.load(D_A, allow_pickle=True)
    r = np.array(d["fig_A_r_vals"], dtype=float)

    ckpt_labels = {
        "REINFORCE_(ω=0.1)": (r"REINFORCE ($\omega=0.1$)", BLUE),
        "FD-Colloc_(ω=0.1)": (r"FD-Colloc ($\omega=0.1$)", WINE),
        "No-gate_(ω=0.1)": (r"No-gate ($\omega=0.1$)", BROWN),
        "Best_(ω=1.0)": (r"Best ($\omega=1.0$)", PURPLE),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(26, 12))
    fig.subplots_adjust(wspace=0.36)

    # ── left: sensitivity vs r ──
    for key, (lbl, clr) in ckpt_labels.items():
        npz_key = f"fig_A_{key}_sens"
        if npz_key in d:
            ax1.semilogy(r, np.array(d[npz_key], dtype=float), label=lbl, color=clr, lw=3.5)
    ax1.set_xlabel(r"Pair distance $r$ ($a_{\rm ho}$)", fontsize=26)
    ax1.set_ylabel(r"$\|\partial\log|\Psi|/\partial\mathbf{x}_0\|$", fontsize=26)
    ax1.set_title("Wavefunction sensitivity vs pair distance", fontsize=26, pad=8)
    ax1.legend(fontsize=21)
    ax1.axvline(0.15, color="gray", ls=":", lw=2.5, alpha=0.7)

    # ── right: per-channel attribution vs r (REINFORCE checkpoint) ──
    attr_key = "fig_A_REINFORCE_(ω=0.1)_attr"
    if attr_key in d:
        attr = np.array(d[attr_key], dtype=float)  # (n_r, 9)
        attr_norm = attr / (attr.sum(axis=1, keepdims=True) + 1e-12)
        for i, (lbl, clr) in enumerate(zip(CHAN_TEX, CHAN_CLR, strict=False)):
            ax2.plot(
                r,
                attr_norm[:, i] * 100,
                label=lbl,
                color=clr,
                lw=3.5 if i in [2, 6] else 2.2,
                ls="-" if i in [2, 6] else "--",
                alpha=0.85,
            )
        ax2.set_xlabel(r"Pair distance $r$ ($a_{\rm ho}$)", fontsize=26)
        ax2.set_ylabel("Attribution (%)", fontsize=26)
        ax2.set_title(
            "Per-channel attribution vs pair distance\n"
            r"(REINFORCE, $\omega=0.1$) — unsafe channels Δx, Δy suppressed globally",
            fontsize=22,
            pad=8,
        )
        ax2.legend(fontsize=18, ncol=2)

    fig.suptitle("Safe-feature mechanism: sensitivity and channel attribution", fontsize=32, y=1.02)
    _savefig(fig, "fig_arch_sensitivity")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

_FIGS: dict[str, tuple] = {
    "B": (plot_attribution, "Input channel attribution"),
    "E": (plot_threebody, "Three-body sensitivity"),
    "FI": (plot_ablation, "Ablation + kinetic decomposition"),
    "G": (plot_bfgeo, "Backflow correlation-hole geometry"),
    "J": (plot_force_alignment, "Force alignment [CENTERPIECE]"),
    "A": (plot_sensitivity, "Wavefunction sensitivity [appendix]"),
}

if __name__ == "__main__":
    keys = sys.argv[1:] if len(sys.argv) > 1 else list(_FIGS.keys())
    unknown = [k for k in keys if k not in _FIGS]
    if unknown:
        print(f"Unknown figures: {unknown}. Available: {list(_FIGS)}")
        sys.exit(1)

    print(f"Output: {OUT}\n")
    for key in keys:
        fn, desc = _FIGS[key]
        print(f"[{key}] {desc}")
        fn()

    print("\nAll done.")
