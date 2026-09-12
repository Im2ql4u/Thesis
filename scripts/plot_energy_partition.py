"""Energy-partition Wigner diagnostics against trap frequency (thesis Fig. 'wigner_ratios').

Re-renders the two panels T/V_int and 2V_trap/V_int versus omega for N = 2, 6, 12 from
``src/energy_summary_out/energy_summary_fixed.csv`` (the |Psi|^2 expectation values of the
production states), in the thesis mplstyle. The figure size is chosen so that the style's
own font sizes and line widths print legibly at the half-text-width the thesis uses; no
colour, font or line setting is overridden.

Run: python scripts/plot_energy_partition.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STYLE = ROOT / "src" / "Thesis_style.mplstyle"
DATA = ROOT / "src" / "energy_summary_out" / "energy_summary_fixed.csv"
OUT = ROOT / "results" / "figures" / "results"

PARTICLE_NUMBERS = (2, 6, 12)  # those discussed in the text and caption
# Printed at 0.48 textwidth (~3.0 in): at this size the style's 34 pt labels and 24 pt ticks
# print at roughly 11 pt and 8 pt, and its 3.4 pt lines at ~1.1 pt.
FIGSIZE = (9.0, 6.6)
T_OVER_VINT_GUIDE = 0.1  # interaction-dominated guide line used in the caption
CLASSICAL_VIRIAL = 1.0  # 2 V_trap = V_int for a classical Coulomb cluster in a harmonic trap


def load_rows() -> dict[int, list[dict[str, float]]]:
    by_n: dict[int, list[dict[str, float]]] = {}
    with DATA.open() as f:
        for r in csv.DictReader(f):
            n = int(r["N"])
            if n in PARTICLE_NUMBERS:
                by_n.setdefault(n, []).append(
                    {k: float(v) for k, v in r.items() if v not in ("", "-")}
                )
    for n in by_n:
        by_n[n].sort(key=lambda r: r["omega"])
    return by_n


def panel(
    by_n: dict[int, list[dict[str, float]]], quantity: str, ylabel: str, guide: float, out_name: str
) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for n, rows in by_n.items():
        w = [r["omega"] for r in rows]
        if quantity == "T_over_Vint":
            y = [r["T_mean"] / r["V_int_mean"] for r in rows]
        else:
            y = [2.0 * r["V_trap_mean"] / r["V_int_mean"] for r in rows]
        ax.plot(w, y, marker="o", label=rf"$N={n}$")
    ax.axhline(guide, color="0.35", linestyle="--", linewidth=1.6)
    ax.set_xscale("log")
    ax.invert_xaxis()  # trap weakens from left to right, as in the text
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / f"{out_name}.pdf")
    plt.close(fig)


def main() -> None:
    plt.style.use(str(STYLE))
    by_n = load_rows()
    panel(
        by_n,
        "T_over_Vint",
        r"$T/V_{\mathrm{int}}$",
        T_OVER_VINT_GUIDE,
        "ratio_T_over_Vint_desc_allN",
    )
    panel(
        by_n,
        "2Vtrap_over_Vint",
        r"$2V_{\mathrm{trap}}/V_{\mathrm{int}}$",
        CLASSICAL_VIRIAL,
        "ratio_2Vtrap_over_Vint_desc_allN",
    )
    print(
        "wrote", OUT / "ratio_T_over_Vint_desc_allN.pdf", "and ratio_2Vtrap_over_Vint_desc_allN.pdf"
    )


if __name__ == "__main__":
    main()
