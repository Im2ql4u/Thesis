#!/usr/bin/env python3
"""Extract best N=12 campaign energies from model_quality_inventory and print LaTeX rows.

Usage:
    python3 scripts/extract_n12_campaign_best.py

Reads: outputs/model_quality_inventory_2026-03-21.json
Prints: ready-to-paste LaTeX table rows for the Campaign (best) column.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INVENTORY = ROOT / "outputs" / "model_quality_inventory_2026-03-21.json"

TABLE_OMEGAS = [1.0, 0.5, 0.1]  # omegas currently in tab:collocation for N=12


def fmt_energy(energy: float, sigma: float) -> str:
    """Format E±σ as LaTeX parenthesis notation, e.g. 65.706(4)."""
    if sigma <= 0:
        return f"{energy:.3f}"
    digits = max(0, -math.floor(math.log10(sigma)))
    sig_rounded = round(sigma * 10**digits)
    return f"{energy:.{digits}f}({sig_rounded})"


def main() -> None:
    data = json.loads(INVENTORY.read_text())
    n12 = [e for e in data if e.get("n") == 12]

    by_omega: dict[float, list] = defaultdict(list)
    for entry in n12:
        by_omega[entry["omega"]].append(entry)

    print("# N=12 Campaign (best) values for tab:collocation")
    print("# Extracted from:", INVENTORY.name)
    print()

    first_row = True
    for omega in TABLE_OMEGAS:
        entries = by_omega.get(omega, [])
        if not entries:
            print(f"# WARNING: no entries found for N=12 omega={omega}")
            continue

        best = min(entries, key=lambda x: x["abs_err_pct"])
        e_str = fmt_energy(best["energy"], best["sigma"])
        err_pct = best["err_pct"]
        err_sign = "+" if err_pct >= 0 else ""
        n_runs = len(entries)

        n_col = "12" if first_row else "  "
        first_row = False

        print(
            f"  {n_col} & {omega:<5} & $<DMC>$  & $<Multi-stage>$  "
            f"& ${e_str}$  & ${err_sign}{err_pct:.3f}$ & BF  "
            f"% n_runs={n_runs}, tag={best['tag']}"
        )

    print()
    print("# Full table rows (replace the 3 N=12 lines in tab:collocation):")
    print("# (fill in DMC and Multi-stage values from existing table)")
    print()
    print("DMC references for N=12:")
    for omega in TABLE_OMEGAS:
        entries = by_omega.get(omega, [])
        if entries:
            print(f"  omega={omega}: e_dmc = {entries[0]['e_dmc']}")


if __name__ == "__main__":
    main()
