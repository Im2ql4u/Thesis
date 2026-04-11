#!/usr/bin/env python3
"""Cross-regime reliability analysis of all weak-form collocation results.

Parses all WEAK-FORM RESULT blocks from logs across outputs/ and results/,
extracts per-epoch ESS/k-hat statistics, and produces a regime × method
reliability table.
"""

import re
import os
import sys
from pathlib import Path
from collections import defaultdict
import statistics
from typing import List, Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent


def find_log_files() -> List[Path]:
    """Find all .log files in outputs/ and results/."""
    logs = []
    for d in ["outputs", "results"]:
        base = ROOT / d
        if base.exists():
            logs.extend(base.rglob("*.log"))
    return sorted(logs)


# Regex for the WEAK-FORM RESULT block (3 lines):
# Line 1:  WEAK-FORM RESULT  N=6  ω=0.5  E_DMC=11.78484
# Line 2:  Mode: bf   Tag: ablation_fixed_minsr_n6_w05
# Line 3:  E = 11.923988 ± 0.003240   err = +1.181%
RE_RESULT_HEADER = re.compile(
    r"WEAK-FORM RESULT\s+N=(\d+)\s+"
    r"[ωw]=([0-9.e+-]+)\s+"
    r"E_DMC=([0-9.enan+-]+)"
)
RE_TAG = re.compile(r"Tag:\s*(\S+)")
RE_ENERGY = re.compile(
    r"E\s*=\s*([0-9.e+-]+)\s*±\s*([0-9.e+-]+)\s+"
    r"err\s*=\s*([+\-0-9.]+)%"
)

# Epoch line: [  10] E=11.8172±0.423 var=1.79e-01 ESS=87 loss=1.039e+00 0.3s err=+0.27% khat=1.19 eta=6m
RE_EPOCH = re.compile(
    r"\[\s*(\d+)\]\s+E=([0-9.e+-]+)±([0-9.e+-]+).*?"
    r"ESS=([0-9.]+).*?"
    r"khat=([0-9.]+)"
)


def parse_log(path: Path) -> List[dict]:
    """Parse one log file, returning list of result records."""
    try:
        text = path.read_text(errors="replace")
    except Exception:
        return []

    lines = text.split("\n")
    results = []

    # 1. Find all WEAK-FORM RESULT blocks
    for i, line in enumerate(lines):
        m = RE_RESULT_HEADER.search(line)
        if not m:
            continue
        n = int(m.group(1))
        omega = float(m.group(2))
        e_dmc = m.group(3)
        e_dmc = float(e_dmc) if e_dmc != "nan" else float("nan")

        tag = ""
        energy = std = err = None
        # Look in next 4 lines for Tag and E
        for j in range(i + 1, min(i + 5, len(lines))):
            mt = RE_TAG.search(lines[j])
            if mt:
                tag = mt.group(1)
            me = RE_ENERGY.search(lines[j])
            if me:
                energy = float(me.group(1))
                std = float(me.group(2))
                err = float(me.group(3))

        results.append({
            "file": str(path.relative_to(ROOT)),
            "N": n,
            "omega": omega,
            "E_DMC": e_dmc,
            "tag": tag,
            "E": energy,
            "std": std,
            "err_pct": err,
        })

    # 2. Parse epoch-level ESS and k-hat for the entire log
    ess_vals = []
    khat_vals = []
    for line in lines:
        me = RE_EPOCH.search(line)
        if me:
            ess_vals.append(float(me.group(4)))
            khat_vals.append(float(me.group(5)))

    # Attach ESS/khat summary to all results from this file
    for r in results:
        if ess_vals:
            r["ess_median"] = statistics.median(ess_vals)
            r["ess_min"] = min(ess_vals)
            r["ess_max"] = max(ess_vals)
        else:
            r["ess_median"] = r["ess_min"] = r["ess_max"] = None
        if khat_vals:
            r["khat_median"] = statistics.median(khat_vals)
            r["khat_frac_gt07"] = sum(1 for k in khat_vals if k > 0.7) / len(khat_vals)
            r["khat_frac_gt1"] = sum(1 for k in khat_vals if k > 1.0) / len(khat_vals)
        else:
            r["khat_median"] = r["khat_frac_gt07"] = r["khat_frac_gt1"] = None

    # 3. Infer optimizer from tag or log content
    for r in results:
        t = r["tag"].lower()
        if "minsr" in t or "sr" in t:
            r["optimizer"] = "MinSR"
        elif "adam" in t:
            r["optimizer"] = "Adam"
        elif "cascade" in t:
            r["optimizer"] = "Cascade"
        else:
            # Check file content for --use-sr flag
            if "--use-sr" in text or "use_sr=True" in text:
                r["optimizer"] = "MinSR"
            else:
                r["optimizer"] = "Unknown"

        # Infer sampling
        if "adaptive" in t or "adapt" in t:
            r["sampling"] = "Adaptive"
        elif "fixed" in t:
            r["sampling"] = "Fixed"
        elif "hisamp" in t or "high_samp" in t:
            r["sampling"] = "HighSamp"
        elif "langevin" in t:
            r["sampling"] = "Langevin"
        else:
            r["sampling"] = "Unknown"

    return results


def main():
    logs = find_log_files()
    print(f"Found {len(logs)} log files")

    all_results = []
    for log in logs:
        all_results.extend(parse_log(log))

    print(f"Parsed {len(all_results)} WEAK-FORM RESULT blocks\n")

    if not all_results:
        print("No results found!")
        return

    # Filter out nan/None energies
    valid = [r for r in all_results if r["E"] is not None and r["err_pct"] is not None]
    print(f"Valid results (non-None E and err): {len(valid)}\n")

    # ===== Table 1: Regime summary =====
    regimes = defaultdict(list)
    for r in valid:
        regimes[(r["N"], r["omega"])].append(r)

    print("=" * 90)
    print(f"{'Regime':<16} {'Count':>5} {'Best err%':>10} {'Worst err%':>11} "
          f"{'Mean err%':>10} {'Std err%':>10} {'Med khat':>9} {'khat>0.7':>9}")
    print("=" * 90)

    for key in sorted(regimes.keys()):
        rlist = regimes[key]
        errs = [abs(r["err_pct"]) for r in rlist]
        best = min(errs)
        worst = max(errs)
        mean_e = statistics.mean(errs)
        std_e = statistics.stdev(errs) if len(errs) > 1 else 0.0
        khats = [r["khat_median"] for r in rlist if r["khat_median"] is not None]
        med_khat = f"{statistics.median(khats):.2f}" if khats else "N/A"
        kfrac = [r["khat_frac_gt07"] for r in rlist if r["khat_frac_gt07"] is not None]
        kfrac_s = f"{statistics.mean(kfrac)*100:.0f}%" if kfrac else "N/A"
        n, w = key
        print(f"N={n:<3} ω={w:<8} {len(rlist):>5} {best:>10.3f} {worst:>11.3f} "
              f"{mean_e:>10.3f} {std_e:>10.3f} {med_khat:>9} {kfrac_s:>9}")
    print()

    # ===== Table 2: Per-regime optimizer breakdown =====
    print("=" * 100)
    print(f"{'Regime':<16} {'Optimizer':<10} {'Count':>5} {'Best err%':>10} "
          f"{'Mean |err|%':>12} {'Std |err|%':>11} {'Med ESS':>8} {'Med khat':>9}")
    print("=" * 100)

    for key in sorted(regimes.keys()):
        rlist = regimes[key]
        by_opt = defaultdict(list)
        for r in rlist:
            by_opt[r["optimizer"]].append(r)
        n, w = key
        for opt in sorted(by_opt.keys()):
            olist = by_opt[opt]
            errs = [abs(r["err_pct"]) for r in olist]
            best = min(errs)
            mean_e = statistics.mean(errs)
            std_e = statistics.stdev(errs) if len(errs) > 1 else 0.0
            ess_vals = [r["ess_median"] for r in olist if r["ess_median"] is not None]
            med_ess = f"{statistics.median(ess_vals):.0f}" if ess_vals else "N/A"
            khats = [r["khat_median"] for r in olist if r["khat_median"] is not None]
            med_khat = f"{statistics.median(khats):.2f}" if khats else "N/A"
            print(f"N={n:<3} ω={w:<8} {opt:<10} {len(olist):>5} {best:>10.3f} "
                  f"{mean_e:>12.3f} {std_e:>11.3f} {med_ess:>8} {med_khat:>9}")
    print()

    # ===== Table 3: Top 5 per regime =====
    print("=" * 120)
    print("TOP 5 RESULTS PER REGIME (by |err|)")
    print("=" * 120)
    for key in sorted(regimes.keys()):
        rlist = sorted(regimes[key], key=lambda r: abs(r["err_pct"]))
        n, w = key
        print(f"\n--- N={n}, ω={w} ({len(rlist)} total runs) ---")
        for i, r in enumerate(rlist[:5]):
            ess_s = f"ESS={r['ess_median']:.0f}" if r["ess_median"] else "ESS=?"
            khat_s = f"khat={r['khat_median']:.2f}" if r["khat_median"] else "khat=?"
            print(f"  {i+1}. err={r['err_pct']:+.3f}%  E={r['E']:.6f}±{r['std']:.6f}  "
                  f"{ess_s}  {khat_s}  opt={r['optimizer']}  samp={r['sampling']}  "
                  f"tag={r['tag']}")
    print()

    # ===== Table 4: Reliability analysis =====
    # For each regime: coefficient of variation of |err|, spread ratio
    print("=" * 90)
    print("RELIABILITY ANALYSIS: Coefficient of Variation of |err| per regime")
    print("(Lower CV = more reliable)")
    print("=" * 90)
    print(f"{'Regime':<16} {'Count':>5} {'Mean |err|%':>12} {'Std |err|%':>11} "
          f"{'CV':>8} {'Interquartile':>14}")
    print("-" * 90)
    for key in sorted(regimes.keys()):
        rlist = regimes[key]
        errs = sorted([abs(r["err_pct"]) for r in rlist])
        if len(errs) < 3:
            continue
        mean_e = statistics.mean(errs)
        std_e = statistics.stdev(errs)
        cv = std_e / mean_e if mean_e > 0 else float("inf")
        q1 = errs[len(errs) // 4]
        q3 = errs[3 * len(errs) // 4]
        n, w = key
        print(f"N={n:<3} ω={w:<8} {len(errs):>5} {mean_e:>12.3f} {std_e:>11.3f} "
              f"{cv:>8.2f} {q1:.3f}–{q3:.3f}")
    print()

    # ===== Table 5: ESS vs error correlation =====
    print("=" * 80)
    print("ESS / K-HAT vs ERROR CORRELATION (per regime)")
    print("=" * 80)
    for key in sorted(regimes.keys()):
        rlist = regimes[key]
        pairs_ess = [(abs(r["err_pct"]), r["ess_median"])
                     for r in rlist if r["ess_median"] is not None and r["err_pct"] is not None]
        pairs_khat = [(abs(r["err_pct"]), r["khat_median"])
                      for r in rlist if r["khat_median"] is not None and r["err_pct"] is not None]
        n, w = key
        if len(pairs_ess) >= 5:
            # Spearman-like: rank correlation (simple approach)
            err_ranks = _rank([p[0] for p in pairs_ess])
            ess_ranks = _rank([p[1] for p in pairs_ess])
            rho_ess = _pearson(err_ranks, ess_ranks)
            khat_ranks = _rank([p[0] for p in pairs_khat])
            khat_v_ranks = _rank([p[1] for p in pairs_khat])
            rho_khat = _pearson(khat_ranks, khat_v_ranks)
            print(f"N={n:<3} ω={w:<8}  n={len(pairs_ess):>3}  "
                  f"rank_corr(|err|, ESS)={rho_ess:+.3f}  "
                  f"rank_corr(|err|, khat)={rho_khat:+.3f}")
        else:
            print(f"N={n:<3} ω={w:<8}  n={len(pairs_ess):>3}  (too few for correlation)")
    print()

    # ===== ω difficulty ranking =====
    print("=" * 80)
    print("REGIME DIFFICULTY: mean |err|% sorted (lower = easier)")
    print("=" * 80)
    regime_stats = []
    for key in sorted(regimes.keys()):
        rlist = regimes[key]
        errs = [abs(r["err_pct"]) for r in rlist]
        regime_stats.append((key, statistics.mean(errs), len(errs)))
    for (n, w), mean_e, cnt in sorted(regime_stats, key=lambda x: x[1]):
        print(f"  N={n:<3} ω={w:<8}  mean|err|={mean_e:.3f}%  (n={cnt})")

    print("\n\nDone.")


def _rank(vals: List[float]) -> List[float]:
    """Compute ranks (1-based, average ties)."""
    indexed = sorted(enumerate(vals), key=lambda x: x[1])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 1) / 2  # 1-based average
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def _pearson(x: List[float], y: List[float]) -> float:
    """Pearson correlation coefficient."""
    n = len(x)
    if n < 2:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    dx = (sum((xi - mx) ** 2 for xi in x)) ** 0.5
    dy = (sum((yi - my) ** 2 for yi in y)) ** 0.5
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


if __name__ == "__main__":
    main()
