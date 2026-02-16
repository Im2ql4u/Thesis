#!/usr/bin/env python3
"""Quick test: just discover pairs and print them."""
import os, sys, json, re
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
MODEL_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "models 2")

def discover_pairs():
    pairs = []
    for Np in ["2p", "6p", "12p"]:
        np_dir = os.path.join(MODEL_ROOT, Np)
        if not os.path.isdir(np_dir):
            continue
        N = int(Np[:-1])
        for wdir in sorted(os.listdir(np_dir)):
            w_path = os.path.join(np_dir, wdir)
            if not os.path.isdir(w_path) or not wdir.startswith("w_"):
                continue
            metas = {}
            for mf in os.listdir(w_path):
                if not mf.endswith(".meta.json"):
                    continue
                name = mf[:-10]
                with open(os.path.join(w_path, mf)) as f:
                    metas[name] = json.load(f)
            bf_names = [n for n in metas if n.lower().startswith("backflow")]
            fn_names = [n for n in metas if n.lower().startswith("f_net")]
            for bf_name in bf_names:
                bf_meta = metas[bf_name]
                bf_ts = bf_meta.get("saved_at", "")[:16]
                bf_class = bf_meta.get("class", "")
                bf_omega = bf_meta.get("omega")
                for fn_name in fn_names:
                    fn_meta = metas[fn_name]
                    fn_ts = fn_meta.get("saved_at", "")[:16]
                    if bf_ts == fn_ts and bf_ts:
                        is_ctnn = "CTNN" in bf_class or "CTNN" in bf_name
                        pair_type = "ctnn" if is_ctnn else "bf"
                        pairs.append({
                            "N": N, "omega": bf_omega, "omega_dir": wdir,
                            "pair_type": pair_type,
                            "bf_name": bf_name, "fn_name": fn_name,
                            "bf_class": bf_class, "fn_class": fn_meta.get("class",""),
                            "saved_at": bf_ts,
                        })
    return pairs

pairs = discover_pairs()
print(f"Found {len(pairs)} total pairs\n")

by_config = defaultdict(list)
for p in pairs:
    key = (p["N"], p["omega"], p["pair_type"])
    by_config[key].append(p)

for key in sorted(by_config.keys()):
    N, omega, ptype = key
    cands = by_config[key]
    print(f"N={N:2d}, ω={omega:<8}, {ptype:4s}: {len(cands)} candidate(s)")
    for c in cands:
        print(f"    {c['bf_name']:<35} + {c['fn_name']:<35} [{c['saved_at']}] {c['bf_class']}")
