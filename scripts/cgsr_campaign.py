#!/usr/bin/env python3
"""
Cascading CG-SR campaign: warm-start across omega and N.

Strategy: ω=1.0 is solved. Cascade warm starts downward in omega and
upward in N. Each wave uses the best checkpoint from the previous wave.

Wave 1 (GPUs 0-4, ~2h): Refine what we have
  GPU 0: N=6  ω=0.5 — refine from camp_n6w05_fix.pt (currently +0.166%)
  GPU 1: N=6  ω=0.5 — variant: higher oversample + lower LR
  GPU 2: N=6  ω=0.1 — warm-start from best ω=0.5 checkpoint (bf_ctnn_vcycle.pt)
  GPU 3: N=6  ω=0.1 — warm-start from sr_n6w01_v2.pt (best prior, +43%)
  GPU 4: N=12 ω=1.0 — resume from camp_n12w1_smoke.pt (+20.7%)

Wave 2 (GPUs 0-5, ~4h): Continue from Wave 1 bests
  Cascade: N=6 ω=0.5 best → seed ω=0.1
  Cascade: N=6 ω=1.0 best → seed N=12 ω=1.0
  + refine all

Wave 3 (GPUs 0-5, ~4h): Final polish
  Ultra-low LR, heavy eval, all targets

DMC references:
  N=6  ω=1.0: 20.15932    N=6  ω=0.5: 11.78484    N=6  ω=0.1: 3.55385
  N=12 ω=1.0: 65.70010    N=12 ω=0.5: 39.15960
  N=20 ω=1.0: 155.88220
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
RESULTS = ROOT / "results" / "arch_colloc"
OUTDIR = ROOT / "outputs" / f"{datetime.now():%Y-%m-%d_%H%M}_cascade_campaign"
LOGDIR = OUTDIR / "logs"

MODULE_CMD = "source /etc/profile.d/lmod.sh 2>/dev/null; module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null"

# Available checkpoints
CKPT = {
    "n6w1_best": str(RESULTS / "camp_n6w1_verify.pt"),        # +0.012%
    "n6w1_sr": str(RESULTS / "sr_cg_anneal_v1.pt"),            # +0.029%
    "n6w05_fix": str(RESULTS / "camp_n6w05_fix.pt"),           # +0.166%
    "n6w01_old": str(RESULTS / "sr_n6w01_v2.pt"),              # +43.7%
    "n12w1_smoke": str(RESULTS / "camp_n12w1_smoke.pt"),       # +20.7%
    "n12w05_smoke": str(RESULTS / "camp_n12w05_smoke.pt"),     # +36.9%
    "n20w1_smoke": str(RESULTS / "camp_n20w1_smoke.pt"),       # +33.9%
    "bf_vcycle": str(RESULTS / "bf_ctnn_vcycle.pt"),            # ω=1.0 warm start
}

DMC_REF = {
    ("6", "1.0"): 20.15932,
    ("6", "0.5"): 11.78484,
    ("6", "0.1"): 3.55385,
    ("12", "1.0"): 65.70010,
    ("12", "0.5"): 39.15960,
    ("20", "1.0"): 155.88220,
}

# Common CG-SR flags (the proven recipe)
def cgsr_flags(
    lr="5e-3", lr_jas="5e-4",
    damping="1e-3", damping_end="1e-4", anneal_ep="400",
    subsample="512", cg_iters="15",
    max_change="0.03", trust="0.3",
    momentum="0.95", grad_clip="0.5", clip_el="5.0",
):
    return [
        "--natural-grad", "--sr-mode", "cg",
        "--lr", lr, "--lr-jas", lr_jas,
        "--fisher-damping", damping,
        "--fisher-damping-end", damping_end,
        "--fisher-damping-anneal", anneal_ep,
        "--fisher-subsample", subsample,
        "--sr-cg-iters", cg_iters,
        "--sr-max-param-change", max_change,
        "--sr-trust-region", trust,
        "--nat-momentum", momentum,
        "--grad-clip", grad_clip, "--clip-el", clip_el,
        "--direct-weight", "0.0",
    ]


# ═══════════════════════════════════════════════════════════
# WAVE 1: Refine existing + warm-start harder targets (~2h)
# ═══════════════════════════════════════════════════════════
WAVE1_JOBS = [
    # GPU 0: N=6 ω=0.5 — refine from +0.166% checkpoint
    {
        "name": "w1_n6w05_refine",
        "gpu": "0",
        "tag": "w1_n6w05_refine",
        "cmd": [
            "--mode", "bf", "--n-elec", "6", "--omega", "0.5",
            "--epochs", "600",
            "--n-coll", "8192", "--oversample", "8", "--micro-batch", "512",
            *cgsr_flags(lr="3e-3", lr_jas="3e-4", damping="5e-4",
                        damping_end="5e-5", anneal_ep="300",
                        subsample="1024", cg_iters="20",
                        max_change="0.03", trust="0.3", clip_el="3.0"),
            "--vmc-every", "30", "--vmc-n", "15000",
            "--n-eval", "50000",
            "--seed", "42",
            "--resume", CKPT["n6w05_fix"],
        ],
    },
    # GPU 1: N=6 ω=0.5 — higher oversample variant from same checkpoint
    {
        "name": "w1_n6w05_hisamp",
        "gpu": "1",
        "tag": "w1_n6w05_hisamp",
        "cmd": [
            "--mode", "bf", "--n-elec", "6", "--omega", "0.5",
            "--epochs", "600",
            "--n-coll", "4096", "--oversample", "16", "--micro-batch", "512",
            *cgsr_flags(lr="5e-3", lr_jas="5e-4", damping="1e-3",
                        damping_end="5e-5", anneal_ep="300",
                        subsample="1024", cg_iters="20",
                        max_change="0.05", trust="0.5", clip_el="3.0"),
            "--vmc-every", "30", "--vmc-n", "15000",
            "--n-eval", "50000",
            "--seed", "123",
            "--resume", CKPT["n6w05_fix"],
        ],
    },
    # GPU 2: N=6 ω=0.1 — warm-start from ω=1.0 weights (transfer learning)
    # The BF/Jastrow architecture should transfer — only the length scale differs.
    {
        "name": "w1_n6w01_transfer",
        "gpu": "2",
        "tag": "w1_n6w01_transfer",
        "cmd": [
            "--mode", "bf", "--n-elec", "6", "--omega", "0.1",
            "--epochs", "800",
            "--n-coll", "4096", "--oversample", "16", "--micro-batch", "512",
            *cgsr_flags(lr="5e-3", lr_jas="5e-4", damping="5e-3",
                        damping_end="1e-4", anneal_ep="400",
                        subsample="512", cg_iters="15",
                        max_change="0.05", trust="0.5", clip_el="3.0"),
            "--vmc-every", "40", "--vmc-n", "10000",
            "--n-eval", "30000",
            "--seed", "42",
            "--init-jas", CKPT["n6w1_best"],
            "--init-bf", CKPT["n6w1_best"],
            "--no-pretrained",
        ],
    },
    # GPU 3: N=6 ω=0.1 — resume from best prior ω=0.1 checkpoint (+43.7%)
    # More oversample + tighter clip_el to reduce variance.
    {
        "name": "w1_n6w01_resume",
        "gpu": "3",
        "tag": "w1_n6w01_resume",
        "cmd": [
            "--mode", "bf", "--n-elec", "6", "--omega", "0.1",
            "--epochs", "800",
            "--n-coll", "4096", "--oversample", "20", "--micro-batch", "512",
            *cgsr_flags(lr="3e-3", lr_jas="3e-4", damping="1e-3",
                        damping_end="5e-5", anneal_ep="400",
                        subsample="512", cg_iters="15",
                        max_change="0.03", trust="0.3", clip_el="3.0"),
            "--vmc-every", "40", "--vmc-n", "10000",
            "--n-eval", "30000",
            "--seed", "42",
            "--resume", CKPT["n6w01_old"],
        ],
    },
    # GPU 4: N=12 ω=1.0 — resume from Phase 1 smoke (+20.7%)
    {
        "name": "w1_n12w1_continue",
        "gpu": "4",
        "tag": "w1_n12w1_cont",
        "cmd": [
            "--mode", "bf", "--n-elec", "12", "--omega", "1.0",
            "--epochs", "800",
            "--n-coll", "4096", "--oversample", "8", "--micro-batch", "512",
            *cgsr_flags(lr="5e-3", lr_jas="5e-4", damping="1e-3",
                        damping_end="1e-4", anneal_ep="400",
                        subsample="512", cg_iters="15",
                        max_change="0.05", trust="0.5"),
            "--vmc-every", "40", "--vmc-n", "10000",
            "--n-eval", "25000",
            "--seed", "42",
            "--resume", CKPT["n12w1_smoke"],
        ],
    },
    # GPU 5: N=12 ω=0.5 — resume from Phase 1 smoke (+36.9%)
    {
        "name": "w1_n12w05_continue",
        "gpu": "5",
        "tag": "w1_n12w05_cont",
        "cmd": [
            "--mode", "bf", "--n-elec", "12", "--omega", "0.5",
            "--epochs", "800",
            "--n-coll", "4096", "--oversample", "10", "--micro-batch", "512",
            *cgsr_flags(lr="5e-3", lr_jas="5e-4", damping="1e-3",
                        damping_end="1e-4", anneal_ep="400",
                        subsample="512", cg_iters="15",
                        max_change="0.05", trust="0.5"),
            "--vmc-every", "40", "--vmc-n", "10000",
            "--n-eval", "25000",
            "--seed", "42",
            "--resume", CKPT["n12w05_smoke"],
        ],
    },
    # GPU 6: N=12 ω=1.0 — from scratch with init from N=6 ω=1.0 (transfer)
    {
        "name": "w1_n12w1_transfer",
        "gpu": "6",
        "tag": "w1_n12w1_xfer",
        "cmd": [
            "--mode", "bf", "--n-elec", "12", "--omega", "1.0",
            "--epochs", "800",
            "--n-coll", "4096", "--oversample", "8", "--micro-batch", "512",
            *cgsr_flags(lr="5e-3", lr_jas="5e-4", damping="1e-3",
                        damping_end="1e-4", anneal_ep="400",
                        subsample="512", cg_iters="15",
                        max_change="0.05", trust="0.5"),
            "--vmc-every", "40", "--vmc-n", "10000",
            "--n-eval", "25000",
            "--seed", "123",
            "--init-jas", CKPT["n6w1_best"],
            "--init-bf", CKPT["n6w1_best"],
            "--no-pretrained",
        ],
    },
    # GPU 7: N=20 ω=1.0 — small arch, resume from Phase 1 smoke (+33.9%)
    {
        "name": "w1_n20w1_continue",
        "gpu": "7",
        "tag": "w1_n20w1_cont",
        "cmd": [
            "--mode", "bf", "--n-elec", "20", "--omega", "1.0",
            "--bf-hidden", "64", "--bf-layers", "2",
            "--epochs", "800",
            "--n-coll", "1024", "--oversample", "8", "--micro-batch", "128",
            *cgsr_flags(lr="3e-3", lr_jas="3e-4", damping="1e-3",
                        damping_end="1e-4", anneal_ep="400",
                        subsample="128", cg_iters="10",
                        max_change="0.05", trust="0.5"),
            "--vmc-every", "50", "--vmc-n", "4000",
            "--n-eval", "10000",
            "--seed", "42",
            "--resume", CKPT["n20w1_smoke"],
        ],
    },
]

# ═══════════════════════════════════════════════════════════
# WAVE 2: Cascade from Wave 1 best (~4h)
# ═══════════════════════════════════════════════════════════
WAVE2_JOBS = [
    # GPU 0: N=6 ω=0.5 — polish Wave 1 best
    {
        "name": "w2_n6w05_polish",
        "gpu": "0",
        "tag": "w2_n6w05_polish",
        "resume_tag": "w1_n6w05_refine",
        "alt_resume_tag": "w1_n6w05_hisamp",
        "cmd": [
            "--mode", "bf", "--n-elec", "6", "--omega", "0.5",
            "--epochs", "1000",
            "--n-coll", "8192", "--oversample", "10", "--micro-batch", "512",
            *cgsr_flags(lr="1e-3", lr_jas="1e-4", damping="1e-4",
                        damping_end="5e-5", anneal_ep="500",
                        subsample="2048", cg_iters="25",
                        max_change="0.02", trust="0.2", clip_el="3.0"),
            "--vmc-every", "25", "--vmc-n", "20000",
            "--n-eval", "60000",
            "--seed", "42",
        ],
    },
    # GPU 1: N=6 ω=0.1 — warm-start from best Wave 1 ω=0.5 (cascade)
    {
        "name": "w2_n6w01_cascade",
        "gpu": "1",
        "tag": "w2_n6w01_cascade",
        "resume_tag": "w1_n6w05_refine",  # cascade from ω=0.5 best
        "cmd": [
            "--mode", "bf", "--n-elec", "6", "--omega", "0.1",
            "--epochs", "1500",
            "--n-coll", "4096", "--oversample", "20", "--micro-batch", "512",
            *cgsr_flags(lr="3e-3", lr_jas="3e-4", damping="1e-3",
                        damping_end="5e-5", anneal_ep="600",
                        subsample="512", cg_iters="15",
                        max_change="0.03", trust="0.3", clip_el="3.0"),
            "--vmc-every", "40", "--vmc-n", "10000",
            "--n-eval", "30000",
            "--seed", "42",
            "--no-pretrained",
        ],
    },
    # GPU 2: N=6 ω=0.1 — continue best Wave 1 ω=0.1 run
    {
        "name": "w2_n6w01_continue",
        "gpu": "2",
        "tag": "w2_n6w01_cont",
        "resume_tag": "w1_n6w01_resume",
        "alt_resume_tag": "w1_n6w01_transfer",
        "cmd": [
            "--mode", "bf", "--n-elec", "6", "--omega", "0.1",
            "--epochs", "1500",
            "--n-coll", "4096", "--oversample", "20", "--micro-batch", "512",
            *cgsr_flags(lr="1e-3", lr_jas="1e-4", damping="5e-4",
                        damping_end="5e-5", anneal_ep="600",
                        subsample="512", cg_iters="20",
                        max_change="0.02", trust="0.2", clip_el="3.0"),
            "--vmc-every", "40", "--vmc-n", "10000",
            "--n-eval", "30000",
            "--seed", "42",
        ],
    },
    # GPU 3: N=12 ω=1.0 — continue best Wave 1
    {
        "name": "w2_n12w1_continue",
        "gpu": "3",
        "tag": "w2_n12w1_cont",
        "resume_tag": "w1_n12w1_cont",
        "alt_resume_tag": "w1_n12w1_xfer",
        "cmd": [
            "--mode", "bf", "--n-elec", "12", "--omega", "1.0",
            "--epochs", "1200",
            "--n-coll", "4096", "--oversample", "10", "--micro-batch", "512",
            *cgsr_flags(lr="3e-3", lr_jas="3e-4", damping="5e-4",
                        damping_end="5e-5", anneal_ep="500",
                        subsample="1024", cg_iters="20",
                        max_change="0.03", trust="0.3"),
            "--vmc-every", "40", "--vmc-n", "12000",
            "--n-eval", "30000",
            "--seed", "42",
        ],
    },
    # GPU 4: N=12 ω=0.5 — continue Wave 1
    {
        "name": "w2_n12w05_continue",
        "gpu": "4",
        "tag": "w2_n12w05_cont",
        "resume_tag": "w1_n12w05_cont",
        "cmd": [
            "--mode", "bf", "--n-elec", "12", "--omega", "0.5",
            "--epochs", "1200",
            "--n-coll", "4096", "--oversample", "10", "--micro-batch", "512",
            *cgsr_flags(lr="3e-3", lr_jas="3e-4", damping="5e-4",
                        damping_end="5e-5", anneal_ep="500",
                        subsample="512", cg_iters="15",
                        max_change="0.03", trust="0.3"),
            "--vmc-every", "40", "--vmc-n", "10000",
            "--n-eval", "25000",
            "--seed", "42",
        ],
    },
    # GPU 5: N=20 ω=1.0 — continue from Wave 1
    {
        "name": "w2_n20w1_continue",
        "gpu": "5",
        "tag": "w2_n20w1_cont",
        "resume_tag": "w1_n20w1_cont",
        "cmd": [
            "--mode", "bf", "--n-elec", "20", "--omega", "1.0",
            "--bf-hidden", "64", "--bf-layers", "2",
            "--epochs", "1000",
            "--n-coll", "1024", "--oversample", "8", "--micro-batch", "128",
            *cgsr_flags(lr="1e-3", lr_jas="1e-4", damping="5e-4",
                        damping_end="5e-5", anneal_ep="500",
                        subsample="128", cg_iters="10",
                        max_change="0.03", trust="0.3"),
            "--vmc-every", "50", "--vmc-n", "4000",
            "--n-eval", "10000",
            "--seed", "42",
        ],
    },
    # GPU 6: N=6 ω=0.1 — aggressive: many samples, strong clipping
    {
        "name": "w2_n6w01_aggressive",
        "gpu": "6",
        "tag": "w2_n6w01_aggr",
        "resume_tag": "w1_n6w01_resume",
        "cmd": [
            "--mode", "bf", "--n-elec", "6", "--omega", "0.1",
            "--epochs", "1500",
            "--n-coll", "8192", "--oversample", "16", "--micro-batch", "512",
            *cgsr_flags(lr="5e-3", lr_jas="5e-4", damping="1e-3",
                        damping_end="1e-4", anneal_ep="600",
                        subsample="1024", cg_iters="20",
                        max_change="0.05", trust="0.5", clip_el="2.0"),
            "--vmc-every", "40", "--vmc-n", "10000",
            "--n-eval", "30000",
            "--seed", "42",
        ],
    },
    # GPU 7: N=12 ω=1.0 — warm-start from N=6 ω=1.0 best, different seed
    {
        "name": "w2_n12w1_xfer2",
        "gpu": "7",
        "tag": "w2_n12w1_xfer2",
        "cmd": [
            "--mode", "bf", "--n-elec", "12", "--omega", "1.0",
            "--epochs", "1200",
            "--n-coll", "4096", "--oversample", "8", "--micro-batch", "512",
            *cgsr_flags(lr="5e-3", lr_jas="5e-4", damping="1e-3",
                        damping_end="1e-4", anneal_ep="500",
                        subsample="512", cg_iters="15",
                        max_change="0.05", trust="0.5"),
            "--vmc-every", "40", "--vmc-n", "10000",
            "--n-eval", "25000",
            "--seed", "77",
            "--init-jas", CKPT["n6w1_best"],
            "--init-bf", CKPT["n6w1_best"],
            "--no-pretrained",
        ],
    },
]

# ═══════════════════════════════════════════════════════════
# WAVE 3: Final polish from Wave 2 best (~4h)
# ═══════════════════════════════════════════════════════════
WAVE3_JOBS = [
    # GPU 0: N=6 ω=0.5 — ultra polish
    {
        "name": "w3_n6w05_ultra",
        "gpu": "0",
        "tag": "w3_n6w05_ultra",
        "resume_tag": "w2_n6w05_polish",
        "cmd": [
            "--mode", "bf", "--n-elec", "6", "--omega", "0.5",
            "--epochs", "800",
            "--n-coll", "8192", "--oversample", "10", "--micro-batch", "512",
            *cgsr_flags(lr="5e-4", lr_jas="5e-5", damping="5e-5",
                        subsample="2048", cg_iters="25",
                        max_change="0.01", trust="0.1", clip_el="3.0"),
            "--vmc-every", "25", "--vmc-n", "25000",
            "--n-eval", "80000",
            "--seed", "42",
        ],
    },
    # GPU 1: N=6 ω=0.1 — polish best cascade
    {
        "name": "w3_n6w01_polish",
        "gpu": "1",
        "tag": "w3_n6w01_polish",
        "resume_tag": "w2_n6w01_cascade",
        "alt_resume_tag": "w2_n6w01_cont",
        "cmd": [
            "--mode", "bf", "--n-elec", "6", "--omega", "0.1",
            "--epochs", "1000",
            "--n-coll", "8192", "--oversample", "16", "--micro-batch", "512",
            *cgsr_flags(lr="5e-4", lr_jas="5e-5", damping="1e-4",
                        damping_end="5e-5", anneal_ep="500",
                        subsample="1024", cg_iters="20",
                        max_change="0.01", trust="0.1", clip_el="3.0"),
            "--vmc-every", "30", "--vmc-n", "12000",
            "--n-eval", "40000",
            "--seed", "42",
        ],
    },
    # GPU 2: N=6 ω=0.1 — polish aggressive variant
    {
        "name": "w3_n6w01_polish_aggr",
        "gpu": "2",
        "tag": "w3_n6w01_pAggr",
        "resume_tag": "w2_n6w01_aggr",
        "cmd": [
            "--mode", "bf", "--n-elec", "6", "--omega", "0.1",
            "--epochs", "1000",
            "--n-coll", "8192", "--oversample", "16", "--micro-batch", "512",
            *cgsr_flags(lr="5e-4", lr_jas="5e-5", damping="1e-4",
                        damping_end="5e-5", anneal_ep="500",
                        subsample="1024", cg_iters="20",
                        max_change="0.01", trust="0.1", clip_el="3.0"),
            "--vmc-every", "30", "--vmc-n", "12000",
            "--n-eval", "40000",
            "--seed", "42",
        ],
    },
    # GPU 3: N=12 ω=1.0 — polish
    {
        "name": "w3_n12w1_polish",
        "gpu": "3",
        "tag": "w3_n12w1_polish",
        "resume_tag": "w2_n12w1_cont",
        "alt_resume_tag": "w2_n12w1_xfer2",
        "cmd": [
            "--mode", "bf", "--n-elec", "12", "--omega", "1.0",
            "--epochs", "800",
            "--n-coll", "4096", "--oversample", "10", "--micro-batch", "512",
            *cgsr_flags(lr="1e-3", lr_jas="1e-4", damping="1e-4",
                        subsample="1024", cg_iters="20",
                        max_change="0.02", trust="0.2"),
            "--vmc-every", "30", "--vmc-n", "12000",
            "--n-eval", "30000",
            "--seed", "42",
        ],
    },
    # GPU 4: N=12 ω=0.5 — polish
    {
        "name": "w3_n12w05_polish",
        "gpu": "4",
        "tag": "w3_n12w05_polish",
        "resume_tag": "w2_n12w05_cont",
        "cmd": [
            "--mode", "bf", "--n-elec", "12", "--omega", "0.5",
            "--epochs", "800",
            "--n-coll", "4096", "--oversample", "10", "--micro-batch", "512",
            *cgsr_flags(lr="1e-3", lr_jas="1e-4", damping="1e-4",
                        subsample="512", cg_iters="20",
                        max_change="0.02", trust="0.2"),
            "--vmc-every", "30", "--vmc-n", "10000",
            "--n-eval", "25000",
            "--seed", "42",
        ],
    },
    # GPU 5: N=20 ω=1.0 — polish
    {
        "name": "w3_n20w1_polish",
        "gpu": "5",
        "tag": "w3_n20w1_polish",
        "resume_tag": "w2_n20w1_cont",
        "cmd": [
            "--mode", "bf", "--n-elec", "20", "--omega", "1.0",
            "--bf-hidden", "64", "--bf-layers", "2",
            "--epochs", "600",
            "--n-coll", "1024", "--oversample", "8", "--micro-batch", "128",
            *cgsr_flags(lr="5e-4", lr_jas="5e-5", damping="5e-5",
                        subsample="128", cg_iters="10",
                        max_change="0.02", trust="0.2"),
            "--vmc-every", "50", "--vmc-n", "4000",
            "--n-eval", "10000",
            "--seed", "42",
        ],
    },
    # GPU 6: N=6 ω=1.0 — keep polishing our best result
    {
        "name": "w3_n6w1_ultra",
        "gpu": "6",
        "tag": "w3_n6w1_ultra",
        "cmd": [
            "--mode", "bf", "--n-elec", "6", "--omega", "1.0",
            "--epochs", "600",
            "--n-coll", "8192", "--oversample", "8", "--micro-batch", "512",
            *cgsr_flags(lr="5e-4", lr_jas="5e-5", damping="5e-5",
                        subsample="2048", cg_iters="25",
                        max_change="0.01", trust="0.1", clip_el="3.0"),
            "--vmc-every", "20", "--vmc-n", "25000",
            "--n-eval", "100000",
            "--seed", "42",
            "--resume", CKPT["n6w1_best"],
        ],
    },
    # GPU 7: N=12 ω=1.0 — extra variant
    {
        "name": "w3_n12w1_extra",
        "gpu": "7",
        "tag": "w3_n12w1_extra",
        "resume_tag": "w2_n12w1_xfer2",
        "cmd": [
            "--mode", "bf", "--n-elec", "12", "--omega", "1.0",
            "--epochs", "800",
            "--n-coll", "4096", "--oversample", "10", "--micro-batch", "512",
            *cgsr_flags(lr="1e-3", lr_jas="1e-4", damping="1e-4",
                        subsample="1024", cg_iters="20",
                        max_change="0.02", trust="0.2"),
            "--vmc-every", "30", "--vmc-n", "12000",
            "--n-eval", "30000",
            "--seed", "77",
        ],
    },
]


def extract_n_omega(cmd: list[str]) -> tuple[str, str]:
    n = omega = "?"
    for i, flag in enumerate(cmd):
        if flag == "--n-elec" and i + 1 < len(cmd):
            n = cmd[i + 1]
        if flag == "--omega" and i + 1 < len(cmd):
            omega = cmd[i + 1]
    return n, omega


def pick_best_resume(job: dict) -> str | None:
    """Pick best checkpoint from resume_tag or alt_resume_tag."""
    for key in ["resume_tag", "alt_resume_tag"]:
        tag = job.get(key)
        if tag:
            path = RESULTS / f"{tag}.pt"
            if path.exists():
                return str(path)
    return None


def launch_wave(wave_name: str, jobs: list[dict], wave_dir: Path) -> list[dict]:
    wave_log = wave_dir / "logs"
    wave_log.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  {wave_name}: {len(jobs)} jobs")
    print(f"  Output: {wave_dir}")
    print(f"{'='*60}\n")

    (wave_dir / "plan.json").write_text(json.dumps(jobs, indent=2, default=str))

    procs = []
    for job in jobs:
        tag = job["tag"]
        gpu = job["gpu"]
        logfile = wave_log / f"{tag}.log"
        cmd = list(job["cmd"])

        if "--tag" not in cmd:
            cmd.extend(["--tag", tag])

        # Resolve resume for Wave 2/3 jobs
        if "resume_tag" in job and "--resume" not in cmd:
            ckpt = pick_best_resume(job)
            if ckpt:
                cmd.extend(["--resume", ckpt])
                print(f"  [{job['name']}] GPU={gpu} resume={Path(ckpt).name}")
            else:
                print(f"  [{job['name']}] GPU={gpu} (no checkpoint found, from scratch)")
        else:
            # Show resume source from cmd
            resume_in_cmd = None
            for i, c in enumerate(cmd):
                if c == "--resume" and i + 1 < len(cmd):
                    resume_in_cmd = Path(cmd[i + 1]).name
            init_in_cmd = None
            for i, c in enumerate(cmd):
                if c == "--init-bf" and i + 1 < len(cmd):
                    init_in_cmd = Path(cmd[i + 1]).name
            if resume_in_cmd:
                print(f"  [{job['name']}] GPU={gpu} resume={resume_in_cmd}")
            elif init_in_cmd:
                print(f"  [{job['name']}] GPU={gpu} init={init_in_cmd}")
            else:
                print(f"  [{job['name']}] GPU={gpu} (from scratch)")

        full_cmd = (
            f"cd {SRC}; {MODULE_CMD}; "
            f"CUDA_MANUAL_DEVICE={gpu} python run_weak_form.py "
            + " ".join(cmd)
        )

        with open(logfile, "w") as lf:
            lf.write(f"# {job['name']} — GPU {gpu}\n")
            lf.write(f"# {full_cmd}\n\n")

        proc = subprocess.Popen(
            ["bash", "-c", f"{full_cmd} >> {logfile} 2>&1"],
            start_new_session=True,
        )
        procs.append((job, proc))
        time.sleep(3)

    print(f"\n  All {len(procs)} jobs launched. PIDs: {[p.pid for _, p in procs]}")
    print(f"  Monitor: tail -f {wave_log}/*.log")

    # Wait
    results = []
    for job, proc in procs:
        rc = proc.wait()
        logfile = wave_log / f"{job['tag']}.log"
        final_E = final_err = "?"
        try:
            lines = logfile.read_text().splitlines()
            for line in reversed(lines):
                if "*** Final:" in line and final_E == "?":
                    final_E = line.split("E =")[1].split("±")[0].strip()
                    final_err = line.split("err =")[1].strip().split("%")[0].strip() + "%"
                    break
        except Exception:
            pass

        n, omega = extract_n_omega(job["cmd"])
        dmc = DMC_REF.get((n, omega), "?")

        result = {
            "name": job["name"],
            "tag": job["tag"],
            "gpu": job["gpu"],
            "N": n, "omega": omega, "dmc": dmc,
            "rc": rc, "final_E": final_E, "final_err": final_err,
        }
        results.append(result)
        status = "OK" if rc == 0 else f"FAIL(rc={rc})"
        print(f"  [{job['name']}] {status}  N={n} ω={omega}  final={final_E} ({final_err})  dmc={dmc}")

    (wave_dir / "summary.json").write_text(json.dumps(results, indent=2))
    return results


def main():
    started = datetime.now()
    print(f"Cascading CG-SR campaign — started {started:%Y-%m-%d %H:%M}")
    print(f"Output: {OUTDIR}")
    print()

    all_results = {}

    for wave_name, jobs in [
        ("Wave 1: Warm-start & Continue", WAVE1_JOBS),
        ("Wave 2: Cascade & Extend", WAVE2_JOBS),
        ("Wave 3: Final Polish", WAVE3_JOBS),
    ]:
        wave_dir = OUTDIR / wave_name.split(":")[0].lower().replace(" ", "_")
        results = launch_wave(wave_name, jobs, wave_dir)
        all_results[wave_name] = results
        time.sleep(5)

    elapsed = datetime.now() - started
    print(f"\n{'='*60}")
    print(f"  CAMPAIGN COMPLETE — {elapsed}")
    print(f"{'='*60}")

    # Best per target
    best = {}
    for wave, results in all_results.items():
        for r in results:
            key = (r["N"], r["omega"])
            try:
                err = float(r["final_err"].replace("%", "").replace("+", ""))
            except (ValueError, AttributeError):
                err = float("inf")
            if key not in best or err < best[key]["err"]:
                best[key] = {**r, "err": err, "wave": wave}

    print("\n  Best per (N, omega):")
    for (n, omega), r in sorted(best.items()):
        print(f"    N={n:>2} ω={omega:>4}  E={r['final_E']:>10}  err={r['final_err']:>8}  "
              f"dmc={r['dmc']}  [{r['wave']}/{r['name']}]")

    (OUTDIR / "campaign_summary.json").write_text(json.dumps(all_results, indent=2))
    print(f"\n  Full summary: {OUTDIR / 'campaign_summary.json'}")


if __name__ == "__main__":
    main()
