"""Test whether hard_cusp_gate=True reconciles the thesis checkpoint: overlap^2 with my true-GS state
(overlap needs only log_psi, not the Laplacian, so it works despite the cdist double-derivative crash)."""
import sys, math, itertools
sys.path.insert(0, "src")
import torch
from analysis.system import System, load_system
dev = "cuda"; DT = torch.float64
def sd_of(p):
    x = torch.load(p, map_location=dev); return x["state_dict"] if isinstance(x, dict) and "state_dict" in x else x
@torch.no_grad()
def overlap_sq(sa, sb, n=3072):
    def h(src, oth):
        x = src.sample(n, steps=400, burn_in=800)
        r = (oth.log_psi(x) - src.log_psi(x)).double(); return torch.logsumexp(r, 0) - math.log(r.numel())
    return float(torch.exp(h(sa, sb) + h(sb, sa)).clamp(max=1.0))
mine = load_system("results/analysis/2026-07-16_scaling/N6_ctnnbf_s0_w1/checkpoint.pt", device=dev, seed=0)
fst = sd_of("results/official_models/6p/w_10/f_netCTNN.pt"); bst = sd_of("results/official_models/6p/w_10/backflowCTNN.pt")
print("overlap^2(thesis, my true-GS -0.008%) across configs (overlap-only, no Laplacian):", flush=True)
for jact, bact, gate, rad, pw in itertools.product(["gelu","silu"], ["silu"], [True, False], [0.30], [2.0]):
    BF = dict(msg_hidden=128, msg_layers=2, hidden=128, layers=3, act=bact, aggregation="sum",
              use_spin=True, same_spin_only=False, out_bound="tanh", bf_scale_init=0.05, zero_init_last=False,
              hard_cusp_gate=gate, cusp_gate_radius_aho=rad, cusp_gate_power=pw)
    s = System(N=6, omega=1.0, d=2, arch="pinn", arch_kwargs=dict(dL=5, hidden_dim=128, n_layers=2, act=jact),
               use_backflow=True, backflow_kwargs=BF, backflow_arch="ctnn", seed=0)
    try: s.f_net.load_state_dict(fst); s.backflow_net.load_state_dict(bst)
    except Exception as e: print(f"  jas={jact} gate={gate}: LOAD FAIL"); continue
    print(f"  jas={jact} bf={bact} gate={gate} rad={rad}: overlap^2={overlap_sq(s, mine):.4f}", flush=True)
