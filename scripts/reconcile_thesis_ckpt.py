"""Localise the +8% thesis-checkpoint mismatch: is the loaded thesis state the SAME as my good state
(overlap^2~1 -> eval/normalisation issue) or DIFFERENT (assembly/config issue)? N=6 omega=1."""
import sys, math, itertools
sys.path.insert(0, "src")
import torch
from analysis.system import System, load_system
from analysis import diagnostics as dg
dev = "cuda"; DT = torch.float64; N, d = 6, 2; REF = 20.15932
def sd_of(p):
    x = torch.load(p, map_location=dev); return x["state_dict"] if isinstance(x, dict) and "state_dict" in x else x
@torch.no_grad()
def overlap_sq(sa, sb, n=2048):
    def h(src, oth):
        x = src.sample(n, steps=400, burn_in=800)
        r = (oth.log_psi(x) - src.log_psi(x)).double(); return torch.logsumexp(r, 0) - math.log(r.numel())
    return float(torch.exp(h(sa, sb) + h(sb, sa)).clamp(max=1.0))
def energy(s, n=2048):
    x = s.sample(n, steps=400, burn_in=800)
    return float(dg.gs_quality(dg.local_energy(s.log_psi, x, s.omega, s.params, chunk=256), ref_energy=REF)["error_pct"])
# my good state
mine = load_system("results/analysis/2026-07-16_scaling/N6_ctnnbf_s0_w1/checkpoint.pt", device=dev, seed=0)
print(f"my good state N=6 w=1: err={energy(mine):+.3f}%", flush=True)
fst = sd_of("results/official_models/6p/w_10/f_netCTNN.pt")
bst = sd_of("results/official_models/6p/w_10/backflowCTNN.pt")
JAS0 = dict(dL=5, hidden_dim=128, n_layers=2)
for jact, bgate in itertools.product(["gelu", "silu"], [False, True]):
    BF = dict(msg_hidden=128, msg_layers=2, hidden=128, layers=3, act="silu", aggregation="sum",
              use_spin=True, same_spin_only=False, out_bound="tanh", bf_scale_init=0.05, zero_init_last=False, hard_cusp_gate=bgate)
    s = System(N=6, omega=1.0, d=2, arch="pinn", arch_kwargs=dict(JAS0, act=jact), use_backflow=True,
               backflow_kwargs=BF, backflow_arch="ctnn", seed=0)
    try:
        s.f_net.load_state_dict(fst); s.backflow_net.load_state_dict(bst)
    except Exception as e:
        print(f"  jas={jact} gate={bgate}: LOAD FAIL {type(e).__name__}"); continue
    print(f"  jas={jact} gate={bgate}: err={energy(s):+.2f}%  overlap^2(thesis,mine)={overlap_sq(s, mine):.4f}", flush=True)
