"""Angular order of the N=6 Wigner ring vs omega: liquid (broad gaps) or crystal (peaked at 72 deg)?
Explains why the independent-uniform-angle ring proposal collapses ESS at omega=0.001."""
import sys, math
sys.path.insert(0, "src")
import torch, numpy as np
from analysis.system import load_system
dev = "cuda"
@torch.no_grad()
def angular_order(ck, tag):
    s = load_system(ck, device=dev, seed=0); W = s.omega
    x = s.sample(2048, steps=500, burn_in=1000).double()
    r = x.norm(dim=-1)
    idx = r.argmin(1); ar = torch.arange(r.shape[0], device=dev)
    keep = torch.ones_like(r, dtype=torch.bool); keep[ar, idx] = False
    th = torch.atan2(x[..., 1], x[..., 0])
    gaps = []
    for b in range(x.shape[0]):
        a = th[b][keep[b]].sort().values
        g = torch.diff(torch.cat([a, a[:1] + 2*math.pi]))
        gaps.append(g)
    gaps = torch.cat(gaps).cpu().numpy() * 180/math.pi
    print(f"  [{tag}] omega={W}: gap mean={gaps.mean():.1f} std={gaps.std():.1f} "
          f"frac<20deg={np.mean(gaps<20):.2f}  (72=pentagon; std->0 crystal, large=liquid)", flush=True)
for w,t in [("w1","1.0 "),("w0p1","0.1 "),("w0p01","0.01"),("w0p001","0.001+25%")]:
    angular_order(f"results/analysis/2026-07-16_scaling/N6_ctnnbf_s0_{w}/checkpoint.pt", t)
