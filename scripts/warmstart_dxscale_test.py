"""Mechanism: does the backflow displacement |dx|/ell survive the cross-omega transfer?
Measure |dx| (in ell units) that the SAME trained backflow produces, evaluated on shell configs, at the
source omega=0.0035 vs the target omega=0.0027. If |dx|/ell shrinks, the outward-density mechanism fails
because the displacement no longer reaches from the Slater core (~1 ell) to the Wigner shell (~3.5 ell)."""
import sys, math
sys.path.insert(0, "src")
import torch
from analysis.system import System
dev="cuda"; DT=torch.float64; N,d=6,2
JAS=dict(dL=5,hidden_dim=128,n_layers=2,act="gelu")
BF=dict(msg_hidden=128,msg_layers=2,hidden=128,layers=3,act="silu",out_bound="tanh",bf_scale_init=0.7,zero_init_last=False)
ck=torch.load("results/analysis/2026-08-18_wigner_cascade/w0.0035.pt",map_location=dev)

def shell_cfg(W,n=1024):
    ell=1/math.sqrt(W); r0=1.334*W**(-2/3.)
    x=torch.empty(n,N,d,device=dev,dtype=DT)
    x[:,0]=torch.randn(n,d,device=dev,dtype=DT)*0.5*ell
    th=2*math.pi*(torch.arange(5,device=dev,dtype=DT)/5)[None,:]+0.3*torch.randn(n,5,device=dev,dtype=DT)
    rho=r0+0.3*ell*torch.randn(n,5,device=dev,dtype=DT)
    x[:,1:,0]=rho*torch.cos(th); x[:,1:,1]=rho*torch.sin(th)
    return x, ell, r0

for W in (0.0035, 0.0027):
    s=System(N=6,omega=W,d=2,arch="pinn",arch_kwargs=JAS,use_backflow=True,backflow_kwargs=BF,backflow_arch="ctnn",seed=0)
    s.f_net.load_state_dict(ck["f_net"]); s.backflow_net.load_state_dict(ck["backflow"])
    x,ell,r0=shell_cfg(W)
    with torch.no_grad():
        dx=s.backflow_net(x,spin=s.spin)
    dxn=dx.norm(dim=-1)  # per-electron |dx| abs
    # is the displacement inward (toward core)? radial component
    rhat=x/x.norm(dim=-1,keepdim=True).clamp_min(1e-12)
    dr=(dx*rhat).sum(-1)  # signed radial displacement (abs)
    print(f"omega={W}: ell={ell:.2f} shell r0={r0/ell:.2f} ell | "
          f"|dx|median={float(dxn.median())/ell:.3f} ell | "
          f"radial dr median={float(dr.median())/ell:+.3f} ell (neg=inward)", flush=True)
