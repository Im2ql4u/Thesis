"""Thesis-integrity check: is the SOURCE cascade rung (w0.0035.pt) actually a Wigner molecule at its
OWN omega=0.0035? Equilibrate its own |Psi|^2 and measure the radial shell. Classical ring = 3.42 ell."""
import sys, math
sys.path.insert(0, "src")
import torch
from analysis.system import System
from functions.Stochastic_Reconfiguration import _persistent_rw
dev="cuda"; DT=torch.float64; N,d=6,2
JAS=dict(dL=5,hidden_dim=128,n_layers=2,act="gelu")
BF=dict(msg_hidden=128,msg_layers=2,hidden=128,layers=3,act="silu",out_bound="tanh",bf_scale_init=0.7,zero_init_last=False)
W=0.0035; ell=1/math.sqrt(W); r0c=1.334*W**(-2/3.)
s=System(N=6,omega=W,d=2,arch="pinn",arch_kwargs=JAS,use_backflow=True,backflow_kwargs=BF,backflow_arch="ctnn",seed=0)
ck=torch.load("results/analysis/2026-08-18_wigner_cascade/w0.0035.pt",map_location=dev)
s.f_net.load_state_dict(ck["f_net"]); s.backflow_net.load_state_dict(ck["backflow"])
print(f"=== source w0.0035.pt at its OWN omega={W}; classical ring {r0c/ell:.2f} ell ===",flush=True)
x0=torch.randn(1024,N,d,device=dev,dtype=DT)*ell
x,_,_,_=_persistent_rw(s.log_psi,x0,steps=1000,sigma=0.4*ell,adapt=True,target=0.5,adapt_lr=0.05)
x,_,_,_=_persistent_rw(s.log_psi,x,steps=400,sigma=0.4*ell,adapt=False,target=0.5,adapt_lr=0.0)
r=x.norm(dim=-1)
qs=torch.tensor([0.1,0.25,0.5,0.75,0.9],device=dev,dtype=DT)
print("  radial |r|/ell quantiles:", [f"{float(v)/ell:.2f}" for v in torch.quantile(r.flatten(),qs)],flush=True)
# also init at shell and see if it holds there (basin test at the source omega)
r0=r0c
xb=torch.empty(1024,N,d,device=dev,dtype=DT); xb[:,0]=torch.randn(1024,d,device=dev,dtype=DT)*0.5*ell
th=2*math.pi*(torch.arange(5,device=dev,dtype=DT)/5)[None,:]+0.3*torch.randn(1024,5,device=dev,dtype=DT)
rho=r0+0.3*ell*torch.randn(1024,5,device=dev,dtype=DT)
xb[:,1:,0]=rho*torch.cos(th); xb[:,1:,1]=rho*torch.sin(th)
xb,_,_,_=_persistent_rw(s.log_psi,xb,steps=1000,sigma=0.4*ell,adapt=True,target=0.5,adapt_lr=0.05)
rb=xb.norm(dim=-1)
print("  shell-init |r|/ell quantiles:", [f"{float(v)/ell:.2f}" for v in torch.quantile(rb.flatten(),qs)],flush=True)
