"""Test the basin hypothesis: warm-start omega=0.0035->0.0027, but INITIALISE the sampler at the
classical Wigner shell (not ell scale). If |Psi|^2 then equilibrates expanded (~3.5 ell') instead of
collapsed (~1 ell'), the cross-omega collapse is a sampler-init basin artifact and the fix is trivial."""
import sys, math
sys.path.insert(0, "src")
import torch
from analysis.system import System
from functions.Stochastic_Reconfiguration import _persistent_rw
dev = "cuda"; DT = torch.float64; N, d = 6, 2
JAS = dict(dL=5, hidden_dim=128, n_layers=2, act="gelu")
BF = dict(msg_hidden=128, msg_layers=2, hidden=128, layers=3, act="silu", out_bound="tanh", bf_scale_init=0.7, zero_init_last=False)
W = 0.0027; ell = 1/math.sqrt(W); r0c = 1.334*W**(-2/3.)   # classical ring radius (abs)
s = System(N=6, omega=W, d=2, arch="pinn", arch_kwargs=JAS, use_backflow=True, backflow_kwargs=BF, backflow_arch="ctnn", seed=0)
ck = torch.load("results/analysis/2026-08-18_wigner_cascade/w0.0035.pt", map_location=dev)
s.f_net.load_state_dict(ck["f_net"]); s.backflow_net.load_state_dict(ck["backflow"])
print(f"=== warm-start -> omega={W}; classical ring r0={r0c/ell:.2f} ell' ===", flush=True)

@torch.no_grad()
def equilibrate(x0, tag):
    x, _, _, _ = _persistent_rw(s.log_psi, x0, steps=1500, sigma=0.4*ell, adapt=True, target=0.5, adapt_lr=0.05)
    x, _, _, _ = _persistent_rw(s.log_psi, x, steps=500, sigma=0.4*ell, adapt=False, target=0.5, adapt_lr=0.0)
    r = x.norm(dim=-1)
    print(f"  [{tag:26}] median |r| = {float(r.median())/ell:.2f} ell'", flush=True)

n = 1024
# A: default init at ell scale (the collapsed basin)
equilibrate(torch.randn(n, N, d, device=dev, dtype=DT)*ell, "init at ell (default)")
# B: init on the classical Wigner shell: ring electrons at r0c, one near origin
xb = torch.empty(n, N, d, device=dev, dtype=DT)
xb[:, 0] = torch.randn(n, d, device=dev, dtype=DT)*0.5*ell
th = 2*math.pi*(torch.arange(5, device=dev, dtype=DT)/5)[None,:] + 0.3*torch.randn(n,5,device=dev,dtype=DT)
rho = r0c + 0.3*ell*torch.randn(n,5,device=dev,dtype=DT)
xb[:, 1:, 0] = rho*torch.cos(th); xb[:, 1:, 1] = rho*torch.sin(th)
equilibrate(xb, "init at Wigner shell")
