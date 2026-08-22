"""Thread 3: isolate which component collapses the state on a cross-omega warm-start.
Load good omega=0.0035 params into an omega=0.0027 System; measure the radial density (in ell' units)
of |Slater|^2, |Slater x Jastrow|^2 (no backflow), and the full |Psi|^2. The classical omega=0.0027 ring
is at 3.57 ell'; the source omega=0.0035 state sits at ~2.9 ell' if absolute positions are preserved.
Whichever component first pulls the density inward is the culprit."""
import sys, math
sys.path.insert(0, "src")
import torch
from analysis.system import System
from functions.Stochastic_Reconfiguration import _persistent_rw
dev = "cuda"; DT = torch.float64; N, d = 6, 2
JAS = dict(dL=5, hidden_dim=128, n_layers=2, act="gelu")
BF = dict(msg_hidden=128, msg_layers=2, hidden=128, layers=3, act="silu", out_bound="tanh", bf_scale_init=0.7, zero_init_last=False)
W = 0.0027; ell = 1/math.sqrt(W)
s = System(N=6, omega=W, d=2, arch="pinn", arch_kwargs=JAS, use_backflow=True, backflow_kwargs=BF, backflow_arch="ctnn", seed=0)
ck = torch.load("results/analysis/2026-08-18_wigner_cascade/w0.0035.pt", map_location=dev)
s.f_net.load_state_dict(ck["f_net"]); s.backflow_net.load_state_dict(ck["backflow"])
print(f"=== warm-start omega=0.0035 -> {W}; ell'={ell:.1f}; classical ring 3.57 ell' ===", flush=True)

@torch.no_grad()
def radial(logfn, tag, n=1024):
    x = torch.randn(n, N, d, device=dev, dtype=DT)*ell
    x, _, _, _ = _persistent_rw(logfn, x, steps=800, sigma=0.4*ell, adapt=True, target=0.5, adapt_lr=0.05)
    x, _, _, _ = _persistent_rw(logfn, x, steps=400, sigma=0.4*ell, adapt=False, target=0.5, adapt_lr=0.0)
    r = x.norm(dim=-1)
    med = float(r.median()); ring = float(r[r > 0.5*r.median()].median())  # ring electrons
    print(f"  [{tag:22}] median |r| = {med/ell:.2f} ell'   ring |r| = {ring/ell:.2f} ell'", flush=True)

# component 1: bare Slater |det|^2
radial(s.log_slater, "bare Slater")
# component 2: Slater x Jastrow (no backflow)
bf = s.backflow_net; s.backflow_net = None
radial(s.log_psi, "Slater x Jastrow")
s.backflow_net = bf
# component 3: full
radial(s.log_psi, "full (+ backflow)")
