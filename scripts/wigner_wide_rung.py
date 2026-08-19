"""Focused test: can a WIDE (exploratory) Wigner-ring proposal drive the omega=0.0027 re-expansion the
narrow local proposal cannot? Start from the good omega=0.0035 state; proposal covers a broad radial band
so IS sees the lower-energy expanded configs and pulls the state out. Narrow via guarded re-fit."""
import sys, math
sys.path.insert(0, "src")
import torch
from analysis.system import System
from analysis import diagnostics as dg
dev = "cuda"; DT = torch.float64; N, d = 6, 2; TWO = 2*math.pi
def _rad_ld(rho, r0, sr): return -0.5*((rho-r0)/sr)**2 - math.log(sr*math.sqrt(TWO))
def sample_dir(n, r0, sr, sc, alpha):
    x = torch.empty(n, N, d, device=dev, dtype=DT); x[:, 0] = torch.randn(n, d, device=dev, dtype=DT)*sc
    rho = (r0 + sr*torch.randn(n, 5, device=dev, dtype=DT)).abs()
    g = torch.distributions.Dirichlet(torch.full((5,), float(alpha), device=dev, dtype=DT)).sample((n,))
    th = TWO*torch.rand(n, 1, device=dev, dtype=DT) + TWO*(torch.cumsum(g, -1)-g)
    x[:, 1:, 0] = rho*torch.cos(th); x[:, 1:, 1] = rho*torch.sin(th); return x
def logq_dir(x, r0, sr, sc, alpha):
    r = x.norm(dim=-1); th = torch.atan2(x[..., 1], x[..., 0]); B = x.shape[0]
    dirich = torch.distributions.Dirichlet(torch.full((5,), float(alpha), device=dev, dtype=DT))
    out = torch.full((B,), -1e30, device=dev, dtype=DT)
    for c in range(N):
        o = [i for i in range(N) if i != c]
        lc = -(x[:, c]**2).sum(-1)/(2*sc**2) - d*0.5*math.log(TWO*sc**2)
        rr = r[:, o]; lrad = (_rad_ld(rr, r0, sr) - torch.log(rr.clamp_min(1e-9))).sum(-1)
        ta_s, _ = torch.sort(th[:, o], dim=-1)
        gaps = (torch.diff(torch.cat([ta_s, ta_s[:, :1]+TWO], -1), dim=-1)/TWO).clamp_min(1e-9); gaps = gaps/gaps.sum(-1, keepdim=True)
        out = torch.logaddexp(out, lc + lrad + dirich.log_prob(gaps))
    return out - math.log(N)
def main():
    W = 0.0027; ell = 1/math.sqrt(W); r0c = 1.334*W**(-2/3.)
    JAS = dict(dL=5, hidden_dim=128, n_layers=2, act="gelu")
    BF = dict(msg_hidden=128, msg_layers=2, hidden=128, layers=3, act="silu", out_bound="tanh", bf_scale_init=0.7, zero_init_last=False)
    s = System(N=6, omega=W, d=2, arch="pinn", arch_kwargs=JAS, use_backflow=True, backflow_kwargs=BF, backflow_arch="ctnn", seed=0)
    ck = torch.load("results/analysis/2026-08-18_wigner_cascade/w0.0035.pt", map_location=dev)
    s.f_net.load_state_dict(ck["f_net"]); s.backflow_net.load_state_dict(ck["backflow"])
    exp = 0.69036*(W/0.01)**0.689
    print(f"=== omega={W} WIDE proposal: r0_classical={r0c/ell:.2f}ell, expected E~{exp:.3f} ===", flush=True)
    r0, sr, sc, alpha = 2.3*ell, 1.3*ell, 1.2*ell, 3.0
    params = [p for m in s.modules() for p in m.parameters()]
    opt = torch.optim.Adam([{"params": list(s.f_net.parameters()), "lr": 1e-5}, {"params": list(s.backflow_net.parameters()), "lr": 1e-4}])
    OVER, B = 3, 4096
    for step in range(701):
        x = sample_dir(OVER*B, r0, sr, sc, alpha)
        with torch.no_grad():
            E_L = dg.local_energy(s.log_psi, x, W, s.params, chunk=1024)
            lw = 2*s.log_psi(x) - logq_dir(x, r0, sr, sc, alpha); lw = lw - lw.max(); w = torch.softmax(lw, 0)
            ess = float(w.sum()**2/(w**2).sum())
            rstate = float((w*x.norm(dim=-1).mean(1)).sum()/w.sum())/ell
        if step % 20 == 0 and step > 0 and ess > 15:
            r0 = 0.7*r0 + 0.3*rstate*ell; sr = max(0.6*ell, 0.9*sr)
        med = E_L.median(); mad = (E_L-med).abs().median()+1e-9; E_cl = E_L.clamp(med-8*mad, med+8*mad); Ew = (w*E_cl).sum()
        loss = 2.0*(w.detach()*(E_cl-Ew).detach()*s.log_psi(x)).sum()
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if step % 100 == 0:
            print(f"  [{step:4d}] Ew={float(Ew):.4f} ESS={ess:6.0f} propr0={r0/ell:.2f}ell stater0={rstate:.2f}ell sr={sr/ell:.2f}", flush=True)
    x = s.sample(4096, steps=500, burn_in=1000)
    Em = dg.gs_quality(dg.local_energy(s.log_psi, x, W, s.params, chunk=256))["E_mean_raw"]
    print(f"\n  FINAL heavy E={Em:.4f} (expected ~{exp:.3f}, ratio {Em/exp:.2f})", flush=True)
main()
