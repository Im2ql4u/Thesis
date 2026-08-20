"""Q3 MCMC-free Wigner cascade v3 = the complete fix:
 - WIDE exploratory proposal per rung (broad radial band, r0 at the midpoint of state..classical) so IS
   can DISCOVER the re-expansion the ω-Slater-rebuild forces;
 - TEMPERED-weight adaptive tracking (re-fit r0/sc/alpha from w^beta every few steps, beta<1) so the
   proposal robustly FOLLOWS the moving state even at low raw ESS -- no hard guard;
 - sigma_r kept wide (from the tempered fit x broaden, floored) so coverage never collapses.
Resumes from the good omega=0.0035 rung; heavy-VMC eval each rung. No MCMC in training."""
import sys, math, os
sys.path.insert(0, "src")
import torch
from analysis.system import System
from analysis import diagnostics as dg
dev = "cuda"; DT = torch.float64; N, d = 6, 2; TWO = 2*math.pi
OUT = "results/analysis/2026-08-18_wigner_v3"; os.makedirs(OUT, exist_ok=True)
JAS = dict(dL=5, hidden_dim=128, n_layers=2, act="gelu")
BF = dict(msg_hidden=128, msg_layers=2, hidden=128, layers=3, act="silu", out_bound="tanh", bf_scale_init=0.7, zero_init_last=False)
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
@torch.no_grad()
def tempered_fit(x, logw, ell, r0_min, beta=0.5, broaden=1.3, sr_floor=0.8):
    wt = torch.softmax(beta*(logw - logw.max()), 0)                 # tempered weights: robust, non-collapsing
    r = x.norm(dim=-1); th = torch.atan2(x[..., 1], x[..., 0]); B = r.shape[0]
    idx = r.argmin(1); ar = torch.arange(B, device=dev); m = torch.ones_like(r, dtype=torch.bool); m[ar, idx] = False
    wr = wt[:, None].expand_as(r); rc = (wt*r[ar, idx]).sum()/wt.sum()
    rr = r[m]; wrr = wr[m]; r0 = (wrr*rr).sum()/wrr.sum()
    sr = ((wrr*(rr-r0)**2).sum()/wrr.sum()).sqrt()
    ta_ring = torch.where(m, th, torch.full_like(th, 1e9)); ta5 = torch.sort(ta_ring, -1).values[:, :5]
    gaps = torch.diff(torch.cat([ta5, ta5[:, :1]+TWO], -1), dim=-1)/TWO
    gstd = float((wt*gaps.std(dim=-1)).sum()/wt.sum()); alpha = max(1.0, min(400.0, (4/25)/(gstd**2)/5 - 0.2))
    return float(max(r0, r0_min)), float(max(broaden*sr, sr_floor*ell)), float(rc+1e-6), alpha
@torch.no_grad()
def fit_state(s, n=2048):
    x = s.sample(n, steps=400, burn_in=800).double(); r = x.norm(dim=-1)
    idx = r.argmin(1); ar = torch.arange(r.shape[0], device=dev); m = torch.ones_like(r, dtype=torch.bool); m[ar, idx] = False
    return float(r[m].mean())
def heavy_E(s, ref, n=4096):
    x = s.sample(n, steps=500, burn_in=1000)
    q = dg.gs_quality(dg.local_energy(s.log_psi, x, s.omega, s.params, chunk=256), ref_energy=ref)
    return q["E_mean_raw"], q.get("error_pct")
def main():
    ck = torch.load("results/analysis/2026-08-18_wigner_cascade/w0.0035.pt", map_location=dev)
    fst, bst = ck["f_net"], ck["backflow"]
    OMEGAS = [0.0027, 0.0021, 0.0016, 0.0012, 0.001]
    OVER, B, STEPS = 2, 4096, 500
    for W in OMEGAS:
        s = System(N=6, omega=W, d=2, arch="pinn", arch_kwargs=JAS, use_backflow=True, backflow_kwargs=BF, backflow_arch="ctnn", seed=0)
        s.f_net.load_state_dict(fst); s.backflow_net.load_state_dict(bst)
        ell = 1/math.sqrt(W); classical = 1.334*W**(-2/3.); exp = 0.69036*(W/0.01)**0.689
        r0s = fit_state(s)                                           # current (under-expanded) ring radius
        r0 = 0.5*(r0s + classical); sr = max(0.6*abs(classical-r0s), 1.0*ell); sc = 1.0*ell; alpha = 3.0   # WIDE start
        print(f"=== omega={W:.4f} classical={classical/ell:.2f}ell state={r0s/ell:.2f}ell -> WIDE r0={r0/ell:.2f} sr={sr/ell:.2f} (exp E~{exp:.3f}) ===", flush=True)
        params = [p for m in s.modules() for p in m.parameters()]
        opt = torch.optim.Adam([{"params": list(s.f_net.parameters()), "lr": 1e-5}, {"params": list(s.backflow_net.parameters()), "lr": 1e-4}])
        for step in range(STEPS):
            x = sample_dir(OVER*B, r0, sr, sc, alpha)
            with torch.no_grad():
                E_L = dg.local_energy(s.log_psi, x, W, s.params, chunk=1024)
                logw = 2*s.log_psi(x) - logq_dir(x, r0, sr, sc, alpha); w = torch.softmax(logw - logw.max(), 0)
                ess = float(w.sum()**2/(w**2).sum())
            if step % 5 == 0 and step > 0:                           # TEMPERED tracking, no hard guard
                r0, sr, sc, alpha = tempered_fit(x, logw, ell, r0_min=0.7*classical)
            med = E_L.median(); mad = (E_L-med).abs().median()+1e-9; E_cl = E_L.clamp(med-8*mad, med+8*mad); Ew = (w*E_cl).sum()
            loss = 2.0*(w.detach()*(E_cl-Ew).detach()*s.log_psi(x)).sum()
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            if step % 125 == 0:
                print(f"  omega={W:.4f} [{step:4d}] Ew={float(Ew):.4f} ESS={ess:6.0f} r0={r0/ell:.2f}ell sr={sr/ell:.2f} a={alpha:.0f}", flush=True)
        fst = {k: v.detach().clone() for k, v in s.f_net.state_dict().items()}
        bst = {k: v.detach().clone() for k, v in s.backflow_net.state_dict().items()}
        torch.save({"f_net": fst, "backflow": bst, "omega": W, "arch": "pinn", "arch_kwargs": JAS,
                    "backflow_arch": "ctnn", "backflow_kwargs": BF, "N": 6}, f"{OUT}/w{W:.4f}.pt")
        E2, er2 = heavy_E(s, 0.140832 if abs(W-0.001) < 1e-9 else None)
        tag = f" ({er2:+.3f}%)" if er2 is not None else f" ratio={E2/exp:.2f}"
        print(f"  omega={W:.4f} DONE heavy E={E2:.4f}{tag}", flush=True)
    print("=== v3 cascade complete ===", flush=True)
main()
