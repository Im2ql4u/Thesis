"""Q3: can fitted-ring collocation REFINE a structurally-correct-but-wrong-scale state toward DMC?
Start from the N=6 omega=0.001 VMC state (+25%: right (1,5) structure, ring over-expanded to ~6.5 ell).
Fit the ring to its CURRENT density, then collocation-refine (ring adaptively re-fit) and watch the
energy descend as the state contracts toward the classical ring (4.2 ell). Run under CUDA."""
import sys, math
sys.path.insert(0, "src")
import torch
from analysis.system import load_system
from analysis import diagnostics as dg
dev = "cuda"; DT = torch.float64; N, d = 6, 2; REF = 0.140832

def _ring_ld(r, r0, sr):
    rho = r.norm(dim=-1).clamp_min(1e-9)
    return -0.5*((rho-r0)/sr)**2 - math.log(sr*math.sqrt(2*math.pi)) - math.log(2*math.pi) - torch.log(rho)
def _cen_ld(r, sc):
    return -(r**2).sum(-1)/(2*sc**2) - d*0.5*math.log(2*math.pi*sc**2)
def logq15(x, r0, sr, sc):
    lc = _cen_ld(x, sc); lr = _ring_ld(x, r0, sr)
    return torch.logsumexp(lc + lr.sum(-1, keepdim=True) - lr, 1) - math.log(N)
def samp15(n, r0, sr, sc):
    x = torch.empty(n, N, d, device=dev, dtype=DT)
    x[:, 0] = torch.randn(n, d, device=dev, dtype=DT) * sc
    rho = (r0 + sr*torch.randn(n, 5, device=dev, dtype=DT)).abs()
    th = 2*math.pi*torch.rand(n, 5, device=dev, dtype=DT)
    x[:, 1:, 0] = rho*torch.cos(th); x[:, 1:, 1] = rho*torch.sin(th)
    return x, logq15(x, r0, sr, sc)
@torch.no_grad()
def fit_ring(x, w=None):
    r = x.norm(dim=-1); idx = r.argmin(1); ar = torch.arange(r.shape[0], device=dev)
    rc = r[ar, idx]; m = torch.ones_like(r, dtype=torch.bool); m[ar, idx] = False
    if w is None:
        rr = r[m]; return float(rr.mean()), float(rr.std()), float(rc.mean()+1e-6)
    wr = w[:, None].expand_as(r)[m]; rr = r[m]
    r0 = (wr*rr).sum()/wr.sum(); sr = ((wr*(rr-r0)**2).sum()/wr.sum()).sqrt().clamp_min(0.15*r0)
    return float(r0), float(sr), float((w*rc).sum()/w.sum()+1e-6)
def heavy_E(s, n=4096):
    x = s.sample(n, steps=400, burn_in=800)
    q = dg.gs_quality(dg.local_energy(s.log_psi, x, s.omega, s.params, chunk=256), ref_energy=REF)
    return q["E_mean_raw"], q.get("error_pct")
def main():
    s = load_system("results/analysis/2026-07-16_scaling/N6_ctnnbf_s0_w0p001/checkpoint.pt", device=dev, seed=0)
    W = s.omega; ell = 1/math.sqrt(W)
    print(f"=== N=6 omega={W}: fitted-ring collocation refine from +25% state, ell={ell:.1f} ===", flush=True)
    E0, er0 = heavy_E(s); print(f"  start: E={E0:.4f} ({er0:+.1f}%)", flush=True)
    with torch.no_grad():
        xs = s.sample(2048, steps=400, burn_in=800).double(); r0, sr, sc = fit_ring(xs)
    print(f"  ring fitted to state: r0={r0:.1f} ({r0/ell:.2f}ell) sr={sr:.1f} sc={sc:.1f}", flush=True)
    x, lq = samp15(4096, r0, sr, sc)
    lw = 2*s.log_psi(x).double() - lq; lw = lw - lw.max(); w = torch.softmax(lw, 0)
    print(f"  ESS at start: {float(w.sum()**2/(w**2).sum()):.1f}/4096", flush=True)
    params = [p for m in s.modules() for p in m.parameters()]
    opt = torch.optim.Adam([{"params": list(s.f_net.parameters()), "lr": 5e-5},
                            {"params": list(s.backflow_net.parameters()), "lr": 5e-4}])
    for step in range(1201):
        x, lq = samp15(4096, r0, sr, sc)
        with torch.no_grad():
            E_L = dg.local_energy(s.log_psi, x, W, s.params, chunk=512)
            lw = 2*s.log_psi(x) - lq; lw = lw - lw.max(); w = torch.softmax(lw, 0)
        med = E_L.median(); mad = (E_L-med).abs().median() + 1e-9
        E_cl = E_L.clamp(med-8*mad, med+8*mad); Ew = (w*E_cl).sum()
        loss = 2.0*(w.detach()*(E_cl-Ew).detach()*s.log_psi(x)).sum()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if step % 100 == 0:
            ess = float(w.sum()**2/(w**2).sum())
            with torch.no_grad(): r0, sr, sc = fit_ring(x, w)          # adaptive re-fit
            print(f"    [{step:4d}] Ew={float(Ew):.4f} ESS={ess:7.1f} r0={r0:.1f}({r0/ell:.2f}ell)", flush=True)
    Ef, erf = heavy_E(s); print(f"\n  FINAL: E={Ef:.5f} ({erf:+.3f}%)  [start {er0:+.1f}% -> {erf:+.3f}%; thesis +0.25%]", flush=True)
main()
