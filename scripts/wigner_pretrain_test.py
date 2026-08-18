"""Q3: ESS-maximising pretrain onto a Wigner-ring proposal, then collocation refinement.
Phase A: minimise Var_{R~q}[2 log|Psi| - log q] (no MCMC) -> ESS -> n by construction.
Phase B: REINFORCE collocation with the ring proposal. Run: python3 scripts/wigner_pretrain_test.py"""
import sys, math
sys.path.insert(0, "src")
import torch
from analysis.system import System
from analysis import diagnostics as dg
dev = "cuda"; DT = torch.float64; N, d = 6, 2
REF = 0.140832

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
def ess_of(s, r0, sr, sc, n=4096):
    x, lq = samp15(n, r0, sr, sc); lp = 2*s.log_psi(x).double() - lq; lp = lp - lp.max()
    return math.exp(2*torch.logsumexp(lp,0).item() - torch.logsumexp(2*lp,0).item())
def heavy_E(s, n=4096):
    x = s.sample(n, steps=400, burn_in=800)
    q = dg.gs_quality(dg.local_energy(s.log_psi, x, s.omega, s.params, chunk=256), ref_energy=REF)
    return q["E_mean_raw"], q.get("error_pct")
def main():
    W = 0.001; ell = 1/math.sqrt(W)
    JAS = dict(dL=5, hidden_dim=128, n_layers=2, act="gelu")
    BF = dict(msg_hidden=128, msg_layers=2, hidden=128, layers=3, act="silu",
              out_bound="tanh", bf_scale_init=0.7, zero_init_last=False)
    s = System(N=6, omega=W, d=2, arch="pinn", arch_kwargs=JAS, use_backflow=True,
               backflow_kwargs=BF, backflow_arch="ctnn", seed=0)
    ck = torch.load("results/analysis/2026-07-16_scaling/N6_ctnnbf_s0_w0p01/checkpoint.pt", map_location=dev)
    s.f_net.load_state_dict(ck["f_net"]); s.backflow_net.load_state_dict(ck["backflow"])
    r0 = 1.334*W**(-2/3.); sr = 0.64*ell; sc = 1.3*ell
    print(f"=== N=6 omega={W}: ESS-max pretrain -> colloc. ring r0={r0:.1f} ({r0/ell:.2f}ell) ===", flush=True)
    print(f"  ESS before: {ess_of(s,r0,sr,sc):.1f}/4096  energy before: {heavy_E(s)[1]:+.1f}%", flush=True)
    params = [p for m in s.modules() for p in m.parameters()]
    optA = torch.optim.Adam(params, lr=1e-3)
    for step in range(601):
        x, lq = samp15(2048, r0, sr, sc)
        loss = (2*s.log_psi(x) - lq).var()
        optA.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); optA.step()
        if step % 150 == 0:
            print(f"    [preA {step:4d}] Var(logw)={float(loss):.3f}  ESS={ess_of(s,r0,sr,sc):7.1f}", flush=True)
    eA = heavy_E(s)[1]
    print(f"  ESS after pretrain: {ess_of(s,r0,sr,sc):.1f}/4096  energy after: {eA:+.1f}%", flush=True)
    optB = torch.optim.Adam([{"params": list(s.f_net.parameters()), "lr": 5e-5},
                             {"params": list(s.backflow_net.parameters()), "lr": 5e-4}])
    for step in range(801):
        x, lq = samp15(4096, r0, sr, sc)
        with torch.no_grad():
            E_L = dg.local_energy(s.log_psi, x, W, s.params, chunk=512)
            lw = 2*s.log_psi(x) - lq; lw = lw - lw.max(); w = torch.softmax(lw, 0)
        med = E_L.median(); mad = (E_L-med).abs().median() + 1e-9
        E_cl = E_L.clamp(med-8*mad, med+8*mad); Ew = (w*E_cl).sum()
        loss = 2.0*(w.detach()*(E_cl-Ew).detach()*s.log_psi(x)).sum()
        optB.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); optB.step()
        if step % 200 == 0:
            print(f"    [preB {step:4d}] Ew={float(Ew):.4f} ESS={float(w.sum()**2/(w**2).sum()):7.1f}", flush=True)
    eB = heavy_E(s)[1]
    print(f"\n  FINAL: err={eB:+.3f}%  [pretrain->{eA:+.1f}%, colloc->{eB:+.3f}%; thesis +0.25%]", flush=True)
main()
