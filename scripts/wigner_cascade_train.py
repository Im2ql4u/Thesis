"""Proper MCMC-free push to N=6 omega=0.001: Wigner-ring (Dirichlet angular gaps) proposal + OVERSAMPLING
+ adaptive re-fit + omega-cascade, seeded from the good omega=0.01 state. REINFORCE, Adam, MAD clip.
No MCMC in training; MCMC only for the final heavy energy certification."""
import sys, math
sys.path.insert(0, "src")
import torch
from analysis.system import System
from analysis import diagnostics as dg
dev = "cuda"; DT = torch.float64; N, d = 6, 2; TWO = 2*math.pi
REF001 = 0.140832

def _rad_ld(rho, r0, sr): return -0.5*((rho-r0)/sr)**2 - math.log(sr*math.sqrt(TWO))
def sample_dir(n, r0, sr, sc, alpha):
    x = torch.empty(n, N, d, device=dev, dtype=DT); x[:, 0] = torch.randn(n, d, device=dev, dtype=DT)*sc
    rho = (r0 + sr*torch.randn(n, 5, device=dev, dtype=DT)).abs()
    g = torch.distributions.Dirichlet(torch.full((5,), float(alpha), device=dev, dtype=DT)).sample((n,))
    th = TWO*torch.rand(n, 1, device=dev, dtype=DT) + TWO*(torch.cumsum(g, -1)-g)
    x[:, 1:, 0] = rho*torch.cos(th); x[:, 1:, 1] = rho*torch.sin(th)
    return x
def logq_dir(x, r0, sr, sc, alpha):
    r = x.norm(dim=-1); th = torch.atan2(x[..., 1], x[..., 0]); B = x.shape[0]
    dirich = torch.distributions.Dirichlet(torch.full((5,), float(alpha), device=dev, dtype=DT))
    out = torch.full((B,), -1e30, device=dev, dtype=DT)
    for c in range(N):
        o = [i for i in range(N) if i != c]
        lc = -(x[:, c]**2).sum(-1)/(2*sc**2) - d*0.5*math.log(TWO*sc**2)
        rr = r[:, o]; lrad = (_rad_ld(rr, r0, sr) - torch.log(rr.clamp_min(1e-9))).sum(-1)
        ta_s, _ = torch.sort(th[:, o], dim=-1)
        gaps = (torch.diff(torch.cat([ta_s, ta_s[:, :1]+TWO], -1), dim=-1)/TWO).clamp_min(1e-9)
        gaps = gaps/gaps.sum(-1, keepdim=True)
        out = torch.logaddexp(out, lc + lrad + dirich.log_prob(gaps))
    return out - math.log(N)
@torch.no_grad()
def fit_ring(s, n=2048):
    x = s.sample(n, steps=400, burn_in=800).double(); r = x.norm(dim=-1)
    idx = r.argmin(1); ar = torch.arange(r.shape[0], device=dev); m = torch.ones_like(r, dtype=torch.bool); m[ar, idx] = False
    th = torch.atan2(x[..., 1], x[..., 0])
    gaps = []
    for b in range(min(512, x.shape[0])):
        a = th[b][m[b]].sort().values; gaps.append(torch.diff(torch.cat([a, a[:1]+TWO])))
    gstd = torch.cat(gaps).std().item()
    frac = (gstd/TWO); alpha = max(1.0, (4/25)/(frac**2) / 5 - 0.2)   # invert Dirichlet gap variance
    return float(r[m].mean()), float(r[m].std()), float(r[ar, idx].mean()+1e-6), alpha
@torch.no_grad()
def refit_weighted(x, w):
    """Cheap online re-fit from the current importance-weighted proposal batch (no MCMC), so the ring
    tracks the wavefunction as it contracts/crystallises. Returns r0, sr, sc, alpha."""
    r = x.norm(dim=-1); th = torch.atan2(x[..., 1], x[..., 0]); B = r.shape[0]
    idx = r.argmin(1); ar = torch.arange(B, device=dev)
    m = torch.ones_like(r, dtype=torch.bool); m[ar, idx] = False
    wr = w[:, None].expand_as(r)
    rc = (w * r[ar, idx]).sum() / w.sum()
    rr = r[m]; wrr = wr[m]
    r0 = (wrr * rr).sum() / wrr.sum()
    sr = ((wrr * (rr - r0) ** 2).sum() / wrr.sum()).sqrt().clamp_min(0.05 * r0)
    # angular gap variance (weighted): fit alpha from the per-config sorted-gap std
    gvar_num = torch.zeros((), device=dev, dtype=DT); wsum = torch.zeros((), device=dev, dtype=DT)
    ta = th.clone()
    ta_ring = torch.where(m, ta, torch.full_like(ta, 1e9))
    ta_s, _ = torch.sort(ta_ring, dim=-1)                       # ring angles first (5), center=+inf last
    ta5 = ta_s[:, :5]
    gaps = torch.diff(torch.cat([ta5, ta5[:, :1] + TWO], -1), dim=-1) / TWO   # frac gaps
    gstd = gaps.std(dim=-1)                                     # per-config
    gstd_w = (w * gstd).sum() / w.sum()
    frac = float(gstd_w); alpha = max(1.0, (4/25) / (frac ** 2) / 5 - 0.2)
    return float(r0), float(sr), float(rc + 1e-6), alpha


def heavy_E(s, n=4096):
    x = s.sample(n, steps=400, burn_in=800)
    q = dg.gs_quality(dg.local_energy(s.log_psi, x, s.omega, s.params, chunk=256), ref_energy=REF001 if abs(s.omega-1e-3)<1e-9 else None)
    return q["E_mean_raw"], q.get("error_pct")
def main():
    JAS = dict(dL=5, hidden_dim=128, n_layers=2, act="gelu")
    BF = dict(msg_hidden=128, msg_layers=2, hidden=128, layers=3, act="silu", out_bound="tanh", bf_scale_init=0.7, zero_init_last=False)
    ck = torch.load("results/analysis/2026-07-16_scaling/N6_ctnnbf_s0_w0p01/checkpoint.pt", map_location=dev)
    fst, bst = ck["f_net"], ck["backflow"]
    OVER = 2; B = 4096
    for W in [0.005, 0.002, 0.001]:
        s = System(N=6, omega=W, d=2, arch="pinn", arch_kwargs=JAS, use_backflow=True, backflow_kwargs=BF, backflow_arch="ctnn", seed=0)
        s.f_net.load_state_dict(fst); s.backflow_net.load_state_dict(bst)   # warm-start from previous omega
        ell = 1/math.sqrt(W)
        r0, sr, sc, alpha = fit_ring(s)
        print(f"=== omega={W} ell={ell:.1f}: ring r0={r0:.1f}({r0/ell:.2f}ell) sr={sr:.1f} alpha={alpha:.0f} ===", flush=True)
        params = [p for m in s.modules() for p in m.parameters()]
        opt = torch.optim.Adam([{"params": list(s.f_net.parameters()), "lr": 1e-5},
                                {"params": list(s.backflow_net.parameters()), "lr": 1e-4}])
        for step in range(401):
            x = sample_dir(OVER*B, r0, sr, sc, alpha)
            with torch.no_grad():
                E_L = dg.local_energy(s.log_psi, x, W, s.params, chunk=1024)
                lw = 2*s.log_psi(x) - logq_dir(x, r0, sr, sc, alpha); lw = lw - lw.max(); w = torch.softmax(lw, 0)
            med = E_L.median(); mad = (E_L-med).abs().median()+1e-9; E_cl = E_L.clamp(med-8*mad, med+8*mad)
            Ew = (w*E_cl).sum()
            loss = 2.0*(w.detach()*(E_cl-Ew).detach()*s.log_psi(x)).sum()
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            if step % 10 == 0 and step > 0:
                r0, sr, sc, alpha = refit_weighted(x, w)   # continuous tracking
            if step % 100 == 0:
                ess = float(w.sum()**2/(w**2).sum())
                print(f"    [{step:4d}] Ew={float(Ew):.4f} ESS={ess:7.1f}/{OVER*B}", flush=True)
        fst = {k: v.detach().clone() for k, v in s.f_net.state_dict().items()}
        bst = {k: v.detach().clone() for k, v in s.backflow_net.state_dict().items()}
        if abs(W-1e-3) < 1e-9:
            Ef, erf = heavy_E(s); print(f"\n  FINAL omega=0.001: E={Ef:.5f} ({erf:+.3f}%)  [thesis colloc +0.25%, DMC ref {REF001}]", flush=True)
main()
