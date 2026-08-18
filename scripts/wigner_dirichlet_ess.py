"""Correct MCMC-free crystalline proposal: angular gaps ~ Dirichlet(alpha) (sum to 2pi, tractable
density, one param alpha = liquid<->crystal). Consistent sampler AND density (the earlier crystalline
test was not). Test ESS vs the N=6 omega=0.001 state; scan alpha; compare to uniform-angle ring."""
import sys, math
sys.path.insert(0, "src")
import torch
from analysis.system import load_system
dev = "cuda"; DT = torch.float64; N, d = 6, 2; TWO = 2*math.pi

def _rad_ld(rho, r0, sr):
    return -0.5*((rho-r0)/sr)**2 - math.log(sr*math.sqrt(TWO))

@torch.no_grad()
def sample_dir(n, r0, sr, sc, alpha):
    x = torch.empty(n, N, d, device=dev, dtype=DT)
    x[:, 0] = torch.randn(n, d, device=dev, dtype=DT)*sc
    rho = (r0 + sr*torch.randn(n, 5, device=dev, dtype=DT)).abs()
    gaps = torch.distributions.Dirichlet(torch.full((5,), float(alpha), device=dev, dtype=DT)).sample((n,))  # (n,5) sum1
    phi = TWO*torch.rand(n, 1, device=dev, dtype=DT)
    pos = torch.cumsum(gaps, dim=-1) - gaps                     # start of each gap, in [0,1)
    th = phi + TWO*pos
    x[:, 1:, 0] = rho*torch.cos(th); x[:, 1:, 1] = rho*torch.sin(th)
    return x

@torch.no_grad()
def logq_dir(x, r0, sr, sc, alpha):
    r = x.norm(dim=-1); th = torch.atan2(x[..., 1], x[..., 0])
    B = x.shape[0]; av = torch.full((5,), float(alpha), device=dev, dtype=DT)
    dirich = torch.distributions.Dirichlet(av)
    out = torch.full((B,), -1e30, device=dev, dtype=DT)
    for c in range(N):
        others = [i for i in range(N) if i != c]
        lc = -(x[:, c]**2).sum(-1)/(2*sc**2) - d*0.5*math.log(TWO*sc**2)
        rr = r[:, others]
        lrad = (_rad_ld(rr, r0, sr) - torch.log(rr.clamp_min(1e-9))).sum(-1)   # radial + polar Jacobian
        ta_s, _ = torch.sort(th[:, others], dim=-1)
        gaps = torch.diff(torch.cat([ta_s, ta_s[:, :1]+TWO], dim=-1), dim=-1) / TWO   # frac gaps, sum 1
        gaps = gaps.clamp_min(1e-9); gaps = gaps/gaps.sum(-1, keepdim=True)
        lang = dirich.log_prob(gaps)                            # exact Dirichlet density
        out = torch.logaddexp(out, lc + lrad + lang)
    return out - math.log(N)

@torch.no_grad()
def ess(s, x, logq):
    lp = 2*s.log_psi(x).double() - logq; lp = lp - lp.max()
    return math.exp(2*torch.logsumexp(lp, 0).item() - torch.logsumexp(2*lp, 0).item())

@torch.no_grad()
def uniform_ring(s, r0, sr, sc, n):
    x = torch.empty(n, N, d, device=dev, dtype=DT); x[:, 0] = torch.randn(n, d, device=dev, dtype=DT)*sc
    rho = (r0+sr*torch.randn(n,5,device=dev,dtype=DT)).abs(); thu = TWO*torch.rand(n,5,device=dev,dtype=DT)
    x[:,1:,0]=rho*torch.cos(thu); x[:,1:,1]=rho*torch.sin(thu)
    r=x.norm(dim=-1)
    def rl(rr): rho=rr.clamp_min(1e-9); return -0.5*((rho-r0)/sr)**2-math.log(sr*math.sqrt(TWO))-math.log(TWO)-torch.log(rho)
    lc=-(x**2).sum(-1)/(2*sc**2)-d*0.5*math.log(TWO*sc**2); lr=rl(r)
    return ess(s, x, torch.logsumexp(lc+lr.sum(-1,keepdim=True)-lr,1)-math.log(N))

s = load_system("results/analysis/2026-07-16_scaling/N6_ctnnbf_s0_w0p001/checkpoint.pt", device=dev, seed=0)
ell = 1/math.sqrt(s.omega)
xs = s.sample(2048, steps=400, burn_in=800).double(); rr = xs.norm(dim=-1)
idx = rr.argmin(1); ar=torch.arange(rr.shape[0],device=dev); m=torch.ones_like(rr,dtype=torch.bool); m[ar,idx]=False
r0=float(rr[m].mean()); sr=float(rr[m].std()); sc=float(rr[ar,idx].mean()+1e-6)
n=4096
print(f"=== N=6 omega={s.omega}: r0={r0:.1f}({r0/ell:.2f}ell) sr={sr:.1f} ===", flush=True)
print(f"  UNIFORM ring (alpha=1 liquid): ESS={uniform_ring(s,r0,sr,sc,n):7.1f}/{n}", flush=True)
for alpha in [1, 5, 20, 50, 92, 200, 500]:
    x = sample_dir(n, r0, sr, sc, alpha)
    print(f"  DIRICHLET alpha={alpha:4d}       : ESS={ess(s,x,logq_dir(x,r0,sr,sc,alpha)):7.1f}/{n}", flush=True)
