"""Does an ANGULARLY-CORRELATED ring proposal (rotated jittered pentagon) fix the ESS collapse at deep
Wigner? Ring electrons sampled at phi + 72*k + jitter (crystalline), vs the old uniform-angle ring.
Density uses the sorted wrap-gaps: q_ang ~ prod_k N(gap_k; 72deg, s). Test vs the omega=0.001 state
(measured angular std 6.7deg). If ESS jumps from ~1, the angular fix is validated."""
import sys, math
sys.path.insert(0, "src")
import torch
from analysis.system import load_system
dev = "cuda"; DT = torch.float64; N, d = 6, 2
TWO = 2*math.pi; PENT = TWO/5

def _rad_ld(rho, r0, sr):  # 1D radial log-density N(rho;r0,sr)
    return -0.5*((rho-r0)/sr)**2 - math.log(sr*math.sqrt(TWO))

@torch.no_grad()
def logq_angular(x, r0, sr, sc, sth):
    """(1,5): center Gaussian + 5 ring electrons with radial Gaussian and crystalline angular gaps.
    Symmetrise over which electron is the center (N terms); ring angular part via sorted wrap-gaps."""
    r = x.norm(dim=-1); th = torch.atan2(x[...,1], x[...,0])           # (B,N)
    B = x.shape[0]
    out = torch.full((B,), -1e30, device=dev, dtype=DT)
    for c in range(N):                                                  # electron c = center
        others = [i for i in range(N) if i != c]
        lc = -(x[:,c]**2).sum(-1)/(2*sc**2) - d*0.5*math.log(TWO*sc**2)
        rr = r[:, others]                                              # (B,5) ring radii
        lrad = _rad_ld(rr, r0, sr).sum(-1) - torch.log(rr.clamp_min(1e-9)).sum(-1)  # radial + polar Jacobian
        ta = th[:, others]                                            # (B,5) ring angles
        ta_s, _ = torch.sort(ta, dim=-1)
        gaps = torch.diff(torch.cat([ta_s, ta_s[:, :1]+TWO], dim=-1), dim=-1)  # (B,5) wrap gaps
        lang = (-0.5*((gaps-PENT)/sth)**2 - math.log(sth*math.sqrt(TWO))).sum(-1) - math.log(TWO)
        out = torch.logaddexp(out, lc + lrad + lang)
    return out - math.log(N)

@torch.no_grad()
def sample_angular(n, r0, sr, sc, sth):
    x = torch.empty(n, N, d, device=dev, dtype=DT)
    x[:, 0] = torch.randn(n, d, device=dev, dtype=DT)*sc
    phi = TWO*torch.rand(n, 1, device=dev, dtype=DT)
    th = phi + PENT*torch.arange(5, device=dev, dtype=DT)[None,:] + sth*torch.randn(n,5,device=dev,dtype=DT)
    rho = (r0 + sr*torch.randn(n,5,device=dev,dtype=DT)).abs()
    x[:,1:,0] = rho*torch.cos(th); x[:,1:,1] = rho*torch.sin(th)
    return x

@torch.no_grad()
def uniform_logq(x, r0, sr, sc):  # old independent-uniform-angle ring, for comparison
    r = x.norm(dim=-1)
    def rl(rr): 
        rho=rr.clamp_min(1e-9); return -0.5*((rho-r0)/sr)**2-math.log(sr*math.sqrt(TWO))-math.log(TWO)-torch.log(rho)
    lc=-(x**2).sum(-1)/(2*sc**2)-d*0.5*math.log(TWO*sc**2); lr=rl(r)
    return torch.logsumexp(lc+lr.sum(-1,keepdim=True)-lr,1)-math.log(N)

@torch.no_grad()
def ess(s, x, logq):
    lp = 2*s.log_psi(x).double() - logq; lp = lp-lp.max()
    return math.exp(2*torch.logsumexp(lp,0).item()-torch.logsumexp(2*lp,0).item())

s = load_system("results/analysis/2026-07-16_scaling/N6_ctnnbf_s0_w0p001/checkpoint.pt", device=dev, seed=0)
ell = 1/math.sqrt(s.omega)
# fit radial from the state
xs = s.sample(2048, steps=400, burn_in=800).double(); rr = xs.norm(dim=-1)
idx = rr.argmin(1); ar=torch.arange(rr.shape[0],device=dev)
m = torch.ones_like(rr,dtype=torch.bool); m[ar,idx]=False
r0=float(rr[m].mean()); sr=float(rr[m].std()); sc=float(rr[ar,idx].mean()+1e-6)
print(f"=== N=6 omega={s.omega} state: r0={r0:.1f}({r0/ell:.2f}ell) sr={sr:.1f} ===", flush=True)
n=4096
# uniform-angle ring (old)
x=sample_angular(n,r0,sr,sc,10.0); print(f"  (using angular sampler for both; density differs)")
xu = None
import types
# old uniform: sample uniform angles
xu = torch.empty(n,N,d,device=dev,dtype=DT); xu[:,0]=torch.randn(n,d,device=dev,dtype=DT)*sc
rho=(r0+sr*torch.randn(n,5,device=dev,dtype=DT)).abs(); thu=TWO*torch.rand(n,5,device=dev,dtype=DT)
xu[:,1:,0]=rho*torch.cos(thu); xu[:,1:,1]=rho*torch.sin(thu)
print(f"  UNIFORM-angle ring : ESS={ess(s,xu,uniform_logq(xu,r0,sr,sc)):7.1f}/{n}", flush=True)
for sth_deg in [5,7,10,15,25]:
    sth=sth_deg*math.pi/180
    xa=sample_angular(n,r0,sr,sc,sth)
    print(f"  CRYSTAL sth={sth_deg:2d}deg   : ESS={ess(s,xa,logq_angular(xa,r0,sr,sc,sth)):7.1f}/{n}", flush=True)
