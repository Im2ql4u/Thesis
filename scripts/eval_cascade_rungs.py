"""Heavy-VMC eval each saved gentle-cascade rung to localise where the descent broke.
Compare to the Wigner-scaling estimate E(w) ~ 0.69036*(w/0.01)^0.689 (anchors E(0.01)=0.690, E(0.001)=0.141)."""
import sys, math, glob, re
sys.path.insert(0, "src")
import torch
from analysis.system import load_system
from analysis import diagnostics as dg
dev = "cuda"
def expected(w): return 0.69036*(w/0.01)**0.689
for ck in sorted(glob.glob("results/analysis/2026-08-18_wigner_cascade/w*.pt"),
                 key=lambda p: -float(re.search(r"w([0-9.]+)\.pt", p).group(1))):
    w = float(re.search(r"w([0-9.]+)\.pt", ck).group(1))
    s = load_system(ck, device=dev, seed=0)
    x = s.sample(4096, steps=500, burn_in=1000)
    E = dg.local_energy(s.log_psi, x, w, s.params, chunk=256)
    q = dg.gs_quality(E, ref_energy=(0.140832 if abs(w-0.001) < 1e-9 else None))
    Em = q["E_mean_raw"]; exp = expected(w)
    r0 = float(x.norm(dim=-1).median())/(1/math.sqrt(w))
    print(f"  omega={w:.4f}: E={Em:.4f}  expected~{exp:.4f}  ratio={Em/exp:.2f}  ring~{r0:.2f}ell", flush=True)
