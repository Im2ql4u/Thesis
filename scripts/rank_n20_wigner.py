"""Q1 closure: backflow displacement rank (participation ratio) at N=20, omega=0.01 (Wigner).
Does the conventional backflow collapse toward rank 1 while CTNN holds ~N, as at N=6/12?"""
import sys, re, glob
sys.path.insert(0, "src")
import torch, numpy as np
from analysis.system import load_system
dev = "cuda"
def effrank(M):
    X = M - M.mean(0, keepdims=True); s = np.linalg.svd(X, compute_uv=False); l = s**2
    return float((l.sum()**2)/(l**2).sum()) if l.sum() > 0 else 0.0
for d in sorted(glob.glob("results/analysis/2026-07-16_scaling/N20_*bf_s*_w0p01")):
    if not glob.glob(d+"/checkpoint.pt"): continue
    m = re.match(r".*/N20_(ctnn|conv)bf_s(\d+)_w0p01$", d)
    arch, seed = m[1], int(m[2])
    s = load_system(d+"/checkpoint.pt", device=dev, seed=0)
    x = s.sample(512, steps=300, burn_in=600)
    with torch.no_grad(): dx = s.backflow_net(x, spin=s.spin)
    bfr = effrank(dx.reshape(x.shape[0], -1).cpu().double().numpy())
    print(f"N=20 {arch} s{seed} w=0.01: BFrank={bfr:5.1f}", flush=True)
    del x; torch.cuda.empty_cache()
