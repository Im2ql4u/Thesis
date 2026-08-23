"""Regenerate the four thesis figures that were never committed:
  - N{2,6,12}_all_densities.pdf : single-particle radial density P(r) across omega,
    computed from the saved |Psi|^2 sample bundles (samples_X_bohr) in results/tables/.
  - all_activations_shared_legend.pdf : activation functions and their 1st-3rd derivatives.
Run: python3 scripts/make_missing_figures.py
"""
import glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results/figures"
OMEGAS = [1.0, 0.5, 0.1, 0.01, 0.001]
WSTR = {1.0:"1.00000", 0.5:"0.50000", 0.1:"0.10000", 0.01:"0.01000", 0.001:"0.00100"}
MAXROWS = 300_000  # subsample cap for the histogram

def load_samples(N, w):
    """Return samples_X_bohr (n, N, 2) for the given (N, omega), from .npz or .pt; pick largest."""
    pat_npz = f"results/tables/**/gr_N{N}_run_omega_{WSTR[w]}.npz"
    pat_pt  = f"results/tables/**/gr_N{N}_run_omega_{WSTR[w]}.pt"
    cands = glob.glob(pat_npz, recursive=True) + glob.glob(pat_pt, recursive=True)
    best = None; best_n = 0
    for f in cands:
        try:
            if f.endswith(".npz"):
                d = np.load(f, allow_pickle=True)
                if "samples_X_bohr" not in d: continue
                n = d["samples_X_bohr"].shape[0]
                if n > best_n: best_n, best = n, ("npz", f)
            else:
                import torch
                d = torch.load(f, map_location="cpu")
                if "samples_X_bohr" not in d: continue
                n = d["samples_X_bohr"].shape[0]
                if n > best_n: best_n, best = n, ("pt", f)
        except Exception as e:
            print(f"    skip {f}: {e!r}")
    if best is None: return None
    kind, f = best
    if kind == "npz":
        X = np.load(f, allow_pickle=True)["samples_X_bohr"]
    else:
        import torch
        X = torch.load(f, map_location="cpu")["samples_X_bohr"].numpy()
    if X.shape[0] > MAXROWS:
        idx = np.random.default_rng(0).choice(X.shape[0], MAXROWS, replace=False)
        X = X[idx]
    return X

def radial_density_figure(N):
    fig, ax = plt.subplots(figsize=(7,4.3))
    cmap = plt.cm.viridis(np.linspace(0.05, 0.9, len(OMEGAS)))
    any_curve = False
    for w, col in zip(OMEGAS, cmap):
        X = load_samples(N, w)
        if X is None:
            print(f"  N={N} w={w}: no samples found"); continue
        r = np.linalg.norm(X, axis=-1).ravel()           # all electron radii (Bohr)
        r = r[np.isfinite(r)]
        # histogram as a probability density in r
        hi = np.quantile(r, 0.999)
        bins = np.linspace(0, hi, 300)
        h, edges = np.histogram(r, bins=bins, density=True)
        centers = 0.5*(edges[:-1]+edges[1:])
        ax.plot(centers, h, color=col, lw=1.8, label=rf"$\omega={w:g}$")
        any_curve = True
    if not any_curve:
        return False
    ax.set_xscale("log")
    ax.set_xlabel(r"$r\ [a_0^\ast]$ (Bohr)")
    ax.set_ylabel(r"$P(r)$")
    ax.set_title(rf"$N={N}$: single-particle radial density across $\omega$")
    ax.legend(frameon=False, ncol=1, fontsize=9)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    path = f"{OUT}/N{N}_all_densities.pdf"
    fig.savefig(path); plt.close(fig)
    print(f"  wrote {path}")
    return True

def activations_figure():
    x = np.linspace(-4, 4, 1000)
    from scipy.special import erf
    def gelu(x): return 0.5*x*(1+erf(x/np.sqrt(2)))
    def silu(x): return x/(1+np.exp(-x))
    def mish(x): return x*np.tanh(np.log1p(np.exp(x)))
    def tanh(x): return np.tanh(x)
    def relu(x): return np.maximum(0,x)
    acts = {"GELU":gelu, "SiLU":silu, "Mish":mish, "tanh":tanh, "ReLU":relu}
    def deriv(f, x, n):
        y = f(x)
        for _ in range(n):
            y = np.gradient(y, x)
        return y
    fig, axes = plt.subplots(2, 2, figsize=(9,6.5), sharex=True)
    titles = [r"$f(x)$", r"$f'(x)$", r"$f''(x)$", r"$f'''(x)$"]
    cmap = plt.cm.tab10(np.linspace(0,1,10))
    for k, ax in enumerate(axes.ravel()):
        for i,(name,f) in enumerate(acts.items()):
            y = f(x) if k==0 else deriv(f, x, k)
            ax.plot(x, y, color=cmap[i], lw=1.6, label=name)
        ax.set_title(titles[k]); ax.axhline(0,color="0.8",lw=0.6)
        ax.spines[["top","right"]].set_visible(False)
    axes[1,0].set_xlabel("$x$"); axes[1,1].set_xlabel("$x$")
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5,1.02))
    fig.tight_layout(rect=[0,0,1,0.97])
    path = f"{OUT}/all_activations_shared_legend.pdf"
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {path}")

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print("Radial-density figures:")
    for N in (2,6,12): radial_density_figure(N)
    print("Activations figure:")
    activations_figure()
    print("done.")
