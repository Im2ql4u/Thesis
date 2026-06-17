"""Body-order (anchored functional ANOVA) decomposition of a wavefunction component.

Decomposes a permutation-symmetric scalar f(x) (e.g. log|Psi| or the Jastrow J) into the variance
explained by <=k-body interactions, using the anchored (cut-HDMR) decomposition:

  T0 = f(a)
  c_i(x)   = f(a; i<-x_i) - T0
  c_ij(x)  = f(a; {i,j}<-x) - c_i - c_j - T0
  ...
  T_k(x)   = T0 + sum_{|S|<=k} c_S(x)
  captured_k = 1 - Var_x[f(x) - T_k(x)] / Var_x[f(x)]

A genuine k-body ansatz needs order k to capture ~all variance: a pairwise (DeepSet) Jastrow is
fully captured at k=2; message-passing (CTNN) leaves residual variance beyond k=2 iff it really
encodes 3+-body correlations. Averaged over several anchors for robustness.
"""

from __future__ import annotations

import itertools

import numpy as np
import torch


@torch.no_grad()
def _eval_subset(f_eval, x: torch.Tensor, anchor: torch.Tensor, subset) -> torch.Tensor:
    """f evaluated on configs where particles in `subset` come from x, the rest from anchor."""
    B, N, d = x.shape
    cfg = anchor.view(1, N, d).expand(B, N, d).clone()
    if subset:
        idx = torch.tensor(subset, device=x.device)
        cfg[:, idx, :] = x[:, idx, :]
    return f_eval(cfg).reshape(B).double()


@torch.no_grad()
def body_order_anova(f_eval, x: torch.Tensor, *, anchors: torch.Tensor, max_order: int = 3) -> dict:
    """captured_k for k=1..max_order, averaged over the given anchor configs.

    f_eval: (M,N,d)->(M,). x: (B,N,d) samples ~|Psi|^2. anchors: (A,N,d)."""
    B, N, d = x.shape
    max_order = min(max_order, N)
    subsets_by_size = {m: list(itertools.combinations(range(N), m)) for m in range(1, max_order + 1)}

    captured = {k: [] for k in range(1, max_order + 1)}
    total_var = []
    for a in range(anchors.shape[0]):
        anchor = anchors[a]
        T0 = f_eval(anchor.view(1, N, d)).reshape(()).double()
        f_full = f_eval(x).reshape(B).double()
        comp = {}  # subset -> c_S(x)  (B,)
        # build components order by order (inclusion-exclusion)
        for m in range(1, max_order + 1):
            for S in subsets_by_size[m]:
                fS = _eval_subset(f_eval, x, anchor, S)
                c = fS - T0
                for r in range(1, m):
                    for sub in itertools.combinations(S, r):
                        c = c - comp[sub]
                comp[S] = c
        var_f = float(f_full.var())
        total_var.append(var_f)
        T = T0.expand(B).clone()
        for k in range(1, max_order + 1):
            for S in subsets_by_size[k]:
                T = T + comp[S]
            resid = float((f_full - T).var())
            captured[k].append(1.0 - resid / (var_f + 1e-30))
    return {
        "captured": {k: float(np.mean(v)) for k, v in captured.items()},
        "captured_std": {k: float(np.std(v)) for k, v in captured.items()},
        "total_var": float(np.mean(total_var)),
        "n_anchors": int(anchors.shape[0]),
        "max_order": max_order,
    }
