# unified_analysis.py — minimal public surface, tqdm integrated, block-wise conditioning
# Works with your utils.inject_params and psi_fn signatures.

import torch
from torch import nn
from torch.nn.utils import parameters_to_vector
from tqdm.auto import tqdm

from utils import inject_params


@inject_params
def analyze_model_all(
    f_net: nn.Module,
    C_occ: torch.Tensor,
    *,
    psi_fn,  # psi_fn(f_net, x, C_occ, ...) -> (logpsi, psi)
    compute_coulomb_interaction,  # callable(x) -> (B,) or (B,1)
    rho_module: nn.Module,  # e.g., f_net.rho
    params=None,  # injected dict
    backflow_net: nn.Module | None = None,
    spin: torch.Tensor | None = None,
    # --- sampling ---
    batch_size: int = 1024,
    sampler_steps: int = 30,
    sampler_step_sigma: float = 0.2,
    # --- estimators ---
    hvp_probes: int = 6,  # Hutchinson probes for Δ logψ
    sr_damping: float = 1e-3,  # λ in (S+λI)Δ = -g
    center_scores: bool = True,  # center the score matrix before building S
    center_O: bool | None = None,  # deprecated alias (backward-compat)
    perm_trials: int = 1,  # 1–3 for smoother ΔE
    # --- curvature (power iter on A = J^T J for residual) ---
    do_curvature_global: bool = True,
    curv_iters: int = 8,
    curv_fd_eps: float | None = None,
    # --- UI ---
    use_tqdm: bool = True,
) -> dict:
    """
    Returns:
      report = {
        "sampler":  {"accept_rate": float},
        "baseline": {"E_mean": float, "E_std": float},
        "blocks":   [ {block,P,grad_norm,step_norm,dE_lin,dE_quad,dE_total}, ... ],
        "rho":      {"saliency": (F,), "perm_deltaE": (F,), "E0": float, "accept_rate": float},
        "curvature": {
            "global": {"lambda_max_est": float, "rayleigh_history": list, "num_params": int},
            "blocks": [
               {"block": str, "P": int, "lam_max": float, "lam_min_rand": float,
                "kappa_est": float, "trA": float, "trA2": float,
                  "eig_mean": float, "eig_rms": float}
            ]
        }
      }
    """
    # ----------------- config -----------------
    device = params["device"]
    dtype = params.get("torch_dtype", torch.float32)
    n_particles = int(params["n_particles"])
    d = int(params["d"])
    omega = float(params["omega"])
    E_target = float(params["E"])
    kappa = float(params.get("kappa", 0.0))

    # Back-compat: if legacy center_O was explicitly provided, honor it
    if center_O is not None:
        center_scores = bool(center_O)

    # ------------ spin ------------
    if spin is None:
        up = n_particles // 2
        spin = torch.cat(
            [torch.zeros(up, dtype=torch.long), torch.ones(n_particles - up, dtype=torch.long)]
        ).to(device)
    else:
        spin = spin.to(device)

    # ------------ closures ------------
    def psi_log_fn(x):
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)
        logpsi, _ = psi_fn(f_net, x, C_occ, backflow_net=backflow_net, spin=spin, params=params)
        return logpsi

    def local_energy_hvp(x):
        x = x.requires_grad_(True)
        logpsi = psi_log_fn(x)

        # ∇ logψ
        g = torch.autograd.grad(logpsi.sum(), x, create_graph=True, retain_graph=True)[0]  # (B,N,d)
        g2 = (g**2).sum(dim=(1, 2))

        # Hutchinson Δ logψ
        B = x.shape[0]
        acc = torch.zeros(B, device=x.device, dtype=x.dtype)
        for _ in range(hvp_probes):
            v = torch.empty_like(x).bernoulli_(0.5).mul_(2).add_(-1)
            hv = torch.autograd.grad((g * v).sum(), x, create_graph=True, retain_graph=True)[0]
            acc += (v * hv).sum(dim=(1, 2))
        lap_log = acc / max(1, hvp_probes)

        # potentials (physical coords)
        V_harm = 0.5 * (omega**2) * (x**2).sum(dim=(1, 2))
        V_int = compute_coulomb_interaction(x)
        V_int = V_int.view(-1) if V_int.ndim > 1 else V_int

        E_L = -0.5 * (lap_log + g2) + (V_harm + V_int)
        return E_L.detach(), logpsi.detach()

    @torch.no_grad()
    def sample_psi2(
        x0, n_steps=sampler_steps, step_sigma=sampler_step_sigma
    ) -> tuple[torch.Tensor, float]:
        x = x0.clone().requires_grad_(True)
        lp = psi_log_fn(x) * 2.0
        acc_sum = 0.0
        rng = range(n_steps)
        if use_tqdm:
            rng = tqdm(rng, desc="Sampler", leave=False)
        for _ in rng:
            prop = (x + torch.randn_like(x) * step_sigma).requires_grad_(True)
            lp_prop = psi_log_fn(prop) * 2.0
            accept = (torch.rand_like(lp_prop).log() < (lp_prop - lp)).view(-1, 1, 1)
            acc_sum += accept.float().mean().item()
            x = torch.where(accept, prop, x)
            lp = torch.where(accept.view(-1), lp_prop, lp)
        return x, acc_sum / max(1, n_steps)

    # ---------- block param map ----------
    def block_map():
        blocks = {
            "phi": getattr(f_net, "phi", None),
            "psi": getattr(f_net, "psi", None),
            "rho": getattr(f_net, "rho", None),
        }
        sizes, names, all_params = [], [], []
        for name, m in blocks.items():
            names.append(name)
            if m is None:
                sizes.append(0)
                continue
            ps = [p for p in m.parameters() if p.requires_grad]
            sizes.append(parameters_to_vector(ps).numel() if ps else 0)
            all_params.extend(ps)
        offsets, off = [], 0
        for s in sizes:
            offsets.append((off, off + s))
            off += s
        return names, sizes, offsets, all_params

    # ---------- robust per-sample score matrix (allow_unused=True, zero-fill) ----------
    def compute_score_matrix(x, params_list: list[torch.Tensor]) -> torch.Tensor:
        """
        Returns a (B, P) matrix of per-sample parameter scores:
            score[b, j] = ∂ logψ(x_b) / ∂ θ_j
        Missing grads are zero-filled to keep shapes consistent.
        """
        if not params_list:
            return torch.zeros(x.shape[0], 0, device=x.device, dtype=x.dtype)
        P = sum(p.numel() for p in params_list)
        B = x.shape[0]
        score_mat = torch.zeros(B, P, device=x.device, dtype=x.dtype)
        it = range(B)
        if use_tqdm:
            it = tqdm(it, total=B, desc="per-sample score grads (∂logψ/∂θ)", leave=False)
        for i in it:
            xi = x[i : i + 1].requires_grad_(True)
            logpsi_i = psi_log_fn(xi)
            grads = torch.autograd.grad(
                logpsi_i, params_list, retain_graph=False, allow_unused=True, create_graph=False
            )
            flat = []
            for g, p in zip(grads, params_list, strict=False):
                flat.append((torch.zeros_like(p) if g is None else g).reshape(-1))
            score_mat[i].copy_(torch.cat(flat))
        return score_mat

    # ---------- CG for SR ----------
    def cg(matvec, b, lam=sr_damping, tol=1e-6, it=100):
        x = torch.zeros_like(b)
        r = b - (matvec(x) + lam * x)
        p = r.clone()
        rs = r @ r
        for _ in range(it):
            Ap = matvec(p) + lam * p
            alpha = rs / (p @ Ap + 1e-20)
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = r @ r
            if rs_new.sqrt() < tol:
                break
            p = r + (rs_new / (rs + 1e-20)) * p
            rs = rs_new
        return x

    # ---------- exact Laplacian for residual ----------
    def laplacian_exact(psi, x):
        g = torch.autograd.grad(psi.sum(), x, create_graph=True)[0]
        lap = 0.0
        for k in range(d):
            gk = g[..., k]
            Hk = torch.autograd.grad(gk.sum(), x, create_graph=True, retain_graph=True)[0][..., k]
            lap = lap + Hk
        return lap.sum(dim=-1, keepdim=True)

    def potential_ho_coulomb(x):
        r2 = (x**2).sum(dim=(1, 2), keepdim=True)
        V_ho = 0.5 * (omega**2) * r2
        if kappa != 0.0:
            B, N, _ = x.shape
            diff = x.unsqueeze(2) - x.unsqueeze(1)
            ii, jj = torch.triu_indices(N, N, 1, device=x.device)
            rij = diff[:, ii, jj, :].norm(dim=-1, keepdim=True).clamp_min(1e-12)
            V_c = (kappa / rij).sum(dim=(1, 2), keepdim=True)
        else:
            V_c = torch.zeros_like(V_ho)
        return V_ho + V_c

    def residual_batch(x):
        logpsi, psi = psi_fn(f_net, x, C_occ, backflow_net=backflow_net, spin=spin, params=params)
        psi = psi.view(-1, 1)
        lap = laplacian_exact(psi, x)
        V = potential_ho_coulomb(x)
        r = (-0.5 * lap) + (V - E_target) * psi
        return r.squeeze(-1)  # (B,)

    # ---------- version-safe Av = J^T (J v) over ALL trainable params ----------
    def Av_full(v_flat, x, fd_eps=None):
        trainable = [p for p in f_net.parameters() if p.requires_grad]
        assert v_flat.numel() == sum(p.numel() for p in trainable)

        # unflatten v
        v_chunks, idx = [], 0
        for p in trainable:
            n = p.numel()
            v_chunks.append(v_flat[idx : idx + n].view_as(p))
            idx += n

        # choose FD step
        if fd_eps is None:
            with torch.no_grad():
                nrm = v_flat.norm().item()
                fd_eps = max(1e-7, 1e-3 * (1.0 if nrm == 0.0 else 1.0 / nrm))

        # central FD under no_grad, then RESTORE before building r0
        with torch.no_grad():
            for p, dv in zip(trainable, v_chunks, strict=False):
                p.add_(+fd_eps * dv)
        r_plus = residual_batch(x)

        with torch.no_grad():
            for p, dv in zip(trainable, v_chunks, strict=False):
                p.add_(-2.0 * fd_eps * dv)
        r_minus = residual_batch(x)

        with torch.no_grad():
            for p, dv in zip(trainable, v_chunks, strict=False):
                p.add_(+fd_eps * dv)  # restore

        Jv = (r_plus - r_minus) / (2.0 * fd_eps)  # (B,)

        # Build a FRESH r0 after all in-place ops are done
        r0 = residual_batch(x)  # (B,)

        grads = torch.autograd.grad(
            r0,
            trainable,
            grad_outputs=Jv,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        G_chunks = [
            (g if g is not None else torch.zeros_like(p))
            for g, p in zip(grads, trainable, strict=False)
        ]
        G = torch.cat([g.reshape(-1) for g in G_chunks])
        return G

    # ---------- block-wise Av = J_b^T (J_b v) ----------
    def _block_param_lists() -> dict[str, list[torch.Tensor]]:
        return {
            "phi": [
                p for p in getattr(f_net, "phi", nn.Identity()).parameters() if p.requires_grad
            ],
            "psi": [
                p for p in getattr(f_net, "psi", nn.Identity()).parameters() if p.requires_grad
            ],
            "rho": [
                p for p in getattr(f_net, "rho", nn.Identity()).parameters() if p.requires_grad
            ],
        }

    def _unflatten_like(vec: torch.Tensor, params_tmpl: list[torch.Tensor]) -> list[torch.Tensor]:
        out, i = [], 0
        for p in params_tmpl:
            n = p.numel()
            out.append(vec[i : i + n].view_as(p))
            i += n
        return out

    @torch.no_grad()
    def _choose_fd_eps_vec(v: torch.Tensor, rel=1e-3, abs_min=1e-7):
        n = float(v.norm().item())
        return max(abs_min, rel if n == 0.0 else rel / n)

    def Av_block(block_params: list[torch.Tensor], x: torch.Tensor, v_b: torch.Tensor, fd_eps=None):
        assert v_b.numel() == sum(p.numel() for p in block_params)
        v_chunks = _unflatten_like(v_b, block_params)
        if fd_eps is None:
            fd_eps = _choose_fd_eps_vec(v_b)

        with torch.no_grad():
            for p, dv in zip(block_params, v_chunks, strict=False):
                p.add_(+fd_eps * dv)
        r_plus = residual_batch(x)

        with torch.no_grad():
            for p, dv in zip(block_params, v_chunks, strict=False):
                p.add_(-2.0 * fd_eps * dv)
        r_minus = residual_batch(x)

        with torch.no_grad():
            for p, dv in zip(block_params, v_chunks, strict=False):
                p.add_(+fd_eps * dv)  # restore

        Jv = (r_plus - r_minus) / (2.0 * fd_eps)  # (B,)

        r0 = residual_batch(x)
        grads = torch.autograd.grad(
            r0, block_params, grad_outputs=Jv, allow_unused=True, retain_graph=False
        )
        Av_chunks = [
            (g if g is not None else torch.zeros_like(p))
            for g, p in zip(grads, block_params, strict=False)
        ]
        return torch.cat([g.reshape(-1) for g in Av_chunks])

    def block_curvature_stats(
        block_params: list[torch.Tensor],
        x: torch.Tensor,
        n_power: int = curv_iters,
        n_rand_min: int = 16,
        n_hutch: int = 16,
        fd_eps=None,
    ):
        if len(block_params) == 0:
            return dict(
                P=0,
                lam_max=0.0,
                lam_min_rand=0.0,
                kappa_est=float("inf"),
                trA=0.0,
                trA2=0.0,
                eig_mean=0.0,
                eig_rms=0.0,
                rays=[],
            )

        P = sum(p.numel() for p in block_params)
        device_ = block_params[0].device
        dtype_ = block_params[0].dtype

        # λ_max via power-iter
        v = torch.randn(P, device=device_, dtype=dtype_)
        v = v / (v.norm() + 1e-12)
        rays = []
        for _ in range(n_power):
            Av = Av_block(block_params, x, v, fd_eps)
            lam = float((v @ Av).item())
            rays.append(lam)
            nrm = Av.norm()
            if nrm < 1e-24:
                break
            v = Av / (nrm + 1e-12)
        lam_max = rays[-1] if rays else 0.0

        # λ_min+ proxy via random Rayleigh minima (keep dtype consistent)
        lam_min_rand = float("inf")
        for _ in range(n_rand_min):
            z = torch.randn(P, device=device_, dtype=dtype_)
            z = z / (z.norm() + 1e-12)
            Az = Av_block(block_params, x, z, fd_eps)
            if Az.dtype != z.dtype:
                Az = Az.to(z.dtype)
            rq = float((z @ Az).item())
            if rq < lam_min_rand:
                lam_min_rand = rq
        lam_min_rand = max(lam_min_rand, 0.0)

        # Hutchinson: tr(A) and tr(A^2) — Rademacher probes with correct dtype
        trA = 0.0
        trA2 = 0.0
        for _ in range(n_hutch):
            # Rademacher vector in the SAME dtype as parameters
            z = torch.randint(0, 2, (P,), device=device_, dtype=torch.long).to(dtype_) * 2 - 1
            z = z / (z.norm() + 1e-12)
            Az = Av_block(block_params, x, z, fd_eps)
            if Az.dtype != z.dtype:
                Az = Az.to(z.dtype)

            trA += float((z @ Az).item())
            trA2 += float((Az @ Az).item())

        trA /= max(1, n_hutch)
        trA2 /= max(1, n_hutch)

        eig_mean = trA / max(1, P)
        eig_rms = (trA2 / max(1, P)) ** 0.5
        kappa_est = float(lam_max / max(lam_min_rand, 1e-12))

        return dict(
            P=P,
            lam_max=lam_max,
            lam_min_rand=lam_min_rand,
            kappa_est=kappa_est,
            trA=trA,
            trA2=trA2,
            eig_mean=eig_mean,
            eig_rms=eig_rms,
            rays=rays,
        )

    # ------------ 1) sample x and baseline E ------------
    x0 = torch.randn(batch_size, n_particles, d, device=device, dtype=dtype)
    x, acc = sample_psi2(x0)
    E_L, _ = local_energy_hvp(x)
    baseline = {"E_mean": float(E_L.mean().item()), "E_std": float(E_L.std().item())}

    # ------------ 2) block-level SR importance ------------
    names, sizes, offsets, all_params = block_map()
    P_total = sum(sizes)
    blocks_out = []
    if P_total > 0:
        score_mat = compute_score_matrix(x, all_params)  # (B,P)
        B = score_mat.shape[0]
        score_mean = (
            score_mat.mean(dim=0)
            if center_scores
            else torch.zeros(P_total, device=x.device, dtype=x.dtype)
        )
        score_centered = score_mat - score_mean
        mu_E = E_L.mean()

        g_vec = 2.0 * ((score_centered * (E_L - mu_E).view(-1, 1)).mean(dim=0))  # (P,)

        def S_matvec(v):
            tmp = score_centered @ v
            return (score_centered.t() @ tmp) / B

        delta = cg(S_matvec, -g_vec)
        Sdelta = S_matvec(delta)

        for name, (lo, hi), sz in zip(names, offsets, sizes, strict=False):
            if sz == 0:
                blocks_out.append(
                    dict(
                        block=name,
                        P=0,
                        grad_norm=0.0,
                        step_norm=0.0,
                        dE_lin=0.0,
                        dE_quad=0.0,
                        dE_total=0.0,
                    )
                )
                continue
            g_b = g_vec[lo:hi]
            d_b = delta[lo:hi]
            Sd_b = Sdelta[lo:hi]
            dE_lin = float(-(g_b @ d_b).item())
            dE_quad = float(-0.5 * (d_b @ Sd_b).item())
            blocks_out.append(
                dict(
                    block=name,
                    P=sz,
                    grad_norm=float(g_b.norm().item()),
                    step_norm=float(d_b.norm().item()),
                    dE_lin=dE_lin,
                    dE_quad=dE_quad,
                    dE_total=dE_lin + dE_quad,
                )
            )

    # ------------ 3) ρ-feature saliency + permutation ΔE ------------
    captured = {"inp": None}

    def cap_hook(module, inp, out):
        captured["inp"] = inp[0]

    h_cap = rho_module.register_forward_hook(cap_hook)

    logpsi = psi_log_fn(x)  # one forward to populate hook
    z = captured["inp"]
    assert z is not None, "ρ hook did not capture its input."
    F = z.shape[-1]

    grads = torch.autograd.grad(logpsi.sum(), z, retain_graph=True)[0]
    reduce_dims = tuple(range(grads.ndim - 1))
    saliency = grads.abs().mean(dim=reduce_dims).detach().cpu()

    perm_spec = {"j": None, "perm_idx": None}

    def pre_hook(module, inp):
        y = inp[0]
        if perm_spec["j"] is None:
            return None
        j = perm_spec["j"]
        idx = perm_spec["perm_idx"]
        y_mod = y.clone()
        y_flat = y_mod.reshape(-1, y_mod.shape[-1])
        y_flat[:, j] = y_flat[idx, j]
        return (y_flat.view_as(y_mod),)

    h_pre = rho_module.register_forward_pre_hook(pre_hook)

    flat_len = z.reshape(-1, F).shape[0]
    base_idx = torch.randperm(flat_len, device=z.device)

    deltaE = torch.zeros(F, device=device)
    feat_iter = range(F)
    if use_tqdm:
        feat_iter = tqdm(feat_iter, desc="ρ permutation ΔE", leave=False)
    for j in feat_iter:
        accum = 0.0
        for _ in range(max(1, perm_trials)):
            perm_spec["j"] = j
            perm_spec["perm_idx"] = base_idx
            E_perm, _ = local_energy_hvp(x)
            accum += E_perm.mean().item()
        deltaE[j] = (accum / max(1, perm_trials)) - baseline["E_mean"]

    h_cap.remove()
    h_pre.remove()

    rho_out = {
        "saliency": saliency,  # (F,), CPU
        "perm_deltaE": deltaE.detach().cpu(),  # (F,), CPU
        "E0": baseline["E_mean"],
        "accept_rate": acc,
    }

    # ------------ 4) Global curvature (A = J^T J) ------------
    curvature = {}
    if do_curvature_global:
        trainable = [p for p in f_net.parameters() if p.requires_grad]
        P = sum(p.numel() for p in trainable)
        if P > 0:
            ref = trainable[0]
            v = torch.randn(P, device=ref.device, dtype=ref.dtype)
            v = v / (v.norm() + 1e-12)
            rays = []
            xb = x.detach().clone().requires_grad_(True)
            it = range(curv_iters)
            if use_tqdm:
                it = tqdm(it, desc="Power iter (A=J^T J)", leave=False)
            for _ in it:
                Av = Av_full(v, xb, curv_fd_eps)
                lam = float((v @ Av).item())
                rays.append(lam)
                n = Av.norm()
                if n < 1e-24:
                    break
                v = Av / n
            curvature["global"] = {
                "lambda_max_est": rays[-1] if len(rays) else 0.0,
                "rayleigh_history": rays,
                "num_params": P,
                "iters": len(rays),
            }
        else:
            curvature["global"] = {
                "lambda_max_est": 0.0,
                "rayleigh_history": [],
                "num_params": 0,
                "iters": 0,
            }

    # ------------ 4b) Block-wise curvature (A_b = J_b^T J_b) ------------
    block_stats = []
    block_params_map = _block_param_lists()
    x_curv = x.detach().clone().requires_grad_(True)
    for bname in ("phi", "psi", "rho"):
        stats = block_curvature_stats(
            block_params_map[bname],
            x_curv,
            n_power=curv_iters,
            n_rand_min=16,
            n_hutch=16,
            fd_eps=curv_fd_eps,
        )
        stats["block"] = bname
        block_stats.append(stats)
    curvature["blocks"] = block_stats

    return {
        "sampler": {"accept_rate": acc},
        "baseline": baseline,
        "blocks": blocks_out,
        "rho": rho_out,
        "curvature": curvature,
    }


def render_analysis_report(
    report: dict,
    rho_feature_names: list[str] | None = None,
    top_k: int = 10,
    to_markdown: str | None = None,
) -> str:
    import io

    out = io.StringIO()

    blocks = report["blocks"]
    rho = report["rho"]
    base = report["baseline"]
    samp = report["sampler"]
    curv = report.get("curvature", {})

    # helper formatters
    def _fmt(x, w=10, p=3, exp=True):
        if isinstance(x, float | int):
            try:
                return f"{x: {w}.{p}e}" if exp else f"{x: {w}.{p}f}"
            except Exception:
                return str(x)
        return f"{str(x):>{w}}"

    def _mk_table(rows, headers):
        if not rows:
            return " (no data)"
        colw = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
        line = " | ".join(h.ljust(colw[i]) for i, h in enumerate(headers))
        sep = "-|-".join("-" * colw[i] for i in range(len(headers)))
        body = "\n".join(" | ".join(r[i].ljust(colw[i]) for i in range(len(headers))) for r in rows)
        return line + "\n" + sep + "\n" + body

    print("# Importance Report", file=out)
    print(
        f"\nBaseline mean E_L: {base['E_mean']:.6f}   σ(E_L): {base['E_std']:.6f}   "
        f"Sampler acceptance: {samp['accept_rate']:.3f}\n",
        file=out,
    )

    # SR-style block importance
    for b in blocks:
        b["dE_total"] = float(b["dE_total"])
    blocks_sorted = sorted(blocks, key=lambda z: z["dE_total"], reverse=True)
    total_drop = sum(max(0.0, b["dE_total"]) for b in blocks_sorted) + 1e-12

    rows = []
    headers = ["block", "P", "‖g‖", "‖Δ‖", "−g·Δ", "−½ΔᵀSΔ", "ΔE total", "share"]
    for b in blocks_sorted:
        share = (max(0.0, b["dE_total"]) / total_drop) if total_drop > 0 else 0.0
        rows.append(
            [
                str(b["block"]),
                str(b["P"]),
                _fmt(b["grad_norm"]),
                _fmt(b["step_norm"]),
                _fmt(b["dE_lin"]),
                _fmt(b["dE_quad"]),
                _fmt(b["dE_total"]),
                f"{100*share:6.2f}%",
            ]
        )
    print("## Block ranking (SR-style)\n", file=out)
    print(_mk_table(rows, headers), file=out)

    # ρ feature importance
    sal = rho["saliency"]
    dE = rho["perm_deltaE"]
    F = sal.numel()
    idx_dE = torch.argsort(dE, descending=True)[: min(top_k, F)].tolist()
    idx_sal = torch.argsort(sal, descending=True)[: min(top_k, F)].tolist()

    def _fname(i):
        if rho_feature_names is not None and i < len(rho_feature_names):
            return rho_feature_names[i]
        return f"feature[{i}]"

    print("\n## ρ-feature importance", file=out)
    print(f"- Baseline mean E_L: {rho['E0']:.6f}", file=out)
    print(f"- Sampler acceptance: {rho['accept_rate']:.3f}", file=out)

    rows = []
    headers = ["rank", "feature", "ΔE (↑=more important)"]
    for r, i in enumerate(idx_dE, 1):
        rows.append([str(r), _fname(i), _fmt(float(dE[i]), p=4)])
    print("\n### Top by permutation ΔE\n", file=out)
    print(_mk_table(rows, headers), file=out)

    rows = []
    headers = ["rank", "feature", "mean |∂logψ/∂z|"]
    for r, i in enumerate(idx_sal, 1):
        rows.append([str(r), _fname(i), _fmt(float(sal[i]), p=4, exp=True)])
    print("\n### Top by saliency\n", file=out)
    print(_mk_table(rows, headers), file=out)

    overlap = len(set(idx_dE).intersection(idx_sal))
    print(f"\nOverlap of top-{min(top_k,F)} lists: {overlap}/{min(top_k,F)}", file=out)

    # Global curvature
    if "global" in curv and curv["global"]:
        g = curv["global"]
        print("\n## Curvature (A = JᵀJ on residual)", file=out)
        print(f"- λ_max (power-iter est): {g.get('lambda_max_est', 0.0):.6e}", file=out)
        print(f"- iters: {g.get('iters', 0)}   params: {g.get('num_params', 0)}", file=out)
        if g.get("rayleigh_history"):
            hist = ", ".join(f"{v:.3e}" for v in g["rayleigh_history"])
            print(f"- Rayleigh history: [{hist}]", file=out)

    # Block-wise curvature
    if "blocks" in curv and curv["blocks"]:
        print("\n## Block curvatures (A_b = J_bᵀJ_b)", file=out)
        rows = []
        headers = ["block", "P", "λ_max", "λ_min+ (rand)", "κ_est", "eig_mean", "eig_rms"]
        for st in sorted(curv["blocks"], key=lambda z: z["lam_max"], reverse=True):
            rows.append(
                [
                    st["block"],
                    str(st["P"]),
                    _fmt(st["lam_max"]),
                    _fmt(st["lam_min_rand"]),
                    _fmt(st["kappa_est"]),
                    _fmt(st["eig_mean"]),
                    _fmt(st["eig_rms"]),
                ]
            )
        print(_mk_table(rows, headers), file=out)

    text = out.getvalue()
    if to_markdown:
        with open(to_markdown, "w") as f:
            f.write(text)
    return text
