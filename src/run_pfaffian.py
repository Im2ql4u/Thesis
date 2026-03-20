"""
Pfaffian wavefunction for collocation training
===============================================
N=6, ω=1.0, E_DMC=20.15932

Replaces the det(up)×det(down) Slater product with a Pfaffian Pf(A)
of a 6×6 antisymmetric orbital matrix, allowing richer nodal topology.

The pairing matrix is spin-dependent:
  A_ij = Φ(r_i)^T F_{σ_i σ_j} Φ(r_j) + δA_ij(neural net)

where F^{↑↓} is initialized from the HF singlet pairing (reproducing
the Slater product), F^{↑↑} and F^{↓↓} start at zero (triplet channels).

A small MLP learns corrections δA_ij from inter-electron features,
giving the Pfaffian enough capacity to change nodal topology.

Combined wavefunction: Ψ = Pf(A) × exp(J(x))
where J is a frozen pre-trained Jastrow factor.
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from functions.Energy import evaluate_energy_vmc
from functions.Neural_Networks import _laplacian_logpsi_exact
from functions.Physics import compute_coulomb_interaction
from functions.Slater_Determinant import evaluate_basis_functions_torch_batch_2d
from jastrow_architectures import CTNNJastrowVCycle
from PINN import CTNNBackflowNet

_manual = os.environ.get("CUDA_MANUAL_DEVICE")
if _manual is not None and torch.cuda.is_available():
    DEVICE = f"cuda:{_manual}" if _manual.isdigit() else _manual
elif torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"
DTYPE = torch.float64
N_ELEC = 6
DIM = 2
OMEGA = 1.0
E_DMC = 20.15932

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "arch_colloc"


def setup():
    N, d = N_ELEC, DIM
    n_occ = N // 2
    nx = ny = 3
    L = max(8.0, 3.0 / math.sqrt(OMEGA))
    config.update(
        omega=OMEGA, n_particles=N, d=d, L=L, n_grid=80,
        nx=nx, ny=ny, basis="cart", device=DEVICE, dtype="float64",
    )
    energies = sorted([(OMEGA * (ix + iy + 1), ix, iy) for ix in range(nx) for iy in range(ny)])
    C = np.zeros((nx * ny, n_occ))
    for k in range(n_occ):
        _, ix, iy = energies[k]
        C[ix * ny + iy, k] = 1.0
    C_occ = torch.tensor(C, dtype=DTYPE, device=DEVICE)
    p = config.get().as_dict()
    p.update(device=DEVICE, torch_dtype=DTYPE, E=E_DMC)
    return C_occ, p


# ─── Pfaffian computation for 6×6 antisymmetric matrices ───
# All 15 perfect matchings of {0,1,2,3,4,5}
_MATCHINGS_6 = [
    ((0,1),(2,3),(4,5)),
    ((0,1),(2,4),(3,5)),
    ((0,1),(2,5),(3,4)),
    ((0,2),(1,3),(4,5)),
    ((0,2),(1,4),(3,5)),
    ((0,2),(1,5),(3,4)),
    ((0,3),(1,2),(4,5)),
    ((0,3),(1,4),(2,5)),
    ((0,3),(1,5),(2,4)),
    ((0,4),(1,2),(3,5)),
    ((0,4),(1,3),(2,5)),
    ((0,4),(1,5),(2,3)),
    ((0,5),(1,2),(3,4)),
    ((0,5),(1,3),(2,4)),
    ((0,5),(1,4),(2,3)),
]

def _compute_matching_signs():
    signs = []
    for matching in _MATCHINGS_6:
        perm = []
        for (i, j) in matching:
            perm.extend([i, j])
        inv = 0
        for a in range(len(perm)):
            for b in range(a+1, len(perm)):
                if perm[a] > perm[b]:
                    inv += 1
        signs.append((-1)**inv)
    return signs

_MATCHING_SIGNS = _compute_matching_signs()


def pfaffian_6x6(A):
    """
    Compute Pfaffian of batch of 6×6 antisymmetric matrices.
    A: (B, 6, 6). Returns: (B,).
    """
    pf = torch.zeros(A.shape[0], device=A.device, dtype=A.dtype)
    for matching, sign in zip(_MATCHINGS_6, _MATCHING_SIGNS):
        term = torch.ones(A.shape[0], device=A.device, dtype=A.dtype) * sign
        for (i, j) in matching:
            term = term * A[:, i, j]
        pf = pf + term
    return pf


class PairCorrectionMLP(nn.Module):
    """
    Small MLP that takes pair features (r²_ij, dx, dy, same_spin) and outputs
    a scalar correction δA_ij to the orbital pairing matrix.
    Uses r² instead of r to avoid gradient singularity at r=0.
    """
    def __init__(self, d=2, hidden=32, layers=2):
        super().__init__()
        # Input: dx, dy, r², same_spin (d+2 features)
        in_dim = d + 2
        mlp = []
        for i in range(layers):
            mlp.append(nn.Linear(in_dim if i == 0 else hidden, hidden))
            mlp.append(nn.SiLU())
        mlp.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*mlp)
        # Zero-init last layer so correction starts at 0
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.scale = nn.Parameter(torch.tensor(0.01))
    
    def forward(self, x, spin):
        """
        x: (B, N, d), spin: (B, N) long
        Returns: dA (B, N, N) antisymmetric correction
        """
        B, N, d = x.shape
        # Pair displacements
        dx = x.unsqueeze(2) - x.unsqueeze(1)  # (B,N,N,d)
        r2 = (dx * dx).sum(dim=-1, keepdim=True)  # (B,N,N,1) — smooth everywhere
        same = (spin.unsqueeze(2) == spin.unsqueeze(1)).float().unsqueeze(-1)  # (B,N,N,1)
        feat = torch.cat([dx, r2, same], dim=-1)  # (B,N,N,d+2)
        
        out = self.net(feat).squeeze(-1)  # (B,N,N)
        # Antisymmetrize
        dA = self.scale * (out - out.transpose(1, 2))
        return dA


class PfaffianNet(nn.Module):
    """
    Spin-dependent Pfaffian: Ψ_Pf = Pf(A(x))
    
    A_ij = Φ(r_i)^T F_{σ_i,σ_j} Φ(r_j) + δA_ij
    
    Three spin channels:
      F_ud: opposite-spin pairing (initialized from HF)
      F_uu: same-spin up-up (initialized to 0)
      F_dd: same-spin down-down (initialized to 0)
    
    δA: learned pair correction from PairCorrectionMLP.
    """
    def __init__(self, n_basis, n_occ, C_occ_init, nx, ny, correction_hidden=32, correction_layers=2, use_mlp=True):
        super().__init__()
        self.n_basis = n_basis
        self.n_occ = n_occ
        self.nx = nx
        self.ny = ny
        self.use_mlp = use_mlp
        
        # Spin-channel pairing matrices (upper triangular → antisymmetrized)
        self.F_ud_raw = nn.Parameter(torch.zeros(n_basis, n_basis))  # opposite-spin
        self.F_uu_raw = nn.Parameter(torch.zeros(n_basis, n_basis))  # up-up
        self.F_dd_raw = nn.Parameter(torch.zeros(n_basis, n_basis))  # down-down
        
        # Neural correction (optional)
        if use_mlp:
            self.pair_mlp = PairCorrectionMLP(d=DIM, hidden=correction_hidden, layers=correction_layers)
        else:
            self.pair_mlp = None
        
        self._init_from_slater(C_occ_init)
    
    def _init_from_slater(self, C_occ):
        """
        Initialize F_ud so that the Pfaffian reproduces the HF Slater product.
        
        det(Φ_up C) × det(Φ_down C) = Pf(A) where
        A_{i↑,j↓} = Σ_k [C^T Φ(r_i)]_k [C^T Φ(r_j)]_k = Φ(r_i)^T (C C^T) Φ(r_j)
        
        So F_ud = C_occ @ C_occ^T (the occupied-space projector, which is symmetric).
        For the antisymmetric A: A_{i,j} = Φ_i^T F_ud Φ_j for i<j (up-down),
        and A_{j,i} = -A_{i,j}.
        
        Since C_occ @ C_occ^T is symmetric, A_{i↑,j↓} = Φ_i^T P Φ_j and
        A_{j↓,i↑} = -A_{i↑,j↓}, which gives the correct antisymmetry.
        """
        C = C_occ.detach().cpu()  # (n_basis, n_occ)
        P = C @ C.T  # symmetric projector
        
        # F_ud_raw: we set it directly; antisymmetrization will be applied
        # but P is symmetric. We need F_ud to be such that 
        # (F_ud_raw - F_ud_raw^T) gives the right structure.
        # Actually for opposite-spin, we DON'T antisymmetrize F itself — 
        # the antisymmetry comes from the (i↑,j↓) vs (j↓,i↑) structure.
        # So F_ud should be the full P matrix, not antisymmetrized.
        # Let's store F_ud_raw = P directly and use it as-is for opposite-spin.
        with torch.no_grad():
            self.F_ud_raw.copy_(P)
            # Same-spin: small random init to break gradient deadlock.
            # (With F_uu=F_dd=0, any matching pairing two same-spin-up
            #  electrons also pairs two same-spin-down, so ∂Pf/∂F_ss = 0.)
            ss_scale = 0.01
            self.F_uu_raw.normal_(0, ss_scale)
            self.F_dd_raw.normal_(0, ss_scale)
    
    def forward(self, x, spin=None, bf_net=None):
        """
        x: (B, N, d) electron positions
        spin: (B, N) long tensor (0=up, 1=down)
        bf_net: optional backflow network; if given, evaluates basis at x_eff = x + bf(x)
        Returns: (sign, logabs) both (B,)
        """
        B, N, d = x.shape
        dev = x.device
        
        if spin is None:
            up = N // 2
            spin = torch.cat([
                torch.zeros(up, dtype=torch.long, device=dev),
                torch.ones(N - up, dtype=torch.long, device=dev),
            ]).unsqueeze(0).expand(B, -1)
        
        # Optionally apply backflow transformation
        if bf_net is not None:
            x_eff = x + bf_net(x, spin=spin)
        else:
            x_eff = x
        
        # Evaluate basis functions: (B, N, n_basis)
        Phi = evaluate_basis_functions_torch_batch_2d(x_eff, self.nx, self.ny)
        
        # Build spin-dependent A matrix
        # For each pair (i,j), select the appropriate F based on spins
        F_uu = self.F_uu_raw - self.F_uu_raw.T  # antisymmetric
        F_dd = self.F_dd_raw - self.F_dd_raw.T  # antisymmetric
        F_ud = self.F_ud_raw  # NOT antisymmetrized (symmetry handled by pair assignment)
        
        # Compute Φ_i^T F Φ_j for each spin channel
        # PhiF_uu = Phi @ F_uu, etc.
        A_uu = torch.bmm(Phi @ F_uu, Phi.transpose(1, 2))  # (B,N,N)
        A_dd = torch.bmm(Phi @ F_dd, Phi.transpose(1, 2))  # (B,N,N)
        A_ud = torch.bmm(Phi @ F_ud, Phi.transpose(1, 2))  # (B,N,N)
        
        # Spin masks: (B,N,N)
        s = spin  # (B,N)
        is_up = (s == 0)  # (B,N) bool
        is_dn = (s == 1)  # (B,N) bool
        
        uu_mask = (is_up.unsqueeze(2) & is_up.unsqueeze(1)).float()
        dd_mask = (is_dn.unsqueeze(2) & is_dn.unsqueeze(1)).float()
        ud_mask = (is_up.unsqueeze(2) & is_dn.unsqueeze(1)).float()
        du_mask = (is_dn.unsqueeze(2) & is_up.unsqueeze(1)).float()
        
        # Combine: A_ij = A_uu if both up, A_dd if both down,
        #          A_ud if i=up j=down, -A_ud^T if i=down j=up
        A = A_uu * uu_mask + A_dd * dd_mask + A_ud * ud_mask - A_ud.transpose(1,2) * du_mask
        
        # Add neural correction (if enabled)
        if self.pair_mlp is not None:
            dA = self.pair_mlp(x, spin)  # (B,N,N) antisymmetric
            A = A + dA
        
        # Compute Pfaffian
        pf = pfaffian_6x6(A)  # (B,)
        
        sign = torch.sign(pf)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        # Use log(pf^2)/2 instead of log(|pf|) for smooth second derivatives
        logabs = 0.5 * torch.log(pf * pf + 1e-60)
        
        return sign, logabs


# ─── psi_fn for Pfaffian + frozen Jastrow ───
def psi_fn_pfaffian(pfaffian_net, f_net, x, params, spin=None, bf_net=None):
    """
    Ψ = Pf(A(x_eff)) × exp(J(x))
    If bf_net is given, x_eff = x + bf(x), otherwise x_eff = x.
    Returns (logpsi, psi) where logpsi = log|Pf| + J
    """
    B, N, d = x.shape
    dev = x.device
    if spin is None:
        up = N // 2
        spin_bn = torch.cat([
            torch.zeros(up, dtype=torch.long, device=dev),
            torch.ones(N - up, dtype=torch.long, device=dev),
        ]).unsqueeze(0).expand(B, -1)
    else:
        s = spin.to(dev).long()
        spin_bn = s.unsqueeze(0).expand(B, -1) if s.dim() == 1 else s
    
    sign, logabs = pfaffian_net(x, spin=spin_bn, bf_net=bf_net)  # (B,)
    f = f_net(x, spin=spin_bn).squeeze(-1)  # (B,) — Jastrow
    
    logpsi = logabs + f
    psi = sign * torch.exp(logpsi)
    return logpsi, psi


# ─── Standard training infrastructure ───
def compute_EL(psi_log_fn, x, omega):
    x = x.detach().requires_grad_(True)
    _, g2, lap = _laplacian_logpsi_exact(psi_log_fn, x)
    B = x.shape[0]
    T = -0.5 * (lap.view(B) + g2.view(B))
    V = 0.5 * omega**2 * (x**2).sum(dim=(1, 2)) + compute_coulomb_interaction(x).view(B)
    return T + V


def huber(r, d):
    a = r.abs()
    return torch.where(a <= d, 0.5 * r**2, d * (a - 0.5 * d))


def sample_gauss(n, omega, sigma_f=1.3):
    s = sigma_f / math.sqrt(omega)
    x = torch.randn(n, N_ELEC, DIM, device=DEVICE, dtype=DTYPE) * s
    Nd = N_ELEC * DIM
    lq = -0.5 * Nd * math.log(2 * math.pi * s**2) - x.reshape(n, -1).pow(2).sum(-1) / (2 * s**2)
    return x, lq


@torch.no_grad()
def screened_colloc(psi_log_fn, n_keep, omega, oversample=8, sigma_fs=(0.8, 1.3, 2.0), explore=0.10):
    n_cand = oversample * n_keep
    nc = len(sigma_fs)
    xs, lqs = [], []
    for i, sf in enumerate(sigma_fs):
        ni = n_cand // nc if i < nc - 1 else n_cand - sum(n_cand // nc for _ in range(i))
        xi, lqi = sample_gauss(ni, omega, sf)
        xs.append(xi); lqs.append(lqi)
    x_all = torch.cat(xs)
    lq_all = torch.cat(lqs)
    lp2 = []
    for i in range(0, len(x_all), 4096):
        lp2.append(2.0 * psi_log_fn(x_all[i : i + 4096]))
    lr = torch.cat(lp2) - lq_all
    n_exp = int(max(0, min(n_keep - 1, round(explore * n_keep))))
    n_top = n_keep - n_exp
    _, idx = torch.sort(lr, descending=True)
    sel = idx[:n_top]
    if n_exp > 0 and idx[n_top:].numel() > 0:
        rest = idx[n_top:]
        sel = torch.cat([sel, rest[torch.randperm(len(rest))[:n_exp]]])
    return x_all[sel[:n_keep]].clone()


def train_pfaffian(
    pfaffian_net, f_net, params,
    *, n_epochs=500, lr=5e-4, lr_min_frac=0.02,
    alpha_end=0.70, n_coll=512, oversample=8,
    micro_batch=32, grad_clip=0.5, replay_frac=0.25,
    qtrim=0.02, huber_d=1.0, print_every=10, patience=120,
    vmc_every=40, vmc_n=6000, tag="pfaffian", bf_net=None,
):
    omega = OMEGA
    E_ref = E_DMC
    up = N_ELEC // 2
    spin = torch.cat(
        [torch.zeros(up, dtype=torch.long), torch.ones(N_ELEC - up, dtype=torch.long)]
    ).to(DEVICE)

    def psi_log_fn(y):
        lp, _ = psi_fn_pfaffian(pfaffian_net, f_net, y, params, spin=spin, bf_net=bf_net)
        return lp

    # Only Pfaffian params are trainable (Jastrow frozen)
    pf_params = [p for p in pfaffian_net.parameters() if p.requires_grad]
    n_pf = sum(p.numel() for p in pf_params)
    n_jas = sum(p.numel() for p in f_net.parameters())
    print(f"  Pfaffian params (trainable): {n_pf:,}")
    print(f"  Jastrow params (frozen):     {n_jas:,}")

    opt = torch.optim.Adam(pf_params, lr=lr)

    def lr_lambda(ep):
        lr_min = lr * lr_min_frac
        return (lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * ep / max(1, n_epochs - 1)))) / lr

    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    p1end = int(0.25 * n_epochs)

    print(f"  Training: {n_epochs} ep, {n_coll} colloc pts, LR={lr}")
    sys.stdout.flush()

    t0 = time.time()
    hist = []
    best_var = best_vmc_err = float("inf")
    best_state = best_vmc_state = {}
    best_vmc_E = None
    no_imp = 0
    replay_X = None

    for ep in range(n_epochs):
        ept0 = time.time()
        alpha = (
            0.0 if ep < p1end
            else 0.5 * alpha_end * (1 - math.cos(math.pi * (ep - p1end) / max(1, n_epochs - p1end - 1)))
        )

        pfaffian_net.eval()
        f_net.eval()
        X = screened_colloc(psi_log_fn, n_coll, omega, oversample=oversample)

        if replay_frac > 0 and replay_X is not None and replay_X.numel() > 0:
            n_rep = int(min(n_coll - 1, round(replay_frac * n_coll)))
            if n_rep > 0:
                n_new = n_coll - n_rep
                if replay_X.shape[0] >= n_rep:
                    idx = torch.randperm(replay_X.shape[0], device=replay_X.device)[:n_rep]
                    rep = replay_X[idx]
                else:
                    rep_idx = torch.randint(0, replay_X.shape[0], (n_rep,), device=replay_X.device)
                    rep = replay_X[rep_idx]
                X = torch.cat([X[:n_new], rep], dim=0)

        pfaffian_net.train()
        f_net.eval()  # Jastrow stays frozen
        opt.zero_grad(set_to_none=True)
        all_EL = []
        nmb = max(1, math.ceil(n_coll / micro_batch))

        for i in range(0, n_coll, micro_batch):
            xb = X[i : i + micro_batch]
            EL = compute_EL(psi_log_fn, xb, omega).view(-1)
            ok = torch.isfinite(EL)
            if not ok.all():
                EL = EL[ok]
            if EL.numel() == 0:
                continue
            if qtrim > 0 and EL.numel() > 20:
                lo = torch.quantile(EL.detach(), qtrim)
                hi = torch.quantile(EL.detach(), 1.0 - qtrim)
                m = (EL.detach() >= lo) & (EL.detach() <= hi)
                EL = EL[m]
                if EL.numel() == 0:
                    continue
            all_EL.append(EL.detach())
            mu = EL.mean().detach()
            E_eff = alpha * E_ref + (1 - alpha) * mu if alpha > 0 else mu
            loss = huber(EL - E_eff, huber_d).mean()
            (loss / nmb).backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(pf_params, grad_clip)
        opt.step()
        sch.step()

        if all_EL:
            ELc = torch.cat(all_EL)
            Em, Ev, Es = ELc.mean().item(), ELc.var().item(), ELc.std().item()
            # Replay buffer
            probe_n = min(128, X.shape[0])
            probe_x = X[:probe_n]
            probe_EL = compute_EL(psi_log_fn, probe_x, omega).view(-1)
            if torch.isfinite(probe_EL).any():
                keep = torch.isfinite(probe_EL)
                probe_x = probe_x[keep]
                probe_EL = probe_EL[keep]
                if probe_x.shape[0] > 4:
                    resid_abs = (probe_EL - probe_EL.mean()).abs()
                    topk = min(max(8, probe_x.shape[0] // 4), probe_x.shape[0])
                    idx_top = torch.topk(resid_abs, k=topk).indices
                    replay_X = probe_x[idx_top].detach().clone()
        else:
            Em = Ev = Es = float("nan")

        epdt = time.time() - ept0
        hist.append(dict(ep=ep, E=Em, var=Ev, alpha=alpha, dt=epdt))

        def _save_state():
            return {"pf_state": {k: v.clone() for k, v in pfaffian_net.state_dict().items()}}

        if math.isfinite(Ev) and Ev < best_var * 0.999:
            best_var = Ev
            best_state = _save_state()
            no_imp = 0
        else:
            no_imp += 1

        # VMC probe — wrap in standard psi_fn interface for evaluate_energy_vmc
        if vmc_every > 0 and ep > 0 and ep % vmc_every == 0:
            # We need a psi_fn compatible with evaluate_energy_vmc
            # It expects psi_fn(f_net, x, C_occ, backflow_net=..., spin=..., params=...)
            # We'll create a wrapper
            class _PfWrapper(nn.Module):
                """Wraps pfaffian_net+f_net to look like a standard f_net for VMC eval."""
                def __init__(self, pf_net, jas_net, params):
                    super().__init__()
                    self.pf_net = pf_net
                    self.jas_net = jas_net
                    self.params = params
                def forward(self, x, spin=None):
                    return self.jas_net(x, spin=spin)

            def _psi_fn_wrap(f_net_wrap, x, C_occ, *, backflow_net=None, orbital_bf_net=None, spin=None, params=None):
                lp, ps = psi_fn_pfaffian(pfaffian_net, f_net, x, params, spin=spin, bf_net=bf_net)
                return lp, ps

            try:
                # Dummy C_occ for interface compat
                C_dummy = torch.eye(9, 3, device=DEVICE, dtype=DTYPE)
                vp = evaluate_energy_vmc(
                    f_net, C_dummy,
                    psi_fn=_psi_fn_wrap,
                    compute_coulomb_interaction=compute_coulomb_interaction,
                    params=params,
                    n_samples=vmc_n, batch_size=512, sampler_steps=40,
                    sampler_step_sigma=0.12, lap_mode="exact",
                    persistent=True, sampler_burn_in=200, sampler_thin=2, progress=False,
                )
                vE = float(vp["E_mean"])
                vErr = abs(vE - E_ref) / abs(E_ref)
                hist[-1].update(vmc_E=vE, vmc_err=vErr)
                if vErr < best_vmc_err:
                    best_vmc_err = vErr
                    best_vmc_E = vE
                    best_vmc_state = _save_state()
            except Exception as e:
                print(f"  VMC probe failed: {e}")

        if patience > 0 and no_imp >= patience and ep > 60:
            print(f"  Early stop ep {ep}")
            sys.stdout.flush()
            break

        if ep % print_every == 0:
            err = (Em - E_ref) / abs(E_ref) * 100 if math.isfinite(Em) else float("nan")
            vs = ""
            if "vmc_E" in hist[-1]:
                vs = f"  vmc={hist[-1]['vmc_E']:.4f}({hist[-1]['vmc_err'] * 100:.2f}%)"
            eta = epdt * (n_epochs - ep - 1) / 60
            # Show F matrix norms
            f_ud_norm = pfaffian_net.F_ud_raw.norm().item()
            f_ss_norm = (pfaffian_net.F_uu_raw.norm().item() + pfaffian_net.F_dd_raw.norm().item())
            print(
                f"  [{ep:3d}] E={Em:.4f}±{Es:.3f} var={Ev:.2e} α={alpha:.2f} "
                f"|Fud|={f_ud_norm:.3f} |Fss|={f_ss_norm:.3f} {epdt:.1f}s err={err:+.2f}% eta={eta:.0f}m{vs}"
            )
            sys.stdout.flush()

    # Restore best
    if best_vmc_state:
        pfaffian_net.load_state_dict(best_vmc_state["pf_state"])
        print(f"  Restored VMC-best E={best_vmc_E:.5f} err={best_vmc_err * 100:.3f}%")
    elif best_state:
        pfaffian_net.load_state_dict(best_state["pf_state"])
        print(f"  Restored var-best var={best_var:.3e}")
    tot = time.time() - t0
    print(f"  Done {tot:.0f}s ({tot / 60:.1f}min)")
    sys.stdout.flush()
    return pfaffian_net, hist


def main():
    ap = argparse.ArgumentParser(description="Pfaffian wavefunction collocation training")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--n-coll", type=int, default=512)
    ap.add_argument("--n-eval", type=int, default=10000)
    ap.add_argument("--patience", type=int, default=120)
    ap.add_argument("--vmc-every", type=int, default=40)
    ap.add_argument("--vmc-n", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--alpha-end", type=float, default=0.70)
    ap.add_argument("--no-mlp", action="store_true", help="Disable pair correction MLP")
    ap.add_argument("--use-backflow", action="store_true", help="Use frozen backflow transformation from BF+Jastrow checkpoint")
    ap.add_argument("--tag", type=str, default="pfaffian")
    a = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    C_occ, params = setup()

    # ─── Load pre-trained Jastrow ───
    bf_net = None
    if a.use_backflow:
        # Load from BF+Jastrow checkpoint (Phase 2)
        bf_ckpt_path = RESULTS_DIR / "bf_ctnn_vcycle.pt"
        print(f"Loading BF+Jastrow from {bf_ckpt_path}")
        ckpt = torch.load(bf_ckpt_path, map_location=DEVICE)
        f_net = CTNNJastrowVCycle(
            n_particles=N_ELEC, d=DIM, omega=OMEGA,
            node_hidden=24, edge_hidden=24, bottleneck_hidden=12,
            n_down=1, n_up=1, msg_layers=1, node_layers=1,
            readout_hidden=64, readout_layers=2, act="silu",
        ).to(DEVICE).to(DTYPE)
        f_net.load_state_dict(ckpt["jas_state"])
        for p in f_net.parameters():
            p.requires_grad = False
        n_jas = sum(p.numel() for p in f_net.parameters())
        print(f"  Jastrow: CTNNJastrowVCycle  {n_jas:,} params  (FROZEN)")
        # Load backflow
        bfc = ckpt["bf_config"]
        bf_net = CTNNBackflowNet(
            d=bfc["d"], msg_hidden=bfc["msg_hidden"], msg_layers=bfc["msg_layers"],
            hidden=bfc["hidden"], layers=bfc["layers"], act=bfc["act"],
            aggregation=bfc["aggregation"], use_spin=bfc["use_spin"],
            same_spin_only=bfc["same_spin_only"], out_bound=bfc["out_bound"],
            bf_scale_init=bfc["bf_scale_init"], zero_init_last=bfc["zero_init_last"],
            omega=bfc["omega"],
        ).to(DEVICE).to(DTYPE)
        bf_net.load_state_dict(ckpt["bf_state"])
        for p in bf_net.parameters():
            p.requires_grad = False
        n_bf = sum(p.numel() for p in bf_net.parameters())
        print(f"  Backflow: CTNNBackflowNet  {n_bf:,} params  (FROZEN)")
        print(f"  Checkpoint energy: E={ckpt.get('E', '?')}")
    else:
        # Load Jastrow only (from Phase 1)
        jas_ckpt_path = RESULTS_DIR / "ctnn_vcycle.pt"
        print(f"Loading Jastrow from {jas_ckpt_path}")
        jas_ckpt = torch.load(jas_ckpt_path, map_location=DEVICE)
        f_net = CTNNJastrowVCycle(
            n_particles=N_ELEC, d=DIM, omega=OMEGA,
            node_hidden=24, edge_hidden=24, bottleneck_hidden=12,
            n_down=1, n_up=1, msg_layers=1, node_layers=1,
            readout_hidden=64, readout_layers=2, act="silu",
        ).to(DEVICE).to(DTYPE)
        f_net.load_state_dict(jas_ckpt["state"])
        for p in f_net.parameters():
            p.requires_grad = False
        n_jas = sum(p.numel() for p in f_net.parameters())
        print(f"  Jastrow: CTNNJastrowVCycle  {n_jas:,} params  (FROZEN)")
        print(f"  Jastrow E={jas_ckpt.get('E', '?')}")

    # ─── PfaffianNet ───
    nx = ny = 3
    n_basis = nx * ny  # 9
    n_occ = N_ELEC // 2  # 3
    pfaffian_net = PfaffianNet(
        n_basis, n_occ, C_occ, nx, ny,
        correction_hidden=32, correction_layers=2,
        use_mlp=not a.no_mlp,
    ).to(DEVICE).to(DTYPE)
    n_pf = sum(p.numel() for p in pfaffian_net.parameters())
    print(f"  Pfaffian: PfaffianNet  {n_pf:,} params  (MLP={'ON' if not a.no_mlp else 'OFF'})")
    print(f"    F_ud (opposite-spin): {n_basis}×{n_basis} = {n_basis**2} params, initialized from HF")
    print(f"    F_uu, F_dd (same-spin): {n_basis}×{n_basis} each, random init (scale=0.01)")
    if pfaffian_net.pair_mlp is not None:
        print(f"    PairCorrectionMLP: {sum(p.numel() for p in pfaffian_net.pair_mlp.parameters()):,} params")
    sys.stdout.flush()

    # ─── Train ───
    print(f"\n{'#' * 60}")
    print("# Pfaffian wavefunction collocation training (Jastrow frozen)")
    print(f"# {a.epochs} epochs, {a.n_coll} colloc pts, α_end={a.alpha_end}")
    print(f"# LR={a.lr}")
    print(f"{'#' * 60}\n")
    sys.stdout.flush()

    t0 = time.time()
    pfaffian_net, hist = train_pfaffian(
        pfaffian_net, f_net, params,
        n_epochs=a.epochs, lr=a.lr, alpha_end=a.alpha_end,
        n_coll=a.n_coll, oversample=8, micro_batch=32,
        print_every=10, replay_frac=0.25,
        patience=a.patience, vmc_every=a.vmc_every, vmc_n=a.vmc_n,
        tag=a.tag, bf_net=bf_net,
    )

    # ─── Final heavy VMC evaluation ───
    if a.n_eval > 0:
        print(f"\n  Final VMC eval: {a.n_eval} samples ...")
        sys.stdout.flush()

        def _psi_fn_wrap(f_net_wrap, x, C_occ, *, backflow_net=None, orbital_bf_net=None, spin=None, params=None):
            lp, ps = psi_fn_pfaffian(pfaffian_net, f_net, x, params, spin=spin, bf_net=bf_net)
            return lp, ps

        C_dummy = torch.eye(9, 3, device=DEVICE, dtype=DTYPE)
        vmc = evaluate_energy_vmc(
            f_net, C_dummy,
            psi_fn=_psi_fn_wrap,
            compute_coulomb_interaction=compute_coulomb_interaction,
            params=params,
            n_samples=a.n_eval, batch_size=512,
            sampler_steps=80, sampler_step_sigma=0.08,
            lap_mode="exact", persistent=True,
            sampler_burn_in=400, sampler_thin=3, progress=True,
        )
        E = float(vmc["E_mean"])
        se = float(vmc["E_stderr"])
        err = (E - E_DMC) / abs(E_DMC) * 100
        wall = time.time() - t0
        print(f"\n  *** Final: E = {E:.5f} ± {se:.5f}   err = {err:+.3f}%  ({wall/60:.1f} min)")
    else:
        E = se = err = wall = float("nan")

    # ─── Save checkpoint ───
    save_path = RESULTS_DIR / f"{a.tag}.pt"
    save_dict = dict(
        tag=a.tag,
        pf_state=pfaffian_net.state_dict(),
        jas_state=f_net.state_dict(),
        pf_class="PfaffianNet",
        jas_class="CTNNJastrowVCycle",
        n_pf_params=n_pf,
        n_jas_params=n_jas,
        pf_config=dict(n_basis=n_basis, n_occ=n_occ, nx=nx, ny=ny),
        E=E, se=se, err=err, hist=hist, wall=wall,
    )
    if bf_net is not None:
        bf_ckpt = torch.load(RESULTS_DIR / "bf_ctnn_vcycle.pt", map_location="cpu")
        save_dict["bf_state"] = bf_net.state_dict()
        save_dict["bf_config"] = bf_ckpt["bf_config"]
    torch.save(save_dict, save_path)
    print(f"  Saved → {save_path}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
