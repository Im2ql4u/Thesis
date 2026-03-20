"""
Neural Orbital Pfaffian with Multi-Channel Expansion
=====================================================
N=6, ω=1.0, E_DMC=20.15932

Diagnosis of old PfaffianNet
-----------------------------
The previous `PfaffianNet` had 243 trainable parameters (three 9×9 matrices),
while backflow had 23,811 and Jastrow had 25,562.  The Pfaffian was a near-HF
projector with zero capacity to change nodal topology — backflow did all the
heavy lifting.

Architecture
------------
Each electron i gets a *neural orbital* vector:

    h_i = [ φ_HO(x_eff_i) ; embed_θ(x_eff_i, s_i) ]   ∈ R^D

where φ_HO is the fixed 9-dim Cartesian HO basis, embed_θ is a learnable
MLP embedding (zero-initialized so we start at HF), and x_eff = x + BF(x)
if a backflow network is present.

For each of K_det Pfaffian channels k = 0, …, K_det−1:

    A^(k)_ij = h_i^T F^(k)_{σ_i,σ_j} h_j      (spin-dependent pairing)
    pf_k = Pf(A^(k))                             (exact via 15 matchings)

Total:

    Ψ = [ Σ_k w_k · Pf(A^(k)) ] × exp(J(x))

Channel 0 is HF-initialized (F_ud = projector P, w_0 = 1).
Channels 1+ start near zero (w_k ≈ 0).

Staged training
---------------
Stage 1 (0 – 25%):  Only Pfaffian params train, BF frozen, α = 0.
Stage 2 (25% – on): BF unfreezes (10× lower LR), α ramps to α_end.

Experiment matrix
-----------------
A: NeurPf(K=1), no BF         →  test Pfaffian capacity alone
B: NeurPf(K=1) + frozen BF    →  test synergy (BF dominant?)
C: NeurPf(K=4) + staged BF    →  full architecture
D: NeurPf(K=4), no BF         →  multi-Pf capacity alone
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

# ─── Constants (identical to run_pfaffian.py) ───
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


# ─── Pfaffian via explicit matchings (N=6) ───
_MATCHINGS_6 = [
    ((0,1),(2,3),(4,5)), ((0,1),(2,4),(3,5)), ((0,1),(2,5),(3,4)),
    ((0,2),(1,3),(4,5)), ((0,2),(1,4),(3,5)), ((0,2),(1,5),(3,4)),
    ((0,3),(1,2),(4,5)), ((0,3),(1,4),(2,5)), ((0,3),(1,5),(2,4)),
    ((0,4),(1,2),(3,5)), ((0,4),(1,3),(2,5)), ((0,4),(1,5),(2,3)),
    ((0,5),(1,2),(3,4)), ((0,5),(1,3),(2,4)), ((0,5),(1,4),(2,3)),
]

def _matching_signs():
    signs = []
    for m in _MATCHINGS_6:
        perm = [i for pair in m for i in pair]
        inv = sum(1 for a in range(len(perm)) for b in range(a+1, len(perm)) if perm[a] > perm[b])
        signs.append((-1)**inv)
    return signs

_SIGNS = _matching_signs()


def pfaffian_6x6(A):
    """Pf of batch of 6×6 antisymmetric matrices.  A: (B,6,6) → (B,)."""
    pf = torch.zeros(A.shape[0], device=A.device, dtype=A.dtype)
    for matching, sign in zip(_MATCHINGS_6, _SIGNS):
        term = torch.ones(A.shape[0], device=A.device, dtype=A.dtype) * sign
        for (i, j) in matching:
            term = term * A[:, i, j]
        pf = pf + term
    return pf


# ═══════════════════════════════════════════════════════════════
#  Neural Orbital Pfaffian
# ═══════════════════════════════════════════════════════════════

class OrbitalEmbedding(nn.Module):
    """
    Learnable single-particle embedding.

    Input:  (x_eff_i ∈ R^d, s_i ∈ {0,1})
    Output: h_learned_i ∈ R^K_emb

    Last layer zero-initialized → initial embedding ≈ 0 → starts at HF.
    """
    def __init__(self, d=2, K_emb=32, hidden=64, layers=2, spin_dim=8):
        super().__init__()
        self.spin_embed = nn.Embedding(2, spin_dim)
        in_dim = d + spin_dim
        mlp = []
        for i in range(layers):
            mlp.append(nn.Linear(in_dim if i == 0 else hidden, hidden))
            mlp.append(nn.SiLU())
        mlp.append(nn.Linear(hidden, K_emb))
        self.mlp = nn.Sequential(*mlp)
        # Zero-init last layer
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x, spin):
        """x: (B,N,d), spin: (B,N) long → (B,N,K_emb)."""
        s_emb = self.spin_embed(spin)                 # (B,N,spin_dim)
        inp = torch.cat([x, s_emb], dim=-1)            # (B,N,d+spin_dim)
        return self.mlp(inp)                            # (B,N,K_emb)


class NeuralPfaffianNet(nn.Module):
    """
    K_det-channel neural orbital Pfaffian.

    Parameters
    ----------
    n_basis : int   — HO basis size (9)
    n_occ   : int   — number of occupied orbitals (3)
    C_occ   : Tensor (n_basis, n_occ) — HF coefficients for initialization
    nx, ny  : int   — HO grid sizes
    K_emb   : int   — learned embedding dimension (default 32)
    K_det   : int   — number of Pfaffian channels (default 4)
    """
    def __init__(self, n_basis, n_occ, C_occ_init, nx, ny, *,
                 K_emb=32, K_det=4, embed_hidden=64, embed_layers=2,
                 spin_dim=8, delta_scale=0.01):
        super().__init__()
        self.n_basis = n_basis
        self.K_emb = K_emb
        self.K_det = K_det
        self.nx = nx
        self.ny = ny
        self.D = n_basis + K_emb   # total orbital dimension
        self.delta_scale = delta_scale

        # ── Learnable embedding ──
        self.orbital_embed = OrbitalEmbedding(
            d=DIM, K_emb=K_emb, hidden=embed_hidden,
            layers=embed_layers, spin_dim=spin_dim,
        )

        # ── Frozen HF projector (channel 0 baseline) ──
        # Stored as buffer so optimizer never touches it.
        P_hf = torch.zeros(self.D, self.D)
        C = C_occ_init.detach().cpu()
        P_hf[:n_basis, :n_basis] = C @ C.T
        self.register_buffer("P_hf", P_hf)

        # ── Learnable ΔF corrections (zero-init → starts at HF) ──
        # Actual F_ud[0] = P_hf + delta_scale * dF_ud[0]
        # For channels 1+: F_ud[k] = delta_scale * dF_ud[k]
        self.dF_ud = nn.ParameterList([nn.Parameter(torch.zeros(self.D, self.D)) for _ in range(K_det)])
        self.dF_uu = nn.ParameterList([nn.Parameter(torch.zeros(self.D, self.D)) for _ in range(K_det)])
        self.dF_dd = nn.ParameterList([nn.Parameter(torch.zeros(self.D, self.D)) for _ in range(K_det)])

        # ── Channel weights ──
        self.w = nn.Parameter(torch.zeros(K_det))

        self._init_weights()

    # ---------- initialization ----------
    def _init_weights(self):
        ss_scale = 1.0  # scaled by delta_scale in forward

        with torch.no_grad():
            # Channel 0: dF starts at zero → F_ud[0] = P_hf exactly
            self.dF_ud[0].zero_()
            # Same-spin: small random to break gradient deadlock
            self.dF_uu[0].normal_(0, ss_scale)
            self.dF_dd[0].normal_(0, ss_scale)

            # Channels 1+: small random, effectively off (w_k = 0)
            for k in range(1, self.K_det):
                self.dF_ud[k].normal_(0, ss_scale)
                self.dF_uu[k].normal_(0, ss_scale)
                self.dF_dd[k].normal_(0, ss_scale)

            # Weights: channel 0 = 1, rest = 0
            self.w.zero_()
            self.w[0] = 1.0

    # ---------- forward ----------
    def forward(self, x, spin=None, bf_net=None):
        """
        x:      (B, N, d) raw electron positions
        spin:   (B, N) long — 0=up 1=down
        bf_net: optional backflow network (frozen or trainable)
        Returns (sign, logabs) each (B,).
        """
        B, N, d = x.shape
        dev = x.device

        if spin is None:
            up = N // 2
            spin = torch.cat([
                torch.zeros(up, dtype=torch.long, device=dev),
                torch.ones(N - up, dtype=torch.long, device=dev),
            ]).unsqueeze(0).expand(B, -1)

        # Coordinate transformation
        x_eff = x + bf_net(x, spin=spin) if bf_net is not None else x

        # ── Single-particle orbitals ──
        Phi_HO = evaluate_basis_functions_torch_batch_2d(x_eff, self.nx, self.ny)  # (B,N,9)
        h_learned = self.orbital_embed(x_eff, spin)                                # (B,N,K_emb)
        h = torch.cat([Phi_HO, h_learned], dim=-1)                                # (B,N,D)

        # ── Spin masks (computed once) ──
        is_up = (spin == 0)
        is_dn = (spin == 1)
        uu = (is_up.unsqueeze(2) & is_up.unsqueeze(1)).float()
        dd = (is_dn.unsqueeze(2) & is_dn.unsqueeze(1)).float()
        ud = (is_up.unsqueeze(2) & is_dn.unsqueeze(1)).float()
        du = (is_dn.unsqueeze(2) & is_up.unsqueeze(1)).float()

        # ── Multi-channel Pfaffian ──
        ds = self.delta_scale
        pf_list = []
        for k in range(self.K_det):
            F_uu_k = ds * (self.dF_uu[k] - self.dF_uu[k].T)       # antisymmetric
            F_dd_k = ds * (self.dF_dd[k] - self.dF_dd[k].T)       # antisymmetric
            # Channel 0: HF projector + small correction
            F_ud_k = (self.P_hf if k == 0 else torch.zeros_like(self.P_hf)) + ds * self.dF_ud[k]

            A_uu = torch.bmm(h @ F_uu_k, h.transpose(1, 2))          # (B,N,N)
            A_dd = torch.bmm(h @ F_dd_k, h.transpose(1, 2))          # (B,N,N)
            A_ud = torch.bmm(h @ F_ud_k, h.transpose(1, 2))          # (B,N,N)

            A = A_uu * uu + A_dd * dd + A_ud * ud - A_ud.transpose(1, 2) * du
            pf_list.append(pfaffian_6x6(A))

        pfs = torch.stack(pf_list, dim=-1)                 # (B, K_det)
        total = (pfs * self.w).sum(dim=-1)                 # (B,)

        sign = torch.sign(total)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        logabs = 0.5 * torch.log(total * total + 1e-60)

        return sign, logabs

    def param_summary(self):
        """Return a formatted parameter count summary."""
        n_embed = sum(p.numel() for p in self.orbital_embed.parameters())
        n_F = sum(p.numel() for p in list(self.dF_ud) + list(self.dF_uu) + list(self.dF_dd))
        n_w = self.w.numel()
        return (
            f"D={self.D} (HO:{self.n_basis}+emb:{self.K_emb}), "
            f"K_det={self.K_det}, delta_scale={self.delta_scale}, "
            f"embed={n_embed:,}, F_matrices={n_F:,}, weights={n_w}, "
            f"total={n_embed + n_F + n_w:,}"
        )


# ═══════════════════════════════════════════════════════════════
#  Wavefunction wrapper
# ═══════════════════════════════════════════════════════════════

def psi_fn_npf(npf_net, f_net, x, params, spin=None, bf_net=None):
    """Ψ = [Σ_k w_k Pf(A^(k))(x_eff)] × exp(J(x))."""
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

    sign, logabs = npf_net(x, spin=spin_bn, bf_net=bf_net)
    f = f_net(x, spin=spin_bn).squeeze(-1)
    logpsi = logabs + f
    psi = sign * torch.exp(logpsi)
    return logpsi, psi


# ═══════════════════════════════════════════════════════════════
#  Training infrastructure
# ═══════════════════════════════════════════════════════════════

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
def screened_colloc(psi_log_fn, n_keep, omega, oversample=8):
    sigma_fs = (0.8, 1.3, 2.0)
    explore = 0.10
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


# ═══════════════════════════════════════════════════════════════
#  Staged training loop
# ═══════════════════════════════════════════════════════════════

def train(
    npf_net, f_net, params,
    *,
    bf_net=None,
    stage_bf=False,
    bf_lr_ratio=0.1,
    embed_lr_ratio=0.1,
    embed_warmup_epochs=50,
    n_epochs=1000,
    lr=5e-4,
    lr_min_frac=0.02,
    alpha_end=0.85,
    n_coll=768,
    oversample=8,
    micro_batch=32,
    grad_clip=0.5,
    replay_frac=0.25,
    qtrim=0.02,
    huber_d=1.0,
    print_every=10,
    patience=250,
    vmc_every=50,
    vmc_n=10000,
    tag="neural_pf",
):
    omega = OMEGA
    E_ref = E_DMC
    up = N_ELEC // 2
    spin = torch.cat(
        [torch.zeros(up, dtype=torch.long), torch.ones(N_ELEC - up, dtype=torch.long)]
    ).to(DEVICE)

    def psi_log_fn(y):
        lp, _ = psi_fn_npf(npf_net, f_net, y, params, spin=spin, bf_net=bf_net)
        return lp

    # ── Optimizer with separate param groups ──
    # Split Pfaffian params: embedding (slow LR) vs F-matrices/weights (fast LR)
    emb_params = list(npf_net.orbital_embed.parameters())
    emb_ids = {id(p) for p in emb_params}
    f_params = [p for p in npf_net.parameters() if id(p) not in emb_ids]
    pf_params = f_params + emb_params  # all Pfaffian params (for grad clipping)
    n_pf = sum(p.numel() for p in pf_params)
    n_jas = sum(p.numel() for p in f_net.parameters())
    n_emb = sum(p.numel() for p in emb_params)
    n_fmat = sum(p.numel() for p in f_params)

    # Freeze embedding during warmup
    emb_frozen = embed_warmup_epochs > 0
    if emb_frozen:
        for p in emb_params:
            p.requires_grad = False

    param_groups = [
        {"params": f_params, "lr": lr},
        {"params": emb_params, "lr": lr * embed_lr_ratio},
    ]

    bf_params = []
    n_bf = 0
    if bf_net is not None and stage_bf:
        bf_params = list(bf_net.parameters())
        n_bf = sum(p.numel() for p in bf_params)
        # Initially frozen — will be unfrozen at stage 2
        for p in bf_params:
            p.requires_grad = False
        param_groups.append({"params": bf_params, "lr": lr * bf_lr_ratio})
    elif bf_net is not None:
        # BF present but frozen for the entire run
        for p in bf_net.parameters():
            p.requires_grad = False
        n_bf = sum(p.numel() for p in bf_net.parameters())

    opt = torch.optim.Adam(param_groups)

    def lr_lambda(ep):
        lr_min = lr * lr_min_frac
        return (lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * ep / max(1, n_epochs - 1)))) / lr

    sch = torch.optim.lr_scheduler.LambdaLR(opt, [lr_lambda] * len(param_groups))
    stage2_start = int(0.25 * n_epochs)

    print(f"  Pfaffian params (trainable): {n_pf:,}  (F-mat: {n_fmat:,}, embed: {n_emb:,})")
    print(f"  Jastrow params (frozen):     {n_jas:,}")
    print(f"  Backflow params:             {n_bf:,}  stage_bf={stage_bf}")
    print(f"  {npf_net.param_summary()}")
    print(f"  Stage 2 (BF unfreeze) at epoch {stage2_start}")
    print(f"  Embed warmup: {embed_warmup_epochs} ep (frozen), then LR={lr * embed_lr_ratio:.1e}")
    print(f"  Training: {n_epochs} ep, {n_coll} colloc pts, LR(F)={lr}, LR(emb)={lr * embed_lr_ratio:.1e}")
    sys.stdout.flush()

    t0 = time.time()
    hist = []
    best_var = best_vmc_err = float("inf")
    best_state = best_vmc_state = {}
    best_vmc_E = None
    no_imp = 0
    replay_X = None
    bf_unfrozen = False

    for ep in range(n_epochs):
        ept0 = time.time()

        # ── Embedding warmup: unfreeze after warmup epochs ──
        if emb_frozen and ep >= embed_warmup_epochs:
            for p in emb_params:
                p.requires_grad = True
            emb_frozen = False
            print(f"  *** Embedding unfrozen at epoch {ep} (LR={lr * embed_lr_ratio:.1e})")
            sys.stdout.flush()

        # ── Staged BF unfreezing ──
        if stage_bf and not bf_unfrozen and ep >= stage2_start and bf_params:
            for p in bf_params:
                p.requires_grad = True
            bf_unfrozen = True
            print(f"  *** Stage 2: BF unfrozen at epoch {ep} (LR={lr * bf_lr_ratio:.1e})")
            sys.stdout.flush()

        # ── Alpha schedule ──
        alpha = (
            0.0 if ep < stage2_start
            else 0.5 * alpha_end * (1 - math.cos(math.pi * (ep - stage2_start) / max(1, n_epochs - stage2_start - 1)))
        )

        # ── Collocation sampling ──
        npf_net.eval()
        f_net.eval()
        if bf_net is not None:
            bf_net.eval()
        X = screened_colloc(psi_log_fn, n_coll, omega, oversample=oversample)

        if replay_frac > 0 and replay_X is not None and replay_X.numel() > 0:
            n_rep = int(min(n_coll - 1, round(replay_frac * n_coll)))
            if n_rep > 0:
                n_new = n_coll - n_rep
                avail = replay_X.shape[0]
                if avail >= n_rep:
                    idx = torch.randperm(avail, device=replay_X.device)[:n_rep]
                    rep = replay_X[idx]
                else:
                    rep = replay_X[torch.randint(0, avail, (n_rep,), device=replay_X.device)]
                X = torch.cat([X[:n_new], rep], dim=0)

        # ── Training step ──
        npf_net.train()
        f_net.eval()
        if bf_net is not None:
            bf_net.train() if bf_unfrozen else bf_net.eval()
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
            all_trainable = pf_params + (bf_params if bf_unfrozen else [])
            nn.utils.clip_grad_norm_(all_trainable, grad_clip)
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
                    replay_X = probe_x[torch.topk(resid_abs, k=topk).indices].detach().clone()
        else:
            Em = Ev = Es = float("nan")

        epdt = time.time() - ept0
        hist.append(dict(ep=ep, E=Em, var=Ev, alpha=alpha, dt=epdt))

        def _save_state():
            st = {"pf_state": {k: v.clone() for k, v in npf_net.state_dict().items()}}
            if bf_net is not None and bf_unfrozen:
                st["bf_state"] = {k: v.clone() for k, v in bf_net.state_dict().items()}
            return st

        if math.isfinite(Ev) and Ev < best_var * 0.999:
            best_var = Ev
            best_state = _save_state()
            no_imp = 0
        else:
            no_imp += 1

        # ── VMC probe ──
        if vmc_every > 0 and ep > 0 and ep % vmc_every == 0:
            def _psi_wrap(f_, x_, C_, *, backflow_net=None, orbital_bf_net=None, spin=None, params=None):
                return psi_fn_npf(npf_net, f_net, x_, params, spin=spin, bf_net=bf_net)
            try:
                C_dummy = torch.eye(9, 3, device=DEVICE, dtype=DTYPE)
                vp = evaluate_energy_vmc(
                    f_net, C_dummy, psi_fn=_psi_wrap,
                    compute_coulomb_interaction=compute_coulomb_interaction,
                    params=params, n_samples=vmc_n, batch_size=512,
                    sampler_steps=40, sampler_step_sigma=0.12, lap_mode="exact",
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

        # ── Early stop ──
        if patience > 0 and no_imp >= patience and ep > 60:
            print(f"  Early stop ep {ep}")
            sys.stdout.flush()
            break

        # ── Logging ──
        if ep % print_every == 0:
            err = (Em - E_ref) / abs(E_ref) * 100 if math.isfinite(Em) else float("nan")
            vs = ""
            if "vmc_E" in hist[-1]:
                vs = f"  vmc={hist[-1]['vmc_E']:.4f}({hist[-1]['vmc_err'] * 100:.2f}%)"
            eta = epdt * (n_epochs - ep - 1) / 60
            w_str = ",".join(f"{wi:.3f}" for wi in npf_net.w.detach().cpu().tolist())
            emb_norm = sum(p.norm().item()**2 for p in npf_net.orbital_embed.parameters())**0.5
            stg = "S1" if ep < stage2_start else "S2"
            print(
                f"  [{ep:4d}] E={Em:.4f}±{Es:.3f} var={Ev:.2e} α={alpha:.2f} "
                f"w=[{w_str}] |emb|={emb_norm:.2f} {stg} {epdt:.1f}s err={err:+.2f}% "
                f"eta={eta:.0f}m{vs}"
            )
            sys.stdout.flush()

    # ── Restore best ──
    if best_vmc_state:
        npf_net.load_state_dict(best_vmc_state["pf_state"])
        if "bf_state" in best_vmc_state and bf_net is not None:
            bf_net.load_state_dict(best_vmc_state["bf_state"])
        print(f"  Restored VMC-best E={best_vmc_E:.5f} err={best_vmc_err * 100:.3f}%")
    elif best_state:
        npf_net.load_state_dict(best_state["pf_state"])
        if "bf_state" in best_state and bf_net is not None:
            bf_net.load_state_dict(best_state["bf_state"])
        print(f"  Restored var-best var={best_var:.3e}")
    tot = time.time() - t0
    print(f"  Done {tot:.0f}s ({tot / 60:.1f}min)")
    sys.stdout.flush()
    return npf_net, bf_net, hist


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Neural Orbital Pfaffian training")
    # Architecture
    ap.add_argument("--K-det", type=int, default=4, help="Number of Pfaffian channels")
    ap.add_argument("--K-emb", type=int, default=32, help="Learned embedding dimension")
    ap.add_argument("--embed-hidden", type=int, default=64, help="Embedding MLP hidden size")
    ap.add_argument("--embed-layers", type=int, default=2, help="Embedding MLP depth")
    ap.add_argument("--spin-dim", type=int, default=8, help="Spin embedding dimension")
    # Backflow control
    ap.add_argument("--use-backflow", action="store_true", help="Load frozen BF from Phase 2 checkpoint")
    ap.add_argument("--stage-bf", action="store_true", help="Unfreeze BF at stage 2 (requires --use-backflow)")
    ap.add_argument("--bf-lr-ratio", type=float, default=0.1, help="BF LR = main LR × ratio")
    ap.add_argument("--embed-lr-ratio", type=float, default=0.1, help="Embed LR = main LR × ratio")
    ap.add_argument("--embed-warmup", type=int, default=50, help="Freeze embedding for first N epochs")
    # Training
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--n-coll", type=int, default=768)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--alpha-end", type=float, default=0.85)
    ap.add_argument("--patience", type=int, default=250)
    # Eval
    ap.add_argument("--vmc-every", type=int, default=50)
    ap.add_argument("--vmc-n", type=int, default=10000)
    ap.add_argument("--n-eval", type=int, default=30000)
    # Misc
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", type=str, default="neural_pf")
    a = ap.parse_args()

    # Reproducibility
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    C_occ, params = setup()

    print(f"{'=' * 64}")
    print(f"  Neural Orbital Pfaffian — {a.tag}")
    print(f"  Device: {DEVICE}   Seed: {a.seed}")
    print(f"  K_det={a.K_det}  K_emb={a.K_emb}  embed_hidden={a.embed_hidden}")
    print(f"  BF={a.use_backflow}  stage_bf={a.stage_bf}  bf_lr_ratio={a.bf_lr_ratio}")
    print(f"  embed_lr_ratio={a.embed_lr_ratio}  embed_warmup={a.embed_warmup}")
    print(f"{'=' * 64}")
    sys.stdout.flush()

    # ── Load Jastrow (always frozen) ──
    bf_net = None
    if a.use_backflow:
        bf_ckpt_path = RESULTS_DIR / "bf_ctnn_vcycle.pt"
        print(f"  Loading BF+Jastrow from {bf_ckpt_path}")
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
        # freeze initially (train() will manage staging)
        for p in bf_net.parameters():
            p.requires_grad = False
        print(f"  Jastrow: {sum(p.numel() for p in f_net.parameters()):,} params (FROZEN)")
        print(f"  Backflow: {sum(p.numel() for p in bf_net.parameters()):,} params")
    else:
        jas_path = RESULTS_DIR / "ctnn_vcycle.pt"
        print(f"  Loading Jastrow from {jas_path}")
        jas_ckpt = torch.load(jas_path, map_location=DEVICE)
        f_net = CTNNJastrowVCycle(
            n_particles=N_ELEC, d=DIM, omega=OMEGA,
            node_hidden=24, edge_hidden=24, bottleneck_hidden=12,
            n_down=1, n_up=1, msg_layers=1, node_layers=1,
            readout_hidden=64, readout_layers=2, act="silu",
        ).to(DEVICE).to(DTYPE)
        f_net.load_state_dict(jas_ckpt["state"])
        for p in f_net.parameters():
            p.requires_grad = False
        print(f"  Jastrow: {sum(p.numel() for p in f_net.parameters()):,} params (FROZEN)")

    # ── Neural Pfaffian ──
    nx = ny = 3
    n_basis = nx * ny
    n_occ = N_ELEC // 2
    npf_net = NeuralPfaffianNet(
        n_basis, n_occ, C_occ, nx, ny,
        K_emb=a.K_emb, K_det=a.K_det,
        embed_hidden=a.embed_hidden, embed_layers=a.embed_layers,
        spin_dim=a.spin_dim,
    ).to(DEVICE).to(DTYPE)
    print(f"  Pfaffian: {npf_net.param_summary()}")
    sys.stdout.flush()

    # ── Train ──
    npf_net, bf_net, hist = train(
        npf_net, f_net, params,
        bf_net=bf_net, stage_bf=a.stage_bf, bf_lr_ratio=a.bf_lr_ratio,
        embed_lr_ratio=a.embed_lr_ratio, embed_warmup_epochs=a.embed_warmup,
        n_epochs=a.epochs, lr=a.lr, alpha_end=a.alpha_end,
        n_coll=a.n_coll, print_every=10, patience=a.patience,
        vmc_every=a.vmc_every, vmc_n=a.vmc_n, tag=a.tag,
    )

    # ── Final heavy VMC eval ──
    E = se = err = wall = float("nan")
    if a.n_eval > 0:
        print(f"\n  Final VMC eval: {a.n_eval} samples ...")
        sys.stdout.flush()

        def _psi_wrap(f_, x_, C_, *, backflow_net=None, orbital_bf_net=None, spin=None, params=None):
            return psi_fn_npf(npf_net, f_net, x_, params, spin=spin, bf_net=bf_net)

        C_dummy = torch.eye(9, 3, device=DEVICE, dtype=DTYPE)
        vmc = evaluate_energy_vmc(
            f_net, C_dummy, psi_fn=_psi_wrap,
            compute_coulomb_interaction=compute_coulomb_interaction,
            params=params, n_samples=a.n_eval, batch_size=512,
            sampler_steps=80, sampler_step_sigma=0.08, lap_mode="exact",
            persistent=True, sampler_burn_in=400, sampler_thin=3, progress=True,
        )
        E = float(vmc["E_mean"])
        se = float(vmc["E_stderr"])
        err = (E - E_DMC) / abs(E_DMC) * 100
        wall = time.time() - (hist[0]["dt"] if hist else 0)
        print(f"\n  *** Final: E = {E:.5f} ± {se:.5f}   err = {err:+.3f}%")

    wall = time.time() - (time.time() - sum(h.get("dt", 0) for h in hist))  # approx

    # ── Save ──
    save_path = RESULTS_DIR / f"{a.tag}.pt"
    ckpt_dict = dict(
        tag=a.tag,
        pf_state=npf_net.state_dict(),
        jas_state=f_net.state_dict(),
        pf_class="NeuralPfaffianNet",
        jas_class="CTNNJastrowVCycle",
        pf_config=dict(
            n_basis=n_basis, n_occ=n_occ, nx=nx, ny=ny,
            K_emb=a.K_emb, K_det=a.K_det,
            embed_hidden=a.embed_hidden, embed_layers=a.embed_layers,
            spin_dim=a.spin_dim,
        ),
        E=E, se=se, err=err, hist=hist,
        seed=a.seed,
    )
    if bf_net is not None:
        ckpt_dict["bf_state"] = bf_net.state_dict()
        ckpt_dict["bf_config"] = {
            k: v for k, v in torch.load(RESULTS_DIR / "bf_ctnn_vcycle.pt", map_location="cpu")["bf_config"].items()
        }
    torch.save(ckpt_dict, save_path)
    print(f"  Saved → {save_path}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
