"""
Alternative Jastrow factor architectures for SD × exp(J).

All follow the same interface as PINN:
    forward(x, spin=None) -> (B, 1)

where x is (B, N, d) particle positions, and the output is the scalar
log Jastrow factor.  Analytic cusps are always included.

Architectures:
  1. PINNLarge — scaled-up version of current PINN (wider, deeper, multi-pool)
  2. CTNNJastrow — message-passing GNN backbone → scalar via pooled readout
  3. DeepSetJastrow — proper DeepSet over all pairs with multi-head pooling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ═════════════════════════════════════════════════════════════════
#  Common: analytic cusps (shared by all architectures)
# ═════════════════════════════════════════════════════════════════


class CuspMixin:
    """Mixin providing analytic electron-electron cusp terms."""

    def _init_cusps(self, n_particles, d, omega):
        self._n_particles_cusp = n_particles
        self._d_cusp = d
        self._omega_cusp = omega
        self.gamma_apara = 1.0 / (d - 1)  # opposite-spin
        self.gamma_para = 1.0 / (d + 1)  # same-spin
        self.cusp_len = 1.0 / (omega**0.5)
        # pair indices
        ii, jj = torch.triu_indices(n_particles, n_particles, offset=1)
        self.register_buffer("_cusp_idx_i", ii, persistent=False)
        self.register_buffer("_cusp_idx_j", jj, persistent=False)

    def _compute_cusps(self, x, spin=None):
        """u_cusp(x) = sum_{i<j} gamma_ij * r_ij * exp(-r_ij / ell)."""
        B, N, d = x.shape
        diff = x[:, self._cusp_idx_i, :] - x[:, self._cusp_idx_j, :]  # (B,P,d)
        r = torch.sqrt((diff**2).sum(-1, keepdim=True) + 1e-30)  # (B,P,1)

        if spin is not None:
            sp = spin.to(x.device).long()
            if sp.ndim == 1:
                sp = sp.unsqueeze(0).expand(B, -1)
            si = sp[:, self._cusp_idx_i]
            sj = sp[:, self._cusp_idx_j]
            same = (si == sj).to(x.dtype).unsqueeze(-1)
        else:
            up = N // 2
            sp = torch.cat(
                [
                    torch.zeros(up, dtype=torch.long, device=x.device),
                    torch.ones(N - up, dtype=torch.long, device=x.device),
                ]
            )
            si = sp[self._cusp_idx_i].unsqueeze(0).expand(B, -1)
            sj = sp[self._cusp_idx_j].unsqueeze(0).expand(B, -1)
            same = (si == sj).to(x.dtype).unsqueeze(-1)

        gamma = same * self.gamma_para + (1.0 - same) * self.gamma_apara
        return (gamma * r * torch.exp(-r)).sum(dim=1)  # (B, 1)


# ═════════════════════════════════════════════════════════════════
#  Architecture 1: PINNLarge
#  Scaled-up PINN with multi-head pooling and deeper readout
# ═════════════════════════════════════════════════════════════════


class PINNLarge(nn.Module, CuspMixin):
    """
    Like PINN but with:
    - Wider descriptor (dL=32 default)
    - Deeper MLPs (n_layers=4)
    - Multi-head pooling: [sum, mean, max] for both φ and ψ branches
    - Deeper readout (3-layer MLP) from the richer pooled vector
    - ~60-100K params
    """

    def __init__(
        self,
        n_particles: int,
        d: int,
        omega: float,
        *,
        dL: int = 32,
        hidden_dim: int = 128,
        n_layers: int = 4,
        readout_hidden: int = 128,
        readout_layers: int = 3,
        act: str = "gelu",
        use_gate: bool = True,
        gate_radius_aho: float = 0.30,
        eps_feat_aho: float = 0.20,
    ):
        super().__init__()
        self._init_cusps(n_particles, d, omega)
        self.n_particles = n_particles
        self.d = d
        self.omega = omega
        self.dL = dL
        self.use_gate = use_gate
        self.gate_radius_aho = gate_radius_aho
        self.eps_feat_aho = eps_feat_aho

        ii, jj = torch.triu_indices(n_particles, n_particles, offset=1)
        self.register_buffer("idx_i", ii, persistent=False)
        self.register_buffer("idx_j", jj, persistent=False)

        _act = self._make_act(act)

        # φ branch: per-particle embeddings
        self.phi = self._build_mlp(d, hidden_dim, dL, n_layers, _act)

        # ψ branch: pair features (6 safe features)
        self.psi = self._build_mlp(6, hidden_dim, dL, n_layers, _act)

        # Multi-head pooling: sum + mean + max for each branch
        # φ: 3 × dL,  ψ: 3 × dL,  extras: 2
        readout_in = 3 * dL + 3 * dL + 2
        self.readout = self._build_mlp(readout_in, readout_hidden, 1, readout_layers, _act)

        # Small-random init so J starts near 0 but gradients flow
        with torch.no_grad():
            nn.init.normal_(self.readout[-1].weight, std=1e-3)
            self.readout[-1].bias.zero_()

        self._initialize_weights()

    def _make_act(self, name):
        return {
            "gelu": nn.GELU,
            "relu": nn.ReLU,
            "silu": nn.SiLU,
            "tanh": nn.Tanh,
            "mish": nn.Mish,
        }[name.lower()]()

    def _build_mlp(self, in_dim, hidden, out_dim, n_layers, act):
        layers = [nn.Linear(in_dim, hidden), act]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), act]
        layers.append(nn.Linear(hidden, out_dim))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def _safe_pair_features(self, r):
        """6 safe pair features (ds/dr → 0 as r → 0)."""
        a_ho = 1.0 / (self.omega**0.5)
        eps = self.eps_feat_aho * a_ho
        r2 = r * r
        rt = torch.sqrt(r2 + eps * eps)
        s1 = torch.log1p((rt / eps) ** 2)
        s2 = r2 / (r2 + eps * eps)
        s3 = (rt / eps) ** 2 * torch.exp(-((rt / eps) ** 2))
        g = torch.as_tensor([0.25, 1.0, 4.0], device=r.device, dtype=r.dtype).view(1, 1, -1)
        rbf = torch.exp(-g * s1)
        return torch.cat([s1, s2, s3, rbf], dim=-1), s1.mean(dim=1, keepdim=False)

    def _short_range_gate(self, r):
        if not self.use_gate:
            return torch.ones_like(r)
        a_ho = 1.0 / (self.omega**0.5)
        rg = self.gate_radius_aho * a_ho
        return (r * r) / (r * r + rg * rg)

    def forward(self, x, spin=None):
        B, N, d = x.shape
        x_sc = x * (self.omega**0.5)

        # Pair geometry
        diff = x_sc.unsqueeze(2) - x_sc.unsqueeze(1)
        diff_pairs = diff[:, self.idx_i, self.idx_j, :]
        r2 = (diff_pairs**2).sum(-1, keepdim=True)
        r = torch.sqrt(r2 + torch.finfo(x.dtype).eps)

        # φ branch: per-particle
        phi_out = self.phi(x_sc.reshape(B * N, d)).reshape(B, N, self.dL)
        phi_sum = phi_out.sum(dim=1)
        phi_mean = phi_out.mean(dim=1)
        phi_max = phi_out.max(dim=1).values

        # ψ branch: pair features
        psi_in, s1_mean = self._safe_pair_features(r)
        P = psi_in.shape[1]
        psi_out = self.psi(psi_in.reshape(B * P, 6)).reshape(B, P, self.dL)
        gate = self._short_range_gate(r)
        psi_out = psi_out * gate
        psi_sum = psi_out.sum(dim=1)
        psi_mean = psi_out.mean(dim=1)
        psi_max = psi_out.max(dim=1).values

        # Extras
        r2_mean = (x_sc**2).mean(dim=(1, 2), keepdim=False).unsqueeze(-1)
        extras = torch.cat([r2_mean, s1_mean], dim=1)

        # Readout: multi-head pooled features
        rho_in = torch.cat([phi_sum, phi_mean, phi_max, psi_sum, psi_mean, psi_max, extras], dim=1)
        f_nn = self.readout(rho_in)

        return f_nn + self._compute_cusps(x, spin)


# ═════════════════════════════════════════════════════════════════
#  Architecture 2: CTNNJastrow
#  Message-passing GNN backbone → scalar Jastrow
# ═════════════════════════════════════════════════════════════════


class CTNNJastrow(nn.Module, CuspMixin):
    """
    Graph neural network Jastrow factor with message passing.

    Unlike the PINN which mean-pools pair features independently,
    this architecture lets pair features communicate via node ↔ edge
    message passing, capturing three-body and higher correlations.

    Architecture:
    1. Node embedding:  (x_i, spin_i) → h_i
    2. Edge embedding:  (r_ij, |r_ij|, |r_ij|²) → h_e
    3. K rounds of message passing:
       - Node → Edge via learned transport
       - Edge update (residual)
       - Edge → Node via learned transport + aggregation
       - Node update (residual)
    4. Readout: pool node + edge features → MLP → scalar

    Key advantage: after K MP steps, each pair feature h_e(i,j)
    has information about the *environment* of electrons i and j,
    enabling effective three-body and higher correlations.
    """

    def __init__(
        self,
        n_particles: int,
        d: int,
        omega: float,
        *,
        node_hidden: int = 64,
        edge_hidden: int = 64,
        n_mp_steps: int = 2,
        msg_layers: int = 2,
        node_layers: int = 2,
        readout_hidden: int = 64,
        readout_layers: int = 2,
        act: str = "silu",
        aggregation: str = "sum",
        use_spin: bool = True,
    ):
        super().__init__()
        self._init_cusps(n_particles, d, omega)
        self.n_particles = n_particles
        self.d = d
        self.omega = omega
        self.n_mp_steps = n_mp_steps
        self.use_spin = use_spin
        self.aggregation = aggregation

        def make_act():
            return {
                "silu": nn.SiLU,
                "gelu": nn.GELU,
                "relu": nn.ReLU,
                "tanh": nn.Tanh,
                "mish": nn.Mish,
            }[act.lower()]()

        # Node embedding
        node_in = d + (1 if use_spin else 0)
        self.node_embed = nn.Linear(node_in, node_hidden)

        # Edge embedding
        edge_in = d + 2  # [r_vec, |r|, |r|²]
        self.edge_embed = self._mlp(edge_in, edge_hidden, edge_hidden, msg_layers, make_act)

        # Per-step MP modules
        self.rho_v_to_e = nn.ModuleList()
        self.edge_updates = nn.ModuleList()
        self.rho_e_to_v = nn.ModuleList()
        self.node_updates = nn.ModuleList()

        for _ in range(n_mp_steps):
            self.rho_v_to_e.append(nn.Linear(node_hidden, edge_hidden, bias=False))
            self.edge_updates.append(
                self._mlp(3 * edge_hidden, edge_hidden, edge_hidden, msg_layers, make_act)
            )
            self.rho_e_to_v.append(nn.Linear(edge_hidden, node_hidden, bias=False))
            self.node_updates.append(
                self._mlp(2 * node_hidden, node_hidden, node_hidden, node_layers, make_act)
            )

        # Readout: pool node features + edge features + extras → scalar
        # [node_sum, node_mean] + [edge_sum, edge_mean] + extras(2)
        readout_in = 2 * node_hidden + 2 * edge_hidden + 2
        layers_f = []
        dim = readout_in
        for _ in range(readout_layers):
            layers_f += [nn.Linear(dim, readout_hidden), make_act()]
            dim = readout_hidden
        layers_f.append(nn.Linear(dim, 1))
        # Small-random init so J starts near 0 but gradients flow
        nn.init.normal_(layers_f[-1].weight, std=1e-3)
        nn.init.zeros_(layers_f[-1].bias)
        self.f_head = nn.Sequential(*layers_f)

        # Pair indices for upper triangle
        ii, jj = torch.triu_indices(n_particles, n_particles, offset=1)
        self.register_buffer("idx_i", ii, persistent=False)
        self.register_buffer("idx_j", jj, persistent=False)

    def _mlp(self, in_dim, hid, out_dim, n_layers, act_fn):
        layers = []
        if n_layers == 1:
            layers.append(nn.Linear(in_dim, out_dim))
        else:
            layers.append(nn.Linear(in_dim, hid))
            layers.append(act_fn())
            for _ in range(n_layers - 2):
                layers.append(nn.Linear(hid, hid))
                layers.append(act_fn())
            layers.append(nn.Linear(hid, out_dim))
        return nn.Sequential(*layers)

    def _aggregate(self, msgs):
        """msgs: (B,N,N,H) → (B,N,H)"""
        if self.aggregation == "sum":
            return msgs.sum(dim=2)
        if self.aggregation == "mean":
            return msgs.mean(dim=2)
        return msgs.max(dim=2).values

    def _resolve_spin(self, spin, B, N, device):
        """Handle spin as (N,) or (B, N) → returns 1D (N,) and 2D (B, N)."""
        if spin is None:
            up = N // 2
            spin_1d = torch.cat(
                [
                    torch.zeros(up, dtype=torch.long, device=device),
                    torch.ones(N - up, dtype=torch.long, device=device),
                ]
            )
        else:
            s = spin.to(device).long()
            spin_1d = s[0] if s.dim() == 2 else s
        spin_2d = spin_1d.unsqueeze(0).expand(B, N)
        return spin_1d, spin_2d

    def forward(self, x, spin=None):
        B, N, d = x.shape
        x_sc = x * (self.omega**0.5)

        # Spin
        spin_1d, spin_2d = self._resolve_spin(spin, B, N, x.device)

        # Node input
        if self.use_spin:
            sf = spin_1d.view(1, N, 1).to(x.dtype).expand(B, N, 1)
            node_in = torch.cat([x_sc, sf], dim=-1)
        else:
            node_in = x_sc

        h_v = self.node_embed(node_in)  # (B, N, node_hidden)

        # Edge geometry (dense N×N)
        r_vec = x_sc.unsqueeze(2) - x_sc.unsqueeze(1)  # (B,N,N,d)
        r2 = (r_vec**2).sum(-1, keepdim=True)
        r1 = torch.sqrt(r2 + 1e-12)
        edge_in = torch.cat([r_vec, r1, r2], dim=-1)
        h_e = self.edge_embed(edge_in)  # (B,N,N,edge_hidden)

        # Edge weight mask (no self-loops)
        eye = torch.eye(N, device=x.device, dtype=x.dtype).view(1, N, N, 1)
        weight = 1.0 - eye

        # Message passing
        for step in range(self.n_mp_steps):
            h_v_i = h_v.unsqueeze(2).expand(B, N, N, h_v.shape[-1])
            h_v_j = h_v.unsqueeze(1).expand(B, N, N, h_v.shape[-1])

            v_i_to_e = self.rho_v_to_e[step](h_v_i)
            v_j_to_e = self.rho_v_to_e[step](h_v_j)

            edge_upd_in = torch.cat([h_e, v_i_to_e, v_j_to_e], dim=-1)
            h_e = h_e + self.edge_updates[step](edge_upd_in)  # residual

            msgs = self.rho_e_to_v[step](h_e) * weight
            m_v = self._aggregate(msgs)

            node_upd_in = torch.cat([h_v, m_v], dim=-1)
            h_v = h_v + self.node_updates[step](node_upd_in)  # residual

        # Readout
        h_v_sum = h_v.sum(dim=1)  # (B, node_hidden)
        h_v_mean = h_v.mean(dim=1)  # (B, node_hidden)

        # Edge features: upper-triangle pairs only
        h_e_pairs = h_e[:, self.idx_i, self.idx_j, :]  # (B, P, edge_hidden)
        h_e_sum = h_e_pairs.sum(dim=1)
        h_e_mean = h_e_pairs.mean(dim=1)

        # Extras
        r2_mean = (x_sc**2).mean(dim=(1, 2)).unsqueeze(-1)
        eps = 0.20 / (self.omega**0.5)
        r_pairs_phys = torch.sqrt(
            (x.unsqueeze(2) - x.unsqueeze(1))[:, self.idx_i, self.idx_j, :]
            .pow(2)
            .sum(-1, keepdim=True)
            + eps**2
        )
        s1_mean = torch.log1p((r_pairs_phys / eps) ** 2).mean(dim=1)

        f_in = torch.cat([h_v_sum, h_v_mean, h_e_sum, h_e_mean, r2_mean, s1_mean], dim=1)
        f_nn = self.f_head(f_in)

        return f_nn + self._compute_cusps(x, spin_1d)


# ═════════════════════════════════════════════════════════════════
#  Architecture 3: DeepSetJastrow
#  Proper DeepSets over pairs with multi-head pooling
# ═════════════════════════════════════════════════════════════════


class DeepSetJastrow(nn.Module, CuspMixin):
    """
    DeepSet Jastrow with pair-level encoder and multi-head pooling.

    Each pair (i,j) gets encoded through a deep MLP using rich features:
    - Safe pair distance features (6 dims, same as PINN)
    - Relative position vector (d dims)
    - Center-of-pair position features (d dims)
    - Spin match indicator (1 dim)

    Pooling uses sum + mean + learned attention weights.

    This is richer than the PINN because:
    1. More input features per pair (including relative orientation)
    2. Attention-weighted pooling preserves more structure than mean
    3. Deeper pair encoder
    """

    def __init__(
        self,
        n_particles: int,
        d: int,
        omega: float,
        *,
        pair_hidden: int = 128,
        pair_layers: int = 4,
        pair_out: int = 32,  # per-pair embedding dimension
        readout_hidden: int = 128,
        readout_layers: int = 3,
        act: str = "gelu",
        use_gate: bool = True,
        gate_radius_aho: float = 0.30,
        eps_feat_aho: float = 0.20,
    ):
        super().__init__()
        self._init_cusps(n_particles, d, omega)
        self.n_particles = n_particles
        self.d = d
        self.omega = omega
        self.pair_out = pair_out
        self.use_gate = use_gate
        self.gate_radius_aho = gate_radius_aho
        self.eps_feat_aho = eps_feat_aho

        ii, jj = torch.triu_indices(n_particles, n_particles, offset=1)
        self.register_buffer("idx_i", ii, persistent=False)
        self.register_buffer("idx_j", jj, persistent=False)

        _act = {
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "mish": nn.Mish,
        }[act.lower()]()

        # Pair features: 6 (safe) + d (rel vec) + d (center) + 1 (spin) = 6 + 2d + 1
        pair_in = 6 + 2 * d + 1
        self.pair_encoder = self._build_mlp(pair_in, pair_hidden, pair_out, pair_layers, _act)

        # Attention head for weighted pooling
        self.attn_head = nn.Sequential(
            nn.Linear(pair_out, pair_out // 2),
            nn.Tanh(),
            nn.Linear(pair_out // 2, 1),
        )

        # Single-particle encoder
        sp_in = d + 1  # position + spin
        self.sp_encoder = self._build_mlp(sp_in, pair_hidden // 2, pair_out // 2, 2, _act)

        # Readout: [pair_sum, pair_mean, pair_attn, sp_sum, sp_mean, extras(2)]
        readout_in = 3 * pair_out + 2 * (pair_out // 2) + 2
        self.readout = self._build_mlp(readout_in, readout_hidden, 1, readout_layers, _act)

        # Small-random init so J starts near 0 but gradients flow
        with torch.no_grad():
            nn.init.normal_(self.readout[-1].weight, std=1e-3)
            self.readout[-1].bias.zero_()

        self._init_weights()

    def _build_mlp(self, in_dim, hidden, out_dim, n_layers, act):
        layers = [nn.Linear(in_dim, hidden), act]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), act]
        layers.append(nn.Linear(hidden, out_dim))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _safe_pair_features(self, r):
        a_ho = 1.0 / (self.omega**0.5)
        eps = self.eps_feat_aho * a_ho
        r2 = r * r
        rt = torch.sqrt(r2 + eps * eps)
        s1 = torch.log1p((rt / eps) ** 2)
        s2 = r2 / (r2 + eps * eps)
        s3 = (rt / eps) ** 2 * torch.exp(-((rt / eps) ** 2))
        g = torch.as_tensor([0.25, 1.0, 4.0], device=r.device, dtype=r.dtype).view(1, 1, -1)
        rbf = torch.exp(-g * s1)
        return torch.cat([s1, s2, s3, rbf], dim=-1), s1.mean(dim=1, keepdim=False)

    def _short_range_gate(self, r):
        if not self.use_gate:
            return torch.ones_like(r)
        a_ho = 1.0 / (self.omega**0.5)
        rg = self.gate_radius_aho * a_ho
        return (r * r) / (r * r + rg * rg)

    def _resolve_spin(self, spin, B, N, device):
        if spin is None:
            up = N // 2
            return torch.cat(
                [
                    torch.zeros(up, dtype=torch.long, device=device),
                    torch.ones(N - up, dtype=torch.long, device=device),
                ]
            )
        s = spin.to(device).long()
        return s[0] if s.dim() == 2 else s

    def forward(self, x, spin=None):
        B, N, d = x.shape
        x_sc = x * (self.omega**0.5)

        # Spin — always 1D (N,)
        spin_vec = self._resolve_spin(spin, B, N, x.device)

        # Pair features
        diff_sc = x_sc[:, self.idx_i, :] - x_sc[:, self.idx_j, :]  # (B,P,d)
        center_sc = 0.5 * (x_sc[:, self.idx_i, :] + x_sc[:, self.idx_j, :])  # (B,P,d)
        r2 = (diff_sc**2).sum(-1, keepdim=True)
        r = torch.sqrt(r2 + torch.finfo(x.dtype).eps)

        safe_feat, s1_mean = self._safe_pair_features(r)  # (B,P,6), (B,1)
        gate = self._short_range_gate(r)

        # Spin match for each pair
        si = spin_vec[self.idx_i]
        sj = spin_vec[self.idx_j]
        spin_match = (si == sj).to(x.dtype).unsqueeze(0).unsqueeze(-1).expand(B, -1, 1)

        # Full pair input: [safe(6), rel_vec(d), center(d), spin_match(1)]
        pair_in = torch.cat([safe_feat, diff_sc, center_sc, spin_match], dim=-1)

        P = pair_in.shape[1]
        pair_emb = self.pair_encoder(pair_in.reshape(B * P, -1)).reshape(B, P, self.pair_out)
        pair_emb = pair_emb * gate  # gate near coalescence

        # Multi-head pooling
        pair_sum = pair_emb.sum(dim=1)
        pair_mean = pair_emb.mean(dim=1)

        # Attention-weighted pool
        attn_logits = self.attn_head(pair_emb)  # (B, P, 1)
        attn_weights = F.softmax(attn_logits, dim=1)
        pair_attn = (pair_emb * attn_weights).sum(dim=1)

        # Single-particle features
        sp_spin = spin_vec.to(x.dtype).unsqueeze(0).unsqueeze(-1).expand(B, N, 1)
        sp_in = torch.cat([x_sc, sp_spin], dim=-1)
        sp_emb = self.sp_encoder(sp_in.reshape(B * N, -1)).reshape(B, N, -1)
        sp_sum = sp_emb.sum(dim=1)
        sp_mean = sp_emb.mean(dim=1)

        # Extras
        r2_mean = (x_sc**2).mean(dim=(1, 2)).unsqueeze(-1)
        extras = torch.cat([r2_mean, s1_mean], dim=1)

        # Readout
        rho_in = torch.cat([pair_sum, pair_mean, pair_attn, sp_sum, sp_mean, extras], dim=1)
        f_nn = self.readout(rho_in)

        return f_nn + self._compute_cusps(x, spin_vec)


class CTNNJastrowAttnGlobal(nn.Module, CuspMixin):
    """
    CTNN variant with stronger global structure:
    - Message passing identical to CTNNJastrow
    - Additional global node context injected into edge updates
    - Edge readout uses sum + mean + attention pooling
    """

    def __init__(
        self,
        n_particles: int,
        d: int,
        omega: float,
        *,
        node_hidden: int = 32,
        edge_hidden: int = 32,
        n_mp_steps: int = 3,
        msg_layers: int = 2,
        node_layers: int = 2,
        readout_hidden: int = 64,
        readout_layers: int = 2,
        act: str = "silu",
        aggregation: str = "sum",
        use_spin: bool = True,
    ):
        super().__init__()
        self._init_cusps(n_particles, d, omega)
        self.n_particles = n_particles
        self.d = d
        self.omega = omega
        self.n_mp_steps = n_mp_steps
        self.use_spin = use_spin
        self.aggregation = aggregation

        def make_act():
            return {
                "silu": nn.SiLU,
                "gelu": nn.GELU,
                "relu": nn.ReLU,
                "tanh": nn.Tanh,
                "mish": nn.Mish,
            }[act.lower()]()

        node_in = d + (1 if use_spin else 0)
        self.node_embed = nn.Linear(node_in, node_hidden)

        edge_in = d + 2  # [r_vec, |r|, |r|²]
        self.edge_embed = self._mlp(edge_in, edge_hidden, edge_hidden, msg_layers, make_act)

        self.rho_v_to_e = nn.ModuleList()
        self.edge_updates = nn.ModuleList()
        self.rho_e_to_v = nn.ModuleList()
        self.node_updates = nn.ModuleList()
        self.global_to_edge = nn.ModuleList()

        for _ in range(n_mp_steps):
            self.rho_v_to_e.append(nn.Linear(node_hidden, edge_hidden, bias=False))
            self.global_to_edge.append(nn.Linear(node_hidden, edge_hidden, bias=False))
            self.edge_updates.append(
                self._mlp(4 * edge_hidden, edge_hidden, edge_hidden, msg_layers, make_act)
            )
            self.rho_e_to_v.append(nn.Linear(edge_hidden, node_hidden, bias=False))
            self.node_updates.append(
                self._mlp(2 * node_hidden, node_hidden, node_hidden, node_layers, make_act)
            )

        self.edge_attn = nn.Sequential(
            nn.Linear(edge_hidden, edge_hidden // 2),
            make_act(),
            nn.Linear(edge_hidden // 2, 1),
        )

        readout_in = 2 * node_hidden + 3 * edge_hidden + 2
        layers_f = []
        dim = readout_in
        for _ in range(readout_layers):
            layers_f += [nn.Linear(dim, readout_hidden), make_act()]
            dim = readout_hidden
        layers_f.append(nn.Linear(dim, 1))
        nn.init.normal_(layers_f[-1].weight, std=1e-3)
        nn.init.zeros_(layers_f[-1].bias)
        self.f_head = nn.Sequential(*layers_f)

        ii, jj = torch.triu_indices(n_particles, n_particles, offset=1)
        self.register_buffer("idx_i", ii, persistent=False)
        self.register_buffer("idx_j", jj, persistent=False)

    def _mlp(self, in_dim, hid, out_dim, n_layers, act_fn):
        layers = []
        if n_layers == 1:
            layers.append(nn.Linear(in_dim, out_dim))
        else:
            layers.append(nn.Linear(in_dim, hid))
            layers.append(act_fn())
            for _ in range(n_layers - 2):
                layers.append(nn.Linear(hid, hid))
                layers.append(act_fn())
            layers.append(nn.Linear(hid, out_dim))
        return nn.Sequential(*layers)

    def _aggregate(self, msgs):
        if self.aggregation == "sum":
            return msgs.sum(dim=2)
        if self.aggregation == "mean":
            return msgs.mean(dim=2)
        return msgs.max(dim=2).values

    def _resolve_spin(self, spin, B, N, device):
        if spin is None:
            up = N // 2
            spin_1d = torch.cat(
                [
                    torch.zeros(up, dtype=torch.long, device=device),
                    torch.ones(N - up, dtype=torch.long, device=device),
                ]
            )
        else:
            s = spin.to(device).long()
            spin_1d = s[0] if s.dim() == 2 else s
        return spin_1d

    def forward(self, x, spin=None):
        B, N, d = x.shape
        x_sc = x * (self.omega**0.5)
        spin_1d = self._resolve_spin(spin, B, N, x.device)

        if self.use_spin:
            sf = spin_1d.view(1, N, 1).to(x.dtype).expand(B, N, 1)
            node_in = torch.cat([x_sc, sf], dim=-1)
        else:
            node_in = x_sc

        h_v = self.node_embed(node_in)

        r_vec = x_sc.unsqueeze(2) - x_sc.unsqueeze(1)
        r2 = (r_vec**2).sum(-1, keepdim=True)
        r1 = torch.sqrt(r2 + 1e-12)
        edge_in = torch.cat([r_vec, r1, r2], dim=-1)
        h_e = self.edge_embed(edge_in)

        eye = torch.eye(N, device=x.device, dtype=x.dtype).view(1, N, N, 1)
        weight = 1.0 - eye

        for step in range(self.n_mp_steps):
            h_v_i = h_v.unsqueeze(2).expand(B, N, N, h_v.shape[-1])
            h_v_j = h_v.unsqueeze(1).expand(B, N, N, h_v.shape[-1])

            v_i_to_e = self.rho_v_to_e[step](h_v_i)
            v_j_to_e = self.rho_v_to_e[step](h_v_j)
            g_v = h_v.mean(dim=1, keepdim=True)
            g_to_e = self.global_to_edge[step](g_v).unsqueeze(2).expand(B, N, N, -1)

            edge_upd_in = torch.cat([h_e, v_i_to_e, v_j_to_e, g_to_e], dim=-1)
            h_e = h_e + self.edge_updates[step](edge_upd_in)

            msgs = self.rho_e_to_v[step](h_e) * weight
            m_v = self._aggregate(msgs)
            node_upd_in = torch.cat([h_v, m_v], dim=-1)
            h_v = h_v + self.node_updates[step](node_upd_in)

        h_v_sum = h_v.sum(dim=1)
        h_v_mean = h_v.mean(dim=1)

        h_e_pairs = h_e[:, self.idx_i, self.idx_j, :]
        h_e_sum = h_e_pairs.sum(dim=1)
        h_e_mean = h_e_pairs.mean(dim=1)

        attn_logits = self.edge_attn(h_e_pairs)
        attn_w = F.softmax(attn_logits, dim=1)
        h_e_attn = (h_e_pairs * attn_w).sum(dim=1)

        r2_mean = (x_sc**2).mean(dim=(1, 2)).unsqueeze(-1)
        eps = 0.20 / (self.omega**0.5)
        r_pairs_phys = torch.sqrt(
            (x.unsqueeze(2) - x.unsqueeze(1))[:, self.idx_i, self.idx_j, :]
            .pow(2)
            .sum(-1, keepdim=True)
            + eps**2
        )
        s1_mean = torch.log1p((r_pairs_phys / eps) ** 2).mean(dim=1)

        f_in = torch.cat(
            [
                h_v_sum,
                h_v_mean,
                h_e_sum,
                h_e_mean,
                h_e_attn,
                r2_mean,
                s1_mean,
            ],
            dim=1,
        )
        f_nn = self.f_head(f_in)

        return f_nn + self._compute_cusps(x, spin_1d)


class TriadicDeepSetJastrow(nn.Module, CuspMixin):
    """
    DeepSet with explicit triadic pair context.

    For each pair (i,j), includes a learned summary over all k≠i,j:
      t_ij = mean_k φ([r_ik, r_jk, |r_ik-r_jk|]).

    This injects explicit three-body structure before pooling.
    """

    def __init__(
        self,
        n_particles: int,
        d: int,
        omega: float,
        *,
        pair_hidden: int = 64,
        pair_layers: int = 3,
        pair_out: int = 24,
        triad_hidden: int = 32,
        readout_hidden: int = 64,
        readout_layers: int = 2,
        act: str = "gelu",
        eps_feat_aho: float = 0.20,
    ):
        super().__init__()
        self._init_cusps(n_particles, d, omega)
        self.n_particles = n_particles
        self.d = d
        self.omega = omega
        self.pair_out = pair_out
        self.eps_feat_aho = eps_feat_aho

        ii, jj = torch.triu_indices(n_particles, n_particles, offset=1)
        self.register_buffer("idx_i", ii, persistent=False)
        self.register_buffer("idx_j", jj, persistent=False)

        _act = {
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "mish": nn.Mish,
        }[act.lower()]()

        self.triad_encoder = self._build_mlp(3, triad_hidden, triad_hidden, 2, _act)

        pair_in = 6 + 2 * d + 1 + triad_hidden
        self.pair_encoder = self._build_mlp(pair_in, pair_hidden, pair_out, pair_layers, _act)

        self.attn_head = nn.Sequential(
            nn.Linear(pair_out, pair_out // 2),
            nn.Tanh(),
            nn.Linear(pair_out // 2, 1),
        )

        self.sp_encoder = self._build_mlp(d + 1, pair_hidden, pair_out // 2, 2, _act)
        readout_in = 3 * pair_out + 2 * (pair_out // 2) + 2
        self.readout = self._build_mlp(readout_in, readout_hidden, 1, readout_layers, _act)
        with torch.no_grad():
            nn.init.normal_(self.readout[-1].weight, std=1e-3)
            self.readout[-1].bias.zero_()

        self._init_weights()

    def _build_mlp(self, in_dim, hidden, out_dim, n_layers, act):
        layers = [nn.Linear(in_dim, hidden), act]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), act]
        layers.append(nn.Linear(hidden, out_dim))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _safe_pair_features(self, r):
        a_ho = 1.0 / (self.omega**0.5)
        eps = self.eps_feat_aho * a_ho
        r2 = r * r
        rt = torch.sqrt(r2 + eps * eps)
        s1 = torch.log1p((rt / eps) ** 2)
        s2 = r2 / (r2 + eps * eps)
        s3 = (rt / eps) ** 2 * torch.exp(-((rt / eps) ** 2))
        g = torch.as_tensor([0.25, 1.0, 4.0], device=r.device, dtype=r.dtype).view(1, 1, -1)
        rbf = torch.exp(-g * s1)
        return torch.cat([s1, s2, s3, rbf], dim=-1), s1.mean(dim=1, keepdim=False)

    def _resolve_spin(self, spin, B, N, device):
        if spin is None:
            up = N // 2
            return torch.cat(
                [
                    torch.zeros(up, dtype=torch.long, device=device),
                    torch.ones(N - up, dtype=torch.long, device=device),
                ]
            )
        s = spin.to(device).long()
        return s[0] if s.dim() == 2 else s

    def _triadic_context(self, x_sc):
        """Return per-pair triadic context (B,P,triad_hidden)."""
        B, N, _ = x_sc.shape
        diff = x_sc.unsqueeze(2) - x_sc.unsqueeze(1)
        r = torch.sqrt((diff**2).sum(-1) + 1e-12)  # (B,N,N)
        i_idx, j_idx = self.idx_i, self.idx_j
        P = i_idx.numel()

        triad_feats = []
        for p in range(P):
            i = int(i_idx[p].item())
            j = int(j_idx[p].item())
            k_mask = [k for k in range(N) if k != i and k != j]
            rik = r[:, i, k_mask]
            rjk = r[:, j, k_mask]
            d = (rik - rjk).abs()
            triad_in = torch.stack([rik, rjk, d], dim=-1)  # (B,K,3)
            K = triad_in.shape[1]
            triad_enc = self.triad_encoder(triad_in.reshape(B * K, 3)).reshape(B, K, -1)
            triad_feats.append(triad_enc.mean(dim=1))
        return torch.stack(triad_feats, dim=1)  # (B,P,triad_hidden)

    def forward(self, x, spin=None):
        B, N, d = x.shape
        x_sc = x * (self.omega**0.5)
        spin_vec = self._resolve_spin(spin, B, N, x.device)

        diff_sc = x_sc[:, self.idx_i, :] - x_sc[:, self.idx_j, :]
        center_sc = 0.5 * (x_sc[:, self.idx_i, :] + x_sc[:, self.idx_j, :])
        r2 = (diff_sc**2).sum(-1, keepdim=True)
        r = torch.sqrt(r2 + torch.finfo(x.dtype).eps)

        safe_feat, s1_mean = self._safe_pair_features(r)

        si = spin_vec[self.idx_i]
        sj = spin_vec[self.idx_j]
        spin_match = (si == sj).to(x.dtype).unsqueeze(0).unsqueeze(-1).expand(B, -1, 1)

        triad = self._triadic_context(x_sc)
        pair_in = torch.cat([safe_feat, diff_sc, center_sc, spin_match, triad], dim=-1)

        P = pair_in.shape[1]
        pair_emb = self.pair_encoder(pair_in.reshape(B * P, -1)).reshape(B, P, self.pair_out)
        pair_sum = pair_emb.sum(dim=1)
        pair_mean = pair_emb.mean(dim=1)
        attn_logits = self.attn_head(pair_emb)
        attn_w = F.softmax(attn_logits, dim=1)
        pair_attn = (pair_emb * attn_w).sum(dim=1)

        sp_spin = spin_vec.to(x.dtype).unsqueeze(0).unsqueeze(-1).expand(B, N, 1)
        sp_in = torch.cat([x_sc, sp_spin], dim=-1)
        sp_emb = self.sp_encoder(sp_in.reshape(B * N, -1)).reshape(B, N, -1)
        sp_sum = sp_emb.sum(dim=1)
        sp_mean = sp_emb.mean(dim=1)

        r2_mean = (x_sc**2).mean(dim=(1, 2)).unsqueeze(-1)
        extras = torch.cat([r2_mean, s1_mean], dim=1)

        rho_in = torch.cat([pair_sum, pair_mean, pair_attn, sp_sum, sp_mean, extras], dim=1)
        f_nn = self.readout(rho_in)
        return f_nn + self._compute_cusps(x, spin_vec)


class CTNNBackflowStyleJastrow(nn.Module, CuspMixin):
    """
    Jastrow model using the same copresheaf interaction pattern as CTNNBackflowNet.
    This is a sanity transfer: keep BF-style message passing, change head to scalar f(x).
    """

    def __init__(
        self,
        n_particles: int,
        d: int,
        omega: float,
        *,
        msg_hidden: int = 32,
        msg_layers: int = 2,
        hidden: int = 32,
        layers: int = 2,
        n_steps: int = 1,
        act: str = "silu",
        aggregation: str = "sum",
        use_spin: bool = True,
        same_spin_only: bool = False,
        readout_hidden: int = 64,
        readout_layers: int = 2,
    ):
        super().__init__()
        self._init_cusps(n_particles, d, omega)
        self.n_particles = n_particles
        self.d = d
        self.omega = omega
        self.use_spin = use_spin
        self.same_spin_only = same_spin_only
        self.aggregation = aggregation
        self.n_steps = n_steps

        def make_act(name: str):
            return {
                "relu": nn.ReLU,
                "gelu": nn.GELU,
                "tanh": nn.Tanh,
                "silu": nn.SiLU,
                "swish": nn.SiLU,
                "mish": nn.Mish,
            }[name.lower()]()

        self._act = make_act(act)

        node_in_dim = d + (1 if use_spin else 0)
        self.node_embed = nn.Linear(node_in_dim, hidden)

        edge_in_dim = d + 2
        self.edge_embed = self._mlp(edge_in_dim, msg_hidden, msg_hidden, msg_layers, self._act)

        self.rho_v_to_e = nn.ModuleList(
            [nn.Linear(hidden, msg_hidden, bias=False) for _ in range(n_steps)]
        )
        self.rho_e_to_v = nn.ModuleList(
            [nn.Linear(msg_hidden, hidden, bias=False) for _ in range(n_steps)]
        )

        self.edge_update = nn.ModuleList(
            [
                self._mlp(3 * msg_hidden, msg_hidden, msg_hidden, msg_layers, self._act)
                for _ in range(n_steps)
            ]
        )
        self.node_update = nn.ModuleList(
            [self._mlp(2 * hidden, hidden, hidden, layers, self._act) for _ in range(n_steps)]
        )

        ii, jj = torch.triu_indices(n_particles, n_particles, offset=1)
        self.register_buffer("idx_i", ii, persistent=False)
        self.register_buffer("idx_j", jj, persistent=False)

        readout_in = 2 * hidden + 2 * msg_hidden + 2
        layers_f = []
        dim = readout_in
        for _ in range(readout_layers):
            layers_f += [nn.Linear(dim, readout_hidden), make_act(act)]
            dim = readout_hidden
        layers_f.append(nn.Linear(dim, 1))
        nn.init.normal_(layers_f[-1].weight, std=1e-3)
        nn.init.zeros_(layers_f[-1].bias)
        self.f_head = nn.Sequential(*layers_f)

    def _mlp(self, in_dim, hid, out_dim, num_layers, act):
        layers = []
        if num_layers == 1:
            layers.append(nn.Linear(in_dim, out_dim))
        else:
            layers += [nn.Linear(in_dim, hid), act]
            for _ in range(num_layers - 2):
                layers += [nn.Linear(hid, hid), act]
            layers.append(nn.Linear(hid, out_dim))
        return nn.Sequential(*layers)

    def _aggregate(self, msgs):
        if self.aggregation == "sum":
            return msgs.sum(dim=2)
        if self.aggregation == "mean":
            return msgs.mean(dim=2)
        return msgs.max(dim=2).values

    def _resolve_spin(self, spin, B, N, device):
        if spin is None:
            up = N // 2
            return torch.cat(
                [
                    torch.zeros(up, dtype=torch.long, device=device),
                    torch.ones(N - up, dtype=torch.long, device=device),
                ]
            )
        s = spin.to(device).long()
        return s[0] if s.dim() == 2 else s

    def forward(self, x, spin=None):
        B, N, d = x.shape
        x_sc = x * (self.omega**0.5)
        spin_1d = self._resolve_spin(spin, B, N, x.device)

        if self.use_spin:
            sf = spin_1d.view(1, N, 1).to(x.dtype).expand(B, N, 1)
            node_in = torch.cat([x_sc, sf], dim=-1)
        else:
            node_in = x_sc
        h_v = self.node_embed(node_in)

        r = x_sc.unsqueeze(2) - x_sc.unsqueeze(1)
        r2 = (r**2).sum(dim=-1, keepdim=True)
        r1 = torch.sqrt(r2 + 1e-12)
        edge_in = torch.cat([r, r1, r2], dim=-1)
        h_e = self.edge_embed(edge_in)

        if self.use_spin:
            s_i = spin_1d.view(1, N, 1, 1).to(x.dtype).expand(B, N, N, 1)
            s_j = spin_1d.view(1, 1, N, 1).to(x.dtype).expand(B, N, N, 1)
            same = (s_i == s_j).to(x.dtype)
            weight = same if self.same_spin_only else torch.ones_like(same)
        else:
            weight = torch.ones_like(r[..., :1])

        eye = torch.eye(N, device=x.device, dtype=x.dtype).view(1, N, N, 1)
        weight = weight * (1.0 - eye)

        for step in range(self.n_steps):
            h_v_i = h_v.unsqueeze(2).expand(B, N, N, h_v.shape[-1])
            h_v_j = h_v.unsqueeze(1).expand(B, N, N, h_v.shape[-1])

            v_i_to_e = self.rho_v_to_e[step](h_v_i)
            v_j_to_e = self.rho_v_to_e[step](h_v_j)
            h_e_new = self.edge_update[step](torch.cat([h_e, v_i_to_e, v_j_to_e], dim=-1))

            msgs_e_to_v = self.rho_e_to_v[step](h_e_new) * weight
            m_v = self._aggregate(msgs_e_to_v)
            h_v = h_v + self.node_update[step](torch.cat([h_v, m_v], dim=-1))
            h_e = h_e + h_e_new

        h_v_sum = h_v.sum(dim=1)
        h_v_mean = h_v.mean(dim=1)
        h_e_pairs = h_e[:, self.idx_i, self.idx_j, :]
        h_e_sum = h_e_pairs.sum(dim=1)
        h_e_mean = h_e_pairs.mean(dim=1)
        r2_mean = (x_sc**2).mean(dim=(1, 2)).unsqueeze(-1)
        diff = (x[:, self.idx_i] - x[:, self.idx_j]).pow(2)
        dist = torch.sqrt(diff.sum(-1, keepdim=True) + 1e-8)
        s1_mean = torch.log1p((dist / 0.2) ** 2).mean(dim=1)

        f_in = torch.cat([h_v_sum, h_v_mean, h_e_sum, h_e_mean, r2_mean, s1_mean], dim=1)
        f_nn = self.f_head(f_in)
        return f_nn + self._compute_cusps(x, spin_1d)


class CTNNJastrowVCycle(nn.Module, CuspMixin):
    """
    Copresheaf CTNN with V-cycle interaction:
      down-pass message updates -> bottleneck coarsening -> up-pass refinement.
    """

    def __init__(
        self,
        n_particles: int,
        d: int,
        omega: float,
        *,
        node_hidden: int = 32,
        edge_hidden: int = 32,
        bottleneck_hidden: int = 16,
        n_down: int = 2,
        n_up: int = 2,
        msg_layers: int = 2,
        node_layers: int = 2,
        readout_hidden: int = 64,
        readout_layers: int = 2,
        act: str = "silu",
        aggregation: str = "sum",
        use_spin: bool = True,
    ):
        super().__init__()
        self._init_cusps(n_particles, d, omega)
        self.n_particles = n_particles
        self.d = d
        self.omega = omega
        self.n_down = n_down
        self.n_up = n_up
        self.aggregation = aggregation
        self.use_spin = use_spin

        def make_act():
            return {
                "silu": nn.SiLU,
                "gelu": nn.GELU,
                "relu": nn.ReLU,
                "tanh": nn.Tanh,
                "mish": nn.Mish,
            }[act.lower()]()

        node_in = d + (1 if use_spin else 0)
        self.node_embed = nn.Linear(node_in, node_hidden)
        self.edge_embed = self._mlp(d + 2, edge_hidden, edge_hidden, msg_layers, make_act)

        self.rho_v_to_e_down = nn.ModuleList(
            [nn.Linear(node_hidden, edge_hidden, bias=False) for _ in range(n_down)]
        )
        self.rho_e_to_v_down = nn.ModuleList(
            [nn.Linear(edge_hidden, node_hidden, bias=False) for _ in range(n_down)]
        )
        self.edge_updates_down = nn.ModuleList(
            [
                self._mlp(3 * edge_hidden, edge_hidden, edge_hidden, msg_layers, make_act)
                for _ in range(n_down)
            ]
        )
        self.node_updates_down = nn.ModuleList(
            [
                self._mlp(2 * node_hidden, node_hidden, node_hidden, node_layers, make_act)
                for _ in range(n_down)
            ]
        )

        self.node_down = nn.Linear(node_hidden, bottleneck_hidden)
        self.edge_down = nn.Linear(edge_hidden, bottleneck_hidden)
        self.node_up = nn.Linear(bottleneck_hidden, node_hidden)
        self.edge_up = nn.Linear(bottleneck_hidden, edge_hidden)

        self.rho_v_to_e_up = nn.ModuleList(
            [nn.Linear(node_hidden, edge_hidden, bias=False) for _ in range(n_up)]
        )
        self.rho_e_to_v_up = nn.ModuleList(
            [nn.Linear(edge_hidden, node_hidden, bias=False) for _ in range(n_up)]
        )
        self.edge_updates_up = nn.ModuleList(
            [
                self._mlp(3 * edge_hidden, edge_hidden, edge_hidden, msg_layers, make_act)
                for _ in range(n_up)
            ]
        )
        self.node_updates_up = nn.ModuleList(
            [
                self._mlp(2 * node_hidden, node_hidden, node_hidden, node_layers, make_act)
                for _ in range(n_up)
            ]
        )

        self.node_skip_fuse = self._mlp(2 * node_hidden, node_hidden, node_hidden, 2, make_act)
        self.edge_skip_fuse = self._mlp(2 * edge_hidden, edge_hidden, edge_hidden, 2, make_act)

        ii, jj = torch.triu_indices(n_particles, n_particles, offset=1)
        self.register_buffer("idx_i", ii, persistent=False)
        self.register_buffer("idx_j", jj, persistent=False)

        readout_in = 2 * node_hidden + 3 * edge_hidden + 2
        layers_f = []
        dim = readout_in
        for _ in range(readout_layers):
            layers_f += [nn.Linear(dim, readout_hidden), make_act()]
            dim = readout_hidden
        layers_f.append(nn.Linear(dim, 1))
        nn.init.normal_(layers_f[-1].weight, std=1e-3)
        nn.init.zeros_(layers_f[-1].bias)
        self.f_head = nn.Sequential(*layers_f)
        self.edge_attn = nn.Sequential(
            nn.Linear(edge_hidden, edge_hidden // 2),
            make_act(),
            nn.Linear(edge_hidden // 2, 1),
        )

    def _mlp(self, in_dim, hid, out_dim, n_layers, act_fn):
        layers = []
        if n_layers == 1:
            layers.append(nn.Linear(in_dim, out_dim))
        else:
            layers += [nn.Linear(in_dim, hid), act_fn()]
            for _ in range(n_layers - 2):
                layers += [nn.Linear(hid, hid), act_fn()]
            layers.append(nn.Linear(hid, out_dim))
        return nn.Sequential(*layers)

    def _aggregate(self, msgs):
        if self.aggregation == "sum":
            return msgs.sum(dim=2)
        if self.aggregation == "mean":
            return msgs.mean(dim=2)
        return msgs.max(dim=2).values

    def _resolve_spin(self, spin, B, N, device):
        if spin is None:
            up = N // 2
            return torch.cat(
                [
                    torch.zeros(up, dtype=torch.long, device=device),
                    torch.ones(N - up, dtype=torch.long, device=device),
                ]
            )
        s = spin.to(device).long()
        return s[0] if s.dim() == 2 else s

    def _message_step(self, h_v, h_e, rho_v_to_e, rho_e_to_v, edge_update, node_update, weight):
        B, N = h_v.shape[0], h_v.shape[1]
        h_v_i = h_v.unsqueeze(2).expand(B, N, N, h_v.shape[-1])
        h_v_j = h_v.unsqueeze(1).expand(B, N, N, h_v.shape[-1])
        edge_in = torch.cat([h_e, rho_v_to_e(h_v_i), rho_v_to_e(h_v_j)], dim=-1)
        h_e_new = h_e + edge_update(edge_in)
        msgs = rho_e_to_v(h_e_new) * weight
        h_v_new = h_v + node_update(torch.cat([h_v, self._aggregate(msgs)], dim=-1))
        return h_v_new, h_e_new

    def forward(self, x, spin=None):
        B, N, d = x.shape
        x_sc = x * (self.omega**0.5)
        spin_1d = self._resolve_spin(spin, B, N, x.device)

        if self.use_spin:
            sf = spin_1d.view(1, N, 1).to(x.dtype).expand(B, N, 1)
            node_in = torch.cat([x_sc, sf], dim=-1)
        else:
            node_in = x_sc
        h_v = self.node_embed(node_in)

        r_vec = x_sc.unsqueeze(2) - x_sc.unsqueeze(1)
        r2 = (r_vec**2).sum(-1, keepdim=True)
        r1 = torch.sqrt(r2 + 1e-12)
        h_e = self.edge_embed(torch.cat([r_vec, r1, r2], dim=-1))

        eye = torch.eye(N, device=x.device, dtype=x.dtype).view(1, N, N, 1)
        weight = 1.0 - eye

        down_skips = []
        for k in range(self.n_down):
            h_v, h_e = self._message_step(
                h_v,
                h_e,
                self.rho_v_to_e_down[k],
                self.rho_e_to_v_down[k],
                self.edge_updates_down[k],
                self.node_updates_down[k],
                weight,
            )
            down_skips.append((h_v, h_e))

        h_v = self.node_up(self.node_down(h_v))
        h_e = self.edge_up(self.edge_down(h_e))

        for k in range(self.n_up):
            skip_v, skip_e = down_skips[-(k + 1)]
            h_v = self.node_skip_fuse(torch.cat([h_v, skip_v], dim=-1))
            h_e = self.edge_skip_fuse(torch.cat([h_e, skip_e], dim=-1))
            h_v, h_e = self._message_step(
                h_v,
                h_e,
                self.rho_v_to_e_up[k],
                self.rho_e_to_v_up[k],
                self.edge_updates_up[k],
                self.node_updates_up[k],
                weight,
            )

        h_v_sum = h_v.sum(dim=1)
        h_v_mean = h_v.mean(dim=1)
        h_e_pairs = h_e[:, self.idx_i, self.idx_j, :]
        h_e_sum = h_e_pairs.sum(dim=1)
        h_e_mean = h_e_pairs.mean(dim=1)
        attn_w = F.softmax(self.edge_attn(h_e_pairs), dim=1)
        h_e_attn = (h_e_pairs * attn_w).sum(dim=1)
        r2_mean = (x_sc**2).mean(dim=(1, 2)).unsqueeze(-1)
        diff = (x[:, self.idx_i] - x[:, self.idx_j]).pow(2)
        dist = torch.sqrt(diff.sum(-1, keepdim=True) + 1e-8)
        s1_mean = torch.log1p((dist / 0.2) ** 2).mean(dim=1)

        f_in = torch.cat([h_v_sum, h_v_mean, h_e_sum, h_e_mean, h_e_attn, r2_mean, s1_mean], dim=1)
        f_nn = self.f_head(f_in)
        return f_nn + self._compute_cusps(x, spin_1d)
