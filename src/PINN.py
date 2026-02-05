import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ZeroJastrow(nn.Module):
    """f(x) ≡ 0. No parameters; always returns zeros of shape (B,1)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)


class DetachWrapper(nn.Module):
    def __init__(self, mod: nn.Module):
        super().__init__()
        self.mod = mod
        for p in self.mod.parameters():
            p.requires_grad = False

    def forward(self, *args, **kwargs):
        with torch.no_grad():
            return self.mod(*args, **kwargs)


class PINN(nn.Module):
    """
    f(x) = rho([ mean_i φ(x_i),  pooled_{i<j} ψ(safe_features(r_ij)) , safe_extras ]) -> (B,1)
    Ψ(x) = SD(x) * exp(f(x)).
    Adds analytic cusp u(r)=γ r exp(-r/ℓ) directly to log Ψ.

    Safe-by-design:
      • Pair features: ds/dr → 0 as r→0 (no extra 1/r in ∇/Δ logΨ).
      • No raw r in 'extras' (uses ⟨r²⟩ and safe ⟨s1⟩).
      • Optional gate χ(r): χ(0)=χ'(0)=0, fades to 1 past ~0.3 a_ho.
      • Optional simple radial attention (degree-normalized).
    """

    def __init__(
        self,
        n_particles: int,
        d: int,
        omega: float,
        *,
        dL: int = 5,
        hidden_dim: int = 128,
        n_layers: int = 2,
        act: str = "gelu",
        init: str = "xavier",
        use_pair_attn: bool = False,
        use_gate: bool = True,
        gate_radius_aho: float = 0.30,
        eps_feat_aho: float = 0.20,
    ):
        super().__init__()
        self.n_particles = int(n_particles)
        self.d = int(d)
        self.dL = int(dL)
        self.omega = float(omega)
        self.use_pair_attn = bool(use_pair_attn)
        self.use_gate = bool(use_gate)
        self.gate_radius_aho = float(gate_radius_aho)
        self.eps_feat_aho = float(eps_feat_aho)

        # indices for i<j pairs
        ii, jj = torch.triu_indices(self.n_particles, self.n_particles, offset=1)
        self.register_buffer("idx_i", ii, persistent=False)
        self.register_buffer("idx_j", jj, persistent=False)

        # activations / MLPs
        self.act = self._make_act(act)
        self.phi = self._build_mlp(self.d, hidden_dim, self.dL, n_layers, self.act)
        self.psi_in_dim = 6
        self.psi = self._build_mlp(self.psi_in_dim, hidden_dim, self.dL, n_layers, self.act)
        # inputs: mean φ (dL) + pooled ψ (dL) + safe extras (2) = 2*dL + 2
        self.rho = self._build_mlp(2 * self.dL + 2, hidden_dim, 1, 2, self.act)

        # optional norms (off by default)
        self.psi_norm = nn.LayerNorm(self.dL, elementwise_affine=True)
        self.rho_norm = nn.LayerNorm(2 * self.dL + 2, elementwise_affine=True)

        self._initialize_weights(init)

        # analytic cusp params (general d): γ_ud=1/(d-1), γ_uu=1/(d+1)
        self.gamma_apara = 1.0 / (self.d - 1)
        self.gamma_para = 1.0 / (self.d + 1)
        self.cusp_len = 1.0 / (self.omega**0.5)

        # attention defaults
        self.attn_rc = 0.4
        self.attn_p = 6.0

    # ----- utils -----
    def _make_act(self, name: str) -> nn.Module:
        name = name.lower()
        if name == "relu":
            return nn.ReLU()
        if name == "gelu":
            return nn.GELU()
        if name == "tanh":
            return nn.Tanh()
        if name in ("silu", "swish"):
            return nn.SiLU()
        if name == "mish":
            return getattr(nn, "Mish", nn.SiLU)()
        if name == "leakyrelu":
            return nn.LeakyReLU(0.1)
        raise ValueError(f"Unknown act '{name}'")

    def _build_mlp(self, in_dim, hidden_dim, out_dim, n_layers, act):
        layers = [nn.Linear(in_dim, hidden_dim), act]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), act]
        layers += [nn.Linear(hidden_dim, out_dim)]
        return nn.Sequential(*layers)

    def _initialize_weights(self, scheme: str):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if scheme == "custom":
                    nn.init.xavier_normal_(m.weight, gain=1.13)
                elif scheme == "xavier":
                    nn.init.xavier_normal_(m.weight)
                elif scheme == "he":
                    nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                elif scheme == "lecun":
                    nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="linear")
                else:
                    raise ValueError(f"Unknown init scheme {scheme}")
                nn.init.zeros_(m.bias)

    # ----- safe radial features & gate -----
    def _safe_pair_features(self, r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        r: (B,P,1)
        Returns (features, s1_mean) where features is (B,P,6) and s1_mean is (B,1) for extras.
        All channels have ds/dr -> 0 as r->0.
        """
        a_ho = 1.0 / (self.omega**0.5)
        eps = self.eps_feat_aho * a_ho
        r2 = r * r
        rt = torch.sqrt(r2 + eps * eps)  # mollified radius

        s1 = torch.log1p((rt / eps) ** 2)  # ds/dr ~ 2r/eps^2
        s2 = r2 / (r2 + eps * eps)  # ds/dr ~ O(r)
        s3 = (rt / eps) ** 2 * torch.exp(-((rt / eps) ** 2))  # bump, ds/dr ~ O(r)

        g = torch.as_tensor([0.25, 1.0, 4.0], device=r.device, dtype=r.dtype).view(1, 1, -1)
        rbf = torch.exp(-g * s1)  # χ(0)=1, χ'(0)=0

        feat = torch.cat([s1, s2, s3, rbf], dim=-1)  # (B,P,6)
        s1_mean = s1.mean(dim=1, keepdim=False)  # (B,1) safe extra
        return feat, s1_mean

    def _short_range_gate(self, r: torch.Tensor) -> torch.Tensor:
        """χ(0)=0, χ'(0)=0, χ→1 after ~gate_radius_aho * a_ho. r: (B,P,1)"""
        if not self.use_gate:
            return torch.ones_like(r)
        a_ho = 1.0 / (self.omega**0.5)
        rg = self.gate_radius_aho * a_ho
        r2 = r * r
        return r2 / (r2 + rg * rg)

    # ----- forward -----
    def forward(self, x: torch.Tensor, spin: torch.Tensor | None = None) -> torch.Tensor:
        """
        x : (B,N,d)  -> used for NN features (phi/psi/extras)
        cusp_coords : (B,N,d) or None; if provided, cusp distances are computed
                      from these *physical* coords (default: x).
        returns: (B,1) = f_NN(x) + u_cusp(cusp_coords)
        """
        B, N, d = x.shape
        assert N == self.n_particles and d == self.d

        # --- NN features & pooling use (possibly backflowed) 'x' ---
        x_scaled = x * (self.omega**0.5)
        diff = x_scaled.unsqueeze(2) - x_scaled.unsqueeze(1)  # (B,N,N,d)
        diff_pairs = diff[:, self.idx_i, self.idx_j, :]  # (B,P,d)
        r2 = (diff_pairs * diff_pairs).sum(dim=-1, keepdim=True)  # (B,P,1)
        r = torch.sqrt(r2 + torch.finfo(x.dtype).eps)

        # φ branch
        phi_flat = x_scaled.reshape(B * N, d)
        phi_out = self.phi(phi_flat).reshape(B, N, self.dL)
        phi_mean = phi_out.mean(dim=1)  # (B,dL)

        # ψ branch (safe pair features)
        psi_in, s1_mean = self._safe_pair_features(r)  # (B,P,6), (B,1)
        psi_out = self.psi(psi_in.reshape(-1, self.psi_in_dim)).reshape(B, -1, self.dL)

        # optional short-range gate
        gate = self._short_range_gate(r)  # (B,P,1)
        psi_out = psi_out * gate
        psi_mean = psi_out.mean(dim=1)  # (B,dL)  (or your attn)

        # SAFE extras only
        r2_mean = (x_scaled**2).mean(dim=(1, 2), keepdim=False).unsqueeze(-1)  # (B,1)
        extras = torch.cat([r2_mean, s1_mean], dim=1)  # (B,2)

        # readout
        rho_in = torch.cat([phi_mean, psi_mean, extras], dim=1)  # (B,2*dL+2)
        out = self.rho(rho_in)  # (B,1)

        # --- Analytic cusp uses *physical* coords by default ---
        diff = x.unsqueeze(2) - x.unsqueeze(1)  # (B,N,N,d)
        diff_pairs = diff[:, self.idx_i, self.idx_j, :]  # (B,P,d)
        r2_c = (diff_pairs * diff_pairs).sum(dim=-1, keepdim=True)  # (B,P,1)
        r_c = torch.sqrt(r2_c + torch.finfo(x.dtype).eps)

        # spin handling (same as before)
        if spin is None:
            up = N // 2
            spin = (
                torch.cat(
                    [
                        torch.zeros(up, dtype=torch.long, device=x.device),
                        torch.ones(N - up, dtype=torch.long, device=x.device),
                    ],
                    dim=0,
                )
                .unsqueeze(0)
                .expand(B, -1)
            )  # (B,N)
        else:
            spin = spin.to(x.device).long()
            if spin.dim() == 1:
                spin = spin.unsqueeze(0).expand(B, -1)

        si = spin[:, self.idx_i]
        sj = spin[:, self.idx_j]
        same_spin = (si == sj).to(x.dtype).unsqueeze(-1)  # (B,P,1)

        gamma_para = torch.as_tensor(self.gamma_para, dtype=x.dtype, device=x.device).view(1, 1, 1)
        gamma_apara = torch.as_tensor(self.gamma_apara, dtype=x.dtype, device=x.device).view(
            1, 1, 1
        )
        gamma = same_spin * gamma_para + (1.0 - same_spin) * gamma_apara  # (B,P,1)

        ell = torch.as_tensor(self.cusp_len, dtype=x.dtype, device=x.device).view(1, 1, 1)
        pair_u = gamma * r_c * torch.exp(-r_c)  # (B,P,1)
        cusp_term = pair_u.sum(dim=1)  # (B,1)

        return out + cusp_term


class BackflowNet(nn.Module):
    """
    Minimal, configurable backflow network.
    Returns Δx with shape (B, N, d).

    Configurable:
      - act: "gelu"|"relu"|"tanh"|"silu"|"mish"|"leakyrelu"|"identity"
      - msg_layers / msg_hidden  (message MLP φ)
      - layers / hidden          (node/update MLP ψ)
      - aggregation: "sum"|"mean"|"max"
      - use_spin & same_spin_only
      - out_bound: "tanh"|"identity"
      - zero_init_last (start near identity)
    """

    def __init__(
        self,
        d: int,
        *,
        msg_hidden: int = 128,
        msg_layers: int = 2,
        hidden: int = 128,
        layers: int = 3,
        act: str = "silu",
        aggregation: str = "sum",
        use_spin: bool = True,
        same_spin_only: bool = False,
        out_bound: str = "tanh",
        bf_scale_init: float = 0.05,
        zero_init_last: bool = True,
    ):
        super().__init__()
        self.d = d
        self.use_spin = use_spin
        self.same_spin_only = same_spin_only
        self.aggregation = aggregation
        self.out_bound = out_bound

        # --- tiny activation factory (no helpers) ---
        def make_act(name: str) -> nn.Module:
            name = name.lower()
            if name == "relu":
                return nn.ReLU()
            if name == "gelu":
                return nn.GELU()
            if name == "tanh":
                return nn.Tanh()
            if name in ("silu", "swish"):
                return nn.SiLU()
            if name == "mish":
                return getattr(nn, "Mish", nn.SiLU)()
            if name == "leakyrelu":
                return nn.LeakyReLU(0.1)
            if name in ("identity", "none"):
                return nn.Identity()
            raise ValueError(f"Unknown activation '{name}'")

        self._act = make_act(act)

        # φ: message MLP → maps (3d+2) → msg_hidden
        msg_in = 3 * d + 2
        self.phi = self._mlp(msg_in, msg_hidden, msg_hidden, msg_layers, self._act)

        # ψ: node/update MLP → maps (d + msg_hidden) → d
        upd_in = d + msg_hidden
        self.psi = self._mlp(upd_in, hidden, d, layers, self._act)

        # positive learnable scale via softplus
        self.bf_scale_raw = nn.Parameter(torch.tensor(math.log(math.exp(bf_scale_init) - 1.0)))
        # zero-init last ψ layer to start Δx≈0 (identity backflow)
        if zero_init_last:
            last = self.psi[-1]  # Linear
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def _mlp(self, in_dim: int, hid: int, out_dim: int, num_layers: int, act: nn.Module):
        assert num_layers >= 1
        layers = []
        if num_layers == 1:
            layers.append(nn.Linear(in_dim, out_dim))
        else:
            layers.append(nn.Linear(in_dim, hid))
            layers.append(act)
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hid, hid))
                layers.append(act)
            layers.append(nn.Linear(hid, out_dim))
        return nn.Sequential(*layers)

    @property
    def bf_scale(self):
        return F.softplus(self.bf_scale_raw)

    def _aggregate(self, m_ij: torch.Tensor) -> torch.Tensor:
        # m_ij: (B, N, N, H)
        if self.aggregation == "sum":
            return m_ij.sum(dim=2)
        if self.aggregation == "mean":
            return m_ij.mean(dim=2)
        if self.aggregation == "max":
            return m_ij.max(dim=2).values
        raise ValueError(f"Unknown aggregation '{self.aggregation}'")

    def forward(self, x: torch.Tensor, spin: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (B, N, d), spin optional: (N,) or (B, N) with {0,1}
        returns Δx: (B, N, d)
        """
        B, N, d = x.shape
        assert d == self.d

        # Pairwise deltas and norms
        r = x.unsqueeze(2) - x.unsqueeze(1)  # (B,N,N,d)
        r2 = (r**2).sum(dim=-1, keepdim=True)  # (B,N,N,1)
        r1 = torch.sqrt(r2 + 1e-12)  # (B,N,N,1)

        xi = x.unsqueeze(2).expand(B, N, N, d)
        xj = x.unsqueeze(1).expand(B, N, N, d)
        msg_in = torch.cat([xi, xj, r, r1, r2], dim=-1)  # (B,N,N,3d+2)

        # Spin weights (optional)
        if self.use_spin and spin is not None:
            if spin.ndim == 1:
                s_i = spin.view(1, N, 1, 1).to(x.dtype).expand(B, N, N, 1)
                s_j = spin.view(1, 1, N, 1).to(x.dtype).expand(B, N, N, 1)
            else:  # (B,N)
                s_i = spin.view(B, N, 1, 1).to(x.dtype).expand(B, N, N, 1)
                s_j = spin.view(B, 1, N, 1).to(x.dtype).expand(B, N, N, 1)
            same = (s_i == s_j).to(x.dtype)
            weight = same if self.same_spin_only else torch.ones_like(same)
        else:
            weight = torch.ones_like(r[..., :1])

        # Remove self-messages
        eye = torch.eye(N, device=x.device, dtype=x.dtype).view(1, N, N, 1)
        weight = weight * (1.0 - eye)

        # Message pass + aggregate
        m_ij = self.phi(msg_in) * weight  # (B,N,N,H)
        m_i = self._aggregate(m_ij)  # (B,N,H)

        # Node/update → Δx_i
        upd = torch.cat([x, m_i], dim=-1)  # (B,N,d+H)
        dx = self.psi(upd)  # (B,N,d)

        # Output bound + positive scale
        if self.out_bound == "tanh":
            dx = torch.tanh(dx)
        elif self.out_bound == "identity":
            pass
        else:
            raise ValueError(f"Unknown out_bound '{self.out_bound}'")
        dx = dx - dx.mean(dim=1, keepdim=True)
        bf_scale = F.softplus(self.bf_scale_raw)  # > 0
        return dx * bf_scale


import torch
import torch.nn as nn


class CTNNBackflowNet(nn.Module):
    """
    Copresheaf / graph-style backflow network.
    Drop-in replacement for BackflowNet.

    Interface:
      forward(x, spin=None) -> Δx with shape (B, N, d)

    Differences vs BackflowNet:
      - Maintains explicit node and edge features.
      - Message passing uses learned linear "transport" maps:
          node -> edge  (rho_v_to_e)
          edge -> node  (rho_e_to_v)
      - Still respects aggregation='sum'|'mean'|'max',
        spin masking, zero-mean Δx, and positive scale via bf_scale.
    """

    def __init__(
        self,
        d: int,
        *,
        msg_hidden: int = 128,
        msg_layers: int = 2,
        hidden: int = 128,
        layers: int = 3,
        act: str = "silu",
        aggregation: str = "sum",
        use_spin: bool = True,
        same_spin_only: bool = False,
        out_bound: str = "tanh",
        bf_scale_init: float = 0.05,
        zero_init_last: bool = True,
        omega: float = 1.0,
    ):
        super().__init__()
        self.d = d
        self.use_spin = use_spin
        self.same_spin_only = same_spin_only
        self.aggregation = aggregation
        self.out_bound = out_bound
        self.omega = omega

        # --- activation factory (same semantics as your original) ---
        def make_act(name: str) -> nn.Module:
            name = name.lower()
            if name == "relu":
                return nn.ReLU()
            if name == "gelu":
                return nn.GELU()
            if name == "tanh":
                return nn.Tanh()
            if name in ("silu", "swish"):
                return nn.SiLU()
            if name == "mish":
                return getattr(nn, "Mish", nn.SiLU)()
            if name == "leakyrelu":
                return nn.LeakyReLU(0.1)
            if name in ("identity", "none"):
                return nn.Identity()
            raise ValueError(f"Unknown activation '{name}'")

        self._act = make_act(act)

        # ---------- Feature dimensions ----------
        # Node input: positions (+ optional spin scalar)
        node_in_dim = d + (1 if use_spin else 0)
        node_hidden = hidden  # node feature dim = hidden
        edge_hidden = msg_hidden  # edge feature dim = msg_hidden

        # ---------- Embeddings ----------
        # Node embedding: (x_i, spin_i) -> h_i
        self.node_embed = nn.Linear(node_in_dim, node_hidden)

        # Edge initial features: use relative geometry only:
        #   [r_ij (d), |r_ij|, |r_ij|^2]  -> edge_hidden
        edge_in_dim = d + 2
        self.edge_embed = self._mlp(
            edge_in_dim,
            edge_hidden,
            edge_hidden,
            msg_layers,
            self._act,
        )

        # ---------- Copresheaf-style transport maps ----------
        # Node -> Edge: maps node space to edge space
        self.rho_v_to_e = nn.Linear(node_hidden, edge_hidden, bias=False)
        # Edge -> Node: maps edge space to node space
        self.rho_e_to_v = nn.Linear(edge_hidden, node_hidden, bias=False)

        # ---------- Update MLPs ----------
        # Edge update: uses current edge feat + transported node feats
        #   [h_e, rho_v_to_e(h_i), rho_v_to_e(h_j)] with dim 3 * edge_hidden
        self.edge_update = self._mlp(
            3 * edge_hidden,
            edge_hidden,
            edge_hidden,
            msg_layers,
            self._act,
        )

        # Node update: residual update on node features
        #   [h_v, m_v] with dim 2 * node_hidden
        self.node_update = self._mlp(
            2 * node_hidden,
            node_hidden,
            node_hidden,
            layers,
            self._act,
        )

        # Final head to produce Δx from node features
        self.dx_head = nn.Linear(node_hidden, d)

        # positive learnable scale via softplus (same semantics)
        self.bf_scale_raw = nn.Parameter(torch.tensor(math.log(math.exp(bf_scale_init) - 1.0)))

        # zero-init last dx layer to start Δx≈0 (identity backflow)
        if zero_init_last:
            nn.init.zeros_(self.dx_head.weight)
            nn.init.zeros_(self.dx_head.bias)

    # ---------- helpers ----------

    def _mlp(self, in_dim: int, hid: int, out_dim: int, num_layers: int, act: nn.Module):
        assert num_layers >= 1
        layers = []
        if num_layers == 1:
            layers.append(nn.Linear(in_dim, out_dim))
        else:
            layers.append(nn.Linear(in_dim, hid))
            layers.append(act)
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hid, hid))
                layers.append(act)
            layers.append(nn.Linear(hid, out_dim))
        return nn.Sequential(*layers)

    @property
    def bf_scale(self):
        return F.softplus(self.bf_scale_raw)

    def _aggregate(self, msgs: torch.Tensor) -> torch.Tensor:
        """
        msgs: (B, N, N, H) — messages from j -> i along last dim H
        returns m_i: (B, N, H)
        """
        if self.aggregation == "sum":
            return msgs.sum(dim=2)
        if self.aggregation == "mean":
            return msgs.mean(dim=2)
        if self.aggregation == "max":
            return msgs.max(dim=2).values
        raise ValueError(f"Unknown aggregation '{self.aggregation}'")

    # ---------- forward pass ----------

    def forward(self, x: torch.Tensor, spin: torch.Tensor | None = None) -> torch.Tensor:
        """
        x:    (B, N, d)
        spin: (N,) or (B, N) with {0,1}, optional

        returns:
          Δx: (B, N, d)
        """
        B, N, d = x.shape
        assert d == self.d
        x = x * (self.omega**0.5)
        # -------- node input features --------
        if self.use_spin and spin is not None:
            if spin.ndim == 1:
                spin_feat = spin.view(1, N, 1).to(x.dtype).expand(B, N, 1)
            else:  # (B,N)
                spin_feat = spin.view(B, N, 1).to(x.dtype)
            node_in = torch.cat([x, spin_feat], dim=-1)  # (B,N,d+1)
        else:
            node_in = x  # (B,N,d)

        # Initial node features h_v: (B,N,node_hidden)
        h_v = self.node_embed(node_in)

        # -------- edge geometry --------
        # pairwise relative vectors and norms
        r = x.unsqueeze(2) - x.unsqueeze(1)  # (B,N,N,d)
        r2 = (r**2).sum(dim=-1, keepdim=True)  # (B,N,N,1)
        r1 = torch.sqrt(r2 + 1e-12)  # (B,N,N,1)

        edge_in = torch.cat([r, r1, r2], dim=-1)  # (B,N,N,d+2)
        # Initial edge features h_e: (B,N,N,edge_hidden)
        h_e = self.edge_embed(edge_in)

        # -------- spin-based edge weights (mask) --------
        if self.use_spin and spin is not None:
            if spin.ndim == 1:
                s_i = spin.view(1, N, 1, 1).to(x.dtype).expand(B, N, N, 1)
                s_j = spin.view(1, 1, N, 1).to(x.dtype).expand(B, N, N, 1)
            else:  # (B,N)
                s_i = spin.view(B, N, 1, 1).to(x.dtype).expand(B, N, N, 1)
                s_j = spin.view(B, 1, N, 1).to(x.dtype).expand(B, N, N, 1)
            same = (s_i == s_j).to(x.dtype)
            if self.same_spin_only:
                weight = same  # only same-spin edges
            else:
                weight = torch.ones_like(same)  # all edges allowed
        else:
            weight = torch.ones_like(r[..., :1])  # (B,N,N,1)

        # remove self-messages
        eye = torch.eye(N, device=x.device, dtype=x.dtype).view(1, N, N, 1)
        weight = weight * (1.0 - eye)  # (B,N,N,1)

        # -------- node -> edge transport --------
        # expand node features to edge positions
        h_v_i = h_v.unsqueeze(2).expand(B, N, N, h_v.shape[-1])
        h_v_j = h_v.unsqueeze(1).expand(B, N, N, h_v.shape[-1])

        v_i_to_e = self.rho_v_to_e(h_v_i)  # (B,N,N,edge_hidden)
        v_j_to_e = self.rho_v_to_e(h_v_j)  # (B,N,N,edge_hidden)

        # edge update (local CTNN step on edges)
        edge_update_in = torch.cat([h_e, v_i_to_e, v_j_to_e], dim=-1)  # (B,N,N,3*edge_hidden)
        h_e_new = self.edge_update(edge_update_in)  # (B,N,N,edge_hidden)

        # -------- edge -> node transport --------
        msgs_e_to_v = self.rho_e_to_v(h_e_new)  # (B,N,N,node_hidden)
        msgs_e_to_v = msgs_e_to_v * weight  # apply spin + no-self mask

        # aggregate messages per node i over neighbors j
        m_v = self._aggregate(msgs_e_to_v)  # (B,N,node_hidden)

        # -------- node update with residual --------
        node_update_in = torch.cat([h_v, m_v], dim=-1)  # (B,N,2*node_hidden)
        delta_h = self.node_update(node_update_in)  # (B,N,node_hidden)
        h_v = h_v + delta_h  # residual update

        # -------- map to Δx --------
        dx = self.dx_head(h_v)  # (B,N,d)

        # output bound
        if self.out_bound == "tanh":
            dx = torch.tanh(dx)
        elif self.out_bound == "identity":
            pass
        else:
            raise ValueError(f"Unknown out_bound '{self.out_bound}'")

        # enforce zero center-of-mass shift
        dx = dx - dx.mean(dim=1, keepdim=True)

        # positive scale factor (same semantics as original)
        bf_scale = F.softplus(self.bf_scale_raw)  # > 0
        return dx * bf_scale


# Usage:
# --- Training Backflow ---
# opt = torch.optim.AdamW([
#     {"params": [p for n,p in bf.named_parameters() if n != "bf_scale_raw"], "lr": 1e-3},
#     {"params": [bf.bf_scale_raw], "lr": 1e-3, "weight_decay": 1e-5},
# ])

# _, bf = train_model(
#     f_net_zero,
#     opt,
#     C_occ,
#     mapper,
#     backflow_net=bf,
#     std=std,
#     print_e=10,
# )
# bf_frozen = DetachWrapper(bf).to(cfg.torch_device, cfg.torch_dtype)

# --- Training phi only ---
# f_net.freeze_rho_as_avg()
# f_net.set_trainable(phi=True, psi=False, rho=False)
# opt = torch.optim.AdamW(f_net.param_groups(lr_phi=1e-3, lr_psi=0, lr_rho=0, wd=1e-4))

# --- Training psi only ---
# f_net.freeze_rho_as_avg()
# f_net.set_trainable(phi=False, psi=True, rho=False)
# opt = torch.optim.AdamW(f_net.param_groups(lr_phi=0, lr_psi=1e-3, lr_rho=0, wd=1e-4))

# --- Training entire network ---
# f_net.unfreeze_rho(reinit_zero=False)   # keep averaging weights as a warm start
# f_net.set_trainable(phi=True, psi=True, rho=True)
# opt = torch.optim.AdamW(f_net.param_groups(lr_phi=1e-4, lr_psi=1e-3, lr_rho=1e-4, wd=0))

# f_net, _ = train_model(
#     f_net,
#     opt,
#     C_occ,
#     mapper,
#     backflow_net=None,
#     std=std,
#     print_e=10,
# )
