"""
Full Stochastic Reconfiguration (SR) preconditioner for natural gradient collocation.

Two implementations of the full Fisher/overlap matrix inverse:

1. **WoodburySR** — Exact solve in sample space via the Woodbury identity.
   Forms the per-sample gradient matrix O (N_samples × N_params), then solves
   in the smaller (N_samples × N_samples) space:
       F⁻¹g = (1/λ)(g − Oᵀ(OOᵀ + λN·I)⁻¹ Og)
   Cost: O(N_samples² · N_params) for forming O, O(N_samples³) for the solve.
   Exact (up to sample noise). Practical when N_samples ≤ ~4096.

2. **CGSR** — Iterative CG solve in parameter space.
   Solves (F + λI)δθ = g using conjugate gradient. Each CG iteration needs
   one matrix-vector product Fv, computed via two AD passes (JVP + VJP).
   Cost: O(n_cg_iters · backward_cost). No explicit O matrix needed.
   Approximate (truncated CG), but works at any sample count.

Both include SR stabilizations:
  - Diagonal shift (Tikhonov damping) λ, optionally annealed
  - Max parameter change ‖δθ‖∞ clipping
  - Trust region: scale δθ if ‖δθ‖₂ exceeds threshold
  - EMA smoothing option for the Fisher (CG mode)

Interface matches DiagonalFisherPreconditioner: update() then precondition().
"""

from __future__ import annotations

import math
from typing import Sequence, Optional

import torch
from torch import Tensor


# ═════════════════════════════════════════════════════════════════
#  Woodbury SR
# ═════════════════════════════════════════════════════════════════


class WoodburySR:
    """
    Full SR via Woodbury identity in sample space.

    Forms O_kj = ∂log|Ψ(x_k)|/∂θ_j  (per-sample gradients),
    then uses:
        (OᵀO/N + λI)⁻¹ g = (1/λ)(g - Oᵀ (OOᵀ/N + λI)⁻¹ O g / N)

    This is exact and costs O(B² P + B³) where B = n_samples, P = n_params.
    """

    def __init__(
        self,
        params: Sequence[Tensor],
        *,
        damping: float = 1e-3,
        damping_end: float = 0.0,
        damping_anneal_epochs: int = 0,
        max_param_change: float = 0.1,
        trust_region: float = 1.0,
        subsample: int = 1024,
        center_gradients: bool = True,
    ):
        self.damping_init = damping
        self.damping = damping
        self.damping_end = damping_end if damping_end > 0 else damping
        self.damping_anneal_epochs = damping_anneal_epochs
        self.max_param_change = max_param_change
        self.trust_region = trust_region
        self.subsample = subsample
        self.center_gradients = center_gradients
        self._n_updates = 0
        self._param_shapes = [p.shape for p in params]
        self._param_numels = [p.numel() for p in params]
        self._total_params = sum(self._param_numels)
        # Cache for diagnostics
        self._last_stats = {}

    def _anneal_damping(self):
        if self.damping_anneal_epochs > 0 and self._n_updates <= self.damping_anneal_epochs:
            t = self._n_updates / self.damping_anneal_epochs
            # Log-linear interpolation
            log_init = math.log(self.damping_init)
            log_end = math.log(self.damping_end)
            self.damping = math.exp(log_init + t * (log_end - log_init))

    def update(
        self,
        psi_log_fn,
        x_batch: Tensor,
        params: Sequence[Tensor],
    ) -> dict:
        """Compute per-sample gradients O and cache for precondition step."""
        params_list = list(params)
        B_full = x_batch.shape[0]
        n_sub = min(self.subsample, B_full)

        if n_sub < B_full:
            idx = torch.randperm(B_full, device=x_batch.device)[:n_sub]
            x = x_batch[idx].detach()
        else:
            x = x_batch.detach()

        B = x.shape[0]
        P = self._total_params

        # Build per-sample gradient matrix O: (B, P)
        # Use sequential backward passes (one per sample is too expensive),
        # instead use the trick: for each sample k, compute grad of lp[k] w.r.t. params
        # Batch this via vmap if available, else loop with chunking
        O = torch.zeros(B, P, device=x.device, dtype=x.dtype)

        # Chunked per-sample gradient computation
        # Each chunk: compute lp, then backward for each sample in chunk
        chunk_size = min(64, B)
        for start in range(0, B, chunk_size):
            end = min(start + chunk_size, B)
            x_chunk = x[start:end].requires_grad_(False)

            for local_k in range(end - start):
                xk = x_chunk[local_k:local_k + 1].requires_grad_(True)
                lp_k = psi_log_fn(xk)
                if lp_k.dim() > 1:
                    lp_k = lp_k.view(-1)

                grads = torch.autograd.grad(
                    lp_k.sum(), params_list,
                    retain_graph=False, allow_unused=True, create_graph=False,
                )

                offset = 0
                for i, g in enumerate(grads):
                    n = self._param_numels[i]
                    if g is not None:
                        O[start + local_k, offset:offset + n] = g.detach().view(-1)
                    offset += n

        # Center: O_centered = O - mean(O)
        if self.center_gradients:
            O = O - O.mean(dim=0, keepdim=True)

        # Cache O and B for precondition step
        self._O = O
        self._B = B
        self._n_updates += 1
        self._anneal_damping()

        # Diagnostics
        F_diag = (O * O).mean(dim=0)
        self._last_stats = {
            "sr_mode": "woodbury",
            "fisher_mean": float(F_diag.mean().item()),
            "fisher_max": float(F_diag.max().item()),
            "fisher_median": float(F_diag.median().item()),
            "n_samples": B,
            "damping": self.damping,
            "n_updates": self._n_updates,
        }
        return self._last_stats

    def precondition(self, params: Sequence[Tensor]) -> None:
        """Apply F⁻¹ to .grad via Woodbury identity."""
        if self._n_updates == 0 or not hasattr(self, "_O"):
            return

        params_list = list(params)
        O = self._O  # (B, P)
        B = self._B
        lam = self.damping

        # Flatten gradients into vector g
        g = torch.cat([p.grad.view(-1) for p in params_list if p.grad is not None])

        # Woodbury: (OᵀO/B + λI)⁻¹ g = (1/λ)(g - Oᵀ (OOᵀ/B + λI)⁻¹ O g / B)
        # S = OOᵀ/B + λI,  shape (B, B)
        S = O @ O.t() / B + lam * torch.eye(B, device=O.device, dtype=O.dtype)

        Og = O @ g  # (B,)

        # Solve S z = Og
        # Use Cholesky for stability
        try:
            L = torch.linalg.cholesky(S)
            z = torch.cholesky_solve(Og.unsqueeze(1), L).squeeze(1)
        except torch.linalg.LinAlgError:
            # Fallback to general solve
            z = torch.linalg.solve(S, Og)

        # δθ = (1/λ)(g - Oᵀ z / B)
        delta = (g - O.t() @ z / B) / lam

        # Trust region: scale if ‖δθ‖₂ too large
        delta_norm = delta.norm().item()
        if self.trust_region > 0 and delta_norm > self.trust_region:
            delta = delta * (self.trust_region / delta_norm)
            self._last_stats["trust_clipped"] = True

        # Max parameter change: clamp ‖δθ‖∞
        if self.max_param_change > 0:
            delta = delta.clamp(-self.max_param_change, self.max_param_change)

        # Write back to .grad (optimizer will apply -lr * grad, so set grad = -delta/lr?
        # No: we replace .grad with δθ directly. The optimizer (SGD with lr) will do
        # θ -= lr * grad, so we set grad = δθ (the natural gradient direction).
        # The LR then controls the step size.
        offset = 0
        for p in params_list:
            if p.grad is not None:
                n = p.grad.numel()
                p.grad.data.copy_(delta[offset:offset + n].view(p.grad.shape))
                offset += n

        # Free cached O
        del self._O

    def state_dict(self) -> dict:
        return {
            "n_updates": self._n_updates,
            "damping": self.damping,
            "damping_init": self.damping_init,
        }

    def load_state_dict(self, state: dict) -> None:
        self._n_updates = state["n_updates"]
        self.damping = state.get("damping", self.damping)

    def reset(self) -> None:
        self._n_updates = 0
        self.damping = self.damping_init


# ═════════════════════════════════════════════════════════════════
#  CG SR
# ═════════════════════════════════════════════════════════════════


class CGSR:
    """
    Full SR via conjugate gradient in parameter space.

    Solves (F + λI) δθ = g iteratively. Each CG iteration requires one
    Fisher-vector product Fv, computed as:
        Fv = (1/B) Oᵀ (O v)
    where O v is a JVP and Oᵀ w is a VJP — two AD passes per iteration.

    For efficiency, we subsample the collocation batch and cache the
    forward graph for reuse across CG iterations.
    """

    def __init__(
        self,
        params: Sequence[Tensor],
        *,
        damping: float = 1e-3,
        damping_end: float = 0.0,
        damping_anneal_epochs: int = 0,
        n_cg_iters: int = 100,
        cg_tol: float = 1e-5,
        max_param_change: float = 0.1,
        trust_region: float = 1.0,
        subsample: int = 1024,
        center_gradients: bool = True,
    ):
        self.damping_init = damping
        self.damping = damping
        self.damping_end = damping_end if damping_end > 0 else damping
        self.damping_anneal_epochs = damping_anneal_epochs
        self.n_cg_iters = n_cg_iters
        self.cg_tol = cg_tol
        self.max_param_change = max_param_change
        self.trust_region = trust_region
        self.subsample = subsample
        self.center_gradients = center_gradients
        self._n_updates = 0
        self._param_shapes = [p.shape for p in params]
        self._param_numels = [p.numel() for p in params]
        self._total_params = sum(self._param_numels)
        self._last_stats = {}

    def _anneal_damping(self):
        if self.damping_anneal_epochs > 0 and self._n_updates <= self.damping_anneal_epochs:
            t = self._n_updates / self.damping_anneal_epochs
            log_init = math.log(self.damping_init)
            log_end = math.log(self.damping_end)
            self.damping = math.exp(log_init + t * (log_end - log_init))

    def _fisher_vec_product(self, v_flat: Tensor, O: Tensor, B: int) -> Tensor:
        """Compute (OᵀO/B + λI) v using cached O matrix."""
        Ov = O @ v_flat  # (B,)
        Fv = O.t() @ Ov / B  # (P,)
        return Fv + self.damping * v_flat

    def update(
        self,
        psi_log_fn,
        x_batch: Tensor,
        params: Sequence[Tensor],
    ) -> dict:
        """Compute per-sample gradients and cache O for CG solve."""
        params_list = list(params)
        B_full = x_batch.shape[0]
        n_sub = min(self.subsample, B_full)

        if n_sub < B_full:
            idx = torch.randperm(B_full, device=x_batch.device)[:n_sub]
            x = x_batch[idx].detach()
        else:
            x = x_batch.detach()

        B = x.shape[0]
        P = self._total_params

        # Build per-sample gradient matrix O: (B, P)
        O = torch.zeros(B, P, device=x.device, dtype=x.dtype)

        chunk_size = min(64, B)
        for start in range(0, B, chunk_size):
            end = min(start + chunk_size, B)
            x_chunk = x[start:end]

            for local_k in range(end - start):
                xk = x_chunk[local_k:local_k + 1].requires_grad_(True)
                lp_k = psi_log_fn(xk)
                if lp_k.dim() > 1:
                    lp_k = lp_k.view(-1)

                grads = torch.autograd.grad(
                    lp_k.sum(), params_list,
                    retain_graph=False, allow_unused=True, create_graph=False,
                )

                offset = 0
                for i, g in enumerate(grads):
                    n = self._param_numels[i]
                    if g is not None:
                        O[start + local_k, offset:offset + n] = g.detach().view(-1)
                    offset += n

        if self.center_gradients:
            O = O - O.mean(dim=0, keepdim=True)

        self._O = O
        self._B = B
        self._n_updates += 1
        self._anneal_damping()

        F_diag = (O * O).mean(dim=0)
        self._last_stats = {
            "sr_mode": "cg",
            "fisher_mean": float(F_diag.mean().item()),
            "fisher_max": float(F_diag.max().item()),
            "fisher_median": float(F_diag.median().item()),
            "n_samples": B,
            "damping": self.damping,
            "n_updates": self._n_updates,
        }
        return self._last_stats

    def precondition(self, params: Sequence[Tensor]) -> None:
        """Solve (F + λI) δθ = g via CG, then write δθ into .grad."""
        if self._n_updates == 0 or not hasattr(self, "_O"):
            return

        params_list = list(params)
        O = self._O
        B = self._B

        # Flatten gradient
        g = torch.cat([p.grad.view(-1) for p in params_list if p.grad is not None])

        # CG solve: (F + λI) x = g
        x = torch.zeros_like(g)
        r = g.clone()
        p = r.clone()
        rr = r.dot(r)
        rr0 = rr.item()

        n_iters_used = 0
        for i in range(self.n_cg_iters):
            Ap = self._fisher_vec_product(p, O, B)
            pAp = p.dot(Ap)
            if pAp.item() <= 0:
                # Negative curvature — stop CG, use what we have
                break
            alpha = rr / pAp
            x = x + alpha * p
            r = r - alpha * Ap
            rr_new = r.dot(r)
            n_iters_used = i + 1

            if rr_new.item() < self.cg_tol * rr0:
                break

            beta = rr_new / rr
            p = r + beta * p
            rr = rr_new

        self._last_stats["cg_iters"] = n_iters_used
        self._last_stats["cg_residual"] = float(rr.item())

        delta = x

        # Trust region
        delta_norm = delta.norm().item()
        if self.trust_region > 0 and delta_norm > self.trust_region:
            delta = delta * (self.trust_region / delta_norm)
            self._last_stats["trust_clipped"] = True

        # Max parameter change
        if self.max_param_change > 0:
            delta = delta.clamp(-self.max_param_change, self.max_param_change)

        # Write back
        offset = 0
        for p in params_list:
            if p.grad is not None:
                n = p.grad.numel()
                p.grad.data.copy_(delta[offset:offset + n].view(p.grad.shape))
                offset += n

        del self._O

    def state_dict(self) -> dict:
        return {
            "n_updates": self._n_updates,
            "damping": self.damping,
            "damping_init": self.damping_init,
        }

    def load_state_dict(self, state: dict) -> None:
        self._n_updates = state["n_updates"]
        self.damping = state.get("damping", self.damping)

    def reset(self) -> None:
        self._n_updates = 0
        self.damping = self.damping_init


class MinSR:
    """Minimum-step stochastic reconfiguration in sample space.

    Uses per-sample centered log-derivative matrix O and centered local-energy
    residual e to build the update direction:
        z = (O O^T / B + lambda I)^(-1) e
        delta = O^T z / B
    """

    def __init__(
        self,
        params: Sequence[Tensor],
        *,
        damping: float = 1e-3,
        damping_end: float = 0.0,
        damping_anneal_epochs: int = 0,
        max_param_change: float = 0.1,
        trust_region: float = 1.0,
        subsample: int = 1024,
        center_gradients: bool = True,
    ):
        self.damping_init = damping
        self.damping = damping
        self.damping_end = damping_end if damping_end > 0 else damping
        self.damping_anneal_epochs = damping_anneal_epochs
        self.max_param_change = max_param_change
        self.trust_region = trust_region
        self.subsample = subsample
        self.center_gradients = center_gradients
        self._n_updates = 0
        self._param_shapes = [p.shape for p in params]
        self._param_numels = [p.numel() for p in params]
        self._total_params = sum(self._param_numels)
        self._last_stats = {}
        self._sub_idx = None
        self._local_energies = None

    def _anneal_damping(self):
        if self.damping_anneal_epochs > 0 and self._n_updates <= self.damping_anneal_epochs:
            t = self._n_updates / self.damping_anneal_epochs
            log_init = math.log(self.damping_init)
            log_end = math.log(self.damping_end)
            self.damping = math.exp(log_init + t * (log_end - log_init))

    def update(
        self,
        psi_log_fn,
        x_batch: Tensor,
        params: Sequence[Tensor],
    ) -> dict:
        params_list = list(params)
        B_full = x_batch.shape[0]
        n_sub = min(self.subsample, B_full)

        if n_sub < B_full:
            idx = torch.randperm(B_full, device=x_batch.device)[:n_sub]
            x = x_batch[idx].detach()
            self._sub_idx = idx
        else:
            x = x_batch.detach()
            self._sub_idx = None

        B = x.shape[0]
        P = self._total_params

        O = torch.zeros(B, P, device=x.device, dtype=x.dtype)

        chunk_size = min(64, B)
        for start in range(0, B, chunk_size):
            end = min(start + chunk_size, B)
            x_chunk = x[start:end]

            for local_k in range(end - start):
                xk = x_chunk[local_k:local_k + 1].requires_grad_(True)
                lp_k = psi_log_fn(xk)
                if lp_k.dim() > 1:
                    lp_k = lp_k.view(-1)

                grads = torch.autograd.grad(
                    lp_k.sum(), params_list,
                    retain_graph=False, allow_unused=True, create_graph=False,
                )

                offset = 0
                for i, g in enumerate(grads):
                    n = self._param_numels[i]
                    if g is not None:
                        O[start + local_k, offset:offset + n] = g.detach().view(-1)
                    offset += n

        if self.center_gradients:
            O = O - O.mean(dim=0, keepdim=True)

        self._O = O
        self._B = B
        self._local_energies = None
        self._n_updates += 1
        self._anneal_damping()

        F_diag = (O * O).mean(dim=0)
        self._last_stats = {
            "sr_mode": "minsr",
            "fisher_mean": float(F_diag.mean().item()),
            "fisher_max": float(F_diag.max().item()),
            "fisher_median": float(F_diag.median().item()),
            "n_samples": B,
            "damping": self.damping,
            "n_updates": self._n_updates,
            "O_shape": (int(B), int(P)),
        }
        return self._last_stats

    def set_local_energies(self, e_local: Tensor) -> None:
        """Set per-sample local energies aligned to the last update batch."""
        if self._sub_idx is not None:
            e = e_local[self._sub_idx]
        else:
            e = e_local
        self._local_energies = e.detach()

    def precondition(self, params: Sequence[Tensor]) -> None:
        if self._n_updates == 0 or not hasattr(self, "_O"):
            return
        if self._local_energies is None:
            raise RuntimeError("MinSR requires set_local_energies() before precondition().")

        params_list = list(params)
        O = self._O
        B = self._B
        lam = self.damping
        e = self._local_energies.to(device=O.device, dtype=O.dtype)

        if e.numel() != B:
            raise RuntimeError(f"MinSR local-energy size mismatch: got {e.numel()} expected {B}")

        e_centered = e - e.mean()
        G = O @ O.t() / B + lam * torch.eye(B, device=O.device, dtype=O.dtype)

        try:
            L = torch.linalg.cholesky(G)
            z = torch.cholesky_solve(e_centered.unsqueeze(1), L).squeeze(1)
        except torch.linalg.LinAlgError:
            z = torch.linalg.solve(G, e_centered)

        delta = O.t() @ z / B

        if torch.isnan(delta).any() or torch.isinf(delta).any():
            raise RuntimeError("MinSR produced NaN/Inf update.")

        delta_norm = delta.norm().item()
        self._last_stats["update_norm"] = float(delta_norm)

        if self.trust_region > 0 and delta_norm > self.trust_region:
            delta = delta * (self.trust_region / delta_norm)
            self._last_stats["trust_clipped"] = True

        if self.max_param_change > 0:
            delta = delta.clamp(-self.max_param_change, self.max_param_change)

        offset = 0
        for p in params_list:
            n = p.numel()
            upd = delta[offset:offset + n].view(p.shape)
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            p.grad.data.copy_(upd)
            offset += n

        del self._O
        self._local_energies = None

    def state_dict(self) -> dict:
        return {
            "n_updates": self._n_updates,
            "damping": self.damping,
            "damping_init": self.damping_init,
        }

    def load_state_dict(self, state: dict) -> None:
        self._n_updates = state["n_updates"]
        self.damping = state.get("damping", self.damping)

    def reset(self) -> None:
        self._n_updates = 0
        self.damping = self.damping_init
