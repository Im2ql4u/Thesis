"""
Diagonal Fisher Information Matrix preconditioner for natural gradient collocation.

The core problem: Adam normalizes parameter updates by the second moment of the
REINFORCE gradient, which conflates physics noise (Var(E_L)) with parameter-space
curvature (Fisher information). Near nodes and coalescence points, E_L blows up,
inflating Adam's denominator and suppressing informative gradient signal everywhere.

Natural gradient separates these:
    δθ = -(S + λI)^{-1} f
where S is the overlap/Fisher matrix and f is the force vector (gradient in
probability space). The Fisher captures model geometry; the force captures physics.

For ~50K parameters, the full Fisher is too large (50K × 50K). This module
implements the diagonal approximation:
    F_ii = E[(∂log Ψ / ∂θ_i)²]   (≈ Var when centered)

estimated cheaply via Hutchinson's trace estimator with Rademacher random vectors.

Key property: for Rademacher v ∈ {±1}^B,
    E_v[(Σ_k v_k · ∂lp_k/∂θ_i)²] = Σ_k (∂lp_k/∂θ_i)²
so each probe (one backward pass) gives an unbiased estimate of the diagonal.

Usage in training loop:
    fisher = DiagonalFisherPreconditioner(all_params, ...)
    for epoch in ...:
        # 1. Compute REINFORCE loss and backward (populates .grad)
        loss.backward()
        # 2. Update Fisher estimate from collocation batch
        fisher.update(psi_log_fn, X_batch, all_params)
        # 3. Precondition gradients in-place
        fisher.precondition(all_params)
        # 4. Clip and step
        clip_grad_norm_(all_params, max_norm)
        optimizer.step()  # SGD with momentum, NOT Adam
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn


class DiagonalFisherPreconditioner:
    """
    Diagonal Fisher preconditioner using Hutchinson trace estimation.

    Maintains an exponentially-weighted running estimate of the diagonal
    Fisher information matrix. On each call to `precondition`, divides
    parameter gradients by (F_diag + damping).

    Parameters
    ----------
    params : list[torch.Tensor]
        Trainable parameters (flat list, order must be consistent).
    damping : float
        Tikhonov regularization added to Fisher diagonal before inversion.
        Prevents division by near-zero Fisher values for weakly-observed params.
        Typical range: 1e-4 to 1e-2.
    ema_decay : float
        Exponential moving average decay for Fisher updates.
        0.95 means ~20 epochs of memory. Higher = more stable but slower to adapt.
    n_probes : int
        Number of Hutchinson probes per update. Each probe costs one backward
        pass through psi_log_fn → params. 4 probes give ~50% relative error
        on individual diagonal entries, but the EMA smoothing compensates.
    subsample : int
        Number of collocation points subsampled for Fisher estimation.
        The Fisher diagonal needs far fewer samples than the gradient itself
        (we're estimating P scalars, not a P-vector direction). 128-512 is
        typically sufficient.
    max_fisher : float
        Upper clamp on Fisher diagonal entries. Prevents a single parameter
        with huge Fisher from completely freezing its updates.
    """

    def __init__(
        self,
        params: Sequence[torch.Tensor],
        *,
        damping: float = 1e-3,
        ema_decay: float = 0.95,
        n_probes: int = 4,
        subsample: int = 256,
        max_fisher: float = 1e6,
    ):
        self.damping = damping
        self.ema_decay = ema_decay
        self.n_probes = n_probes
        self.subsample = subsample
        self.max_fisher = max_fisher
        self._n_updates = 0

        # Running Fisher estimate: one tensor per parameter, same shape
        self._fisher: list[torch.Tensor | None] = [None] * len(list(params))
        # Store param shapes for validation
        self._param_shapes = [p.shape for p in params]

    @torch.no_grad()
    def _validate_params(self, params: Sequence[torch.Tensor]) -> None:
        params_list = list(params)
        if len(params_list) != len(self._param_shapes):
            raise ValueError(
                f"Expected {len(self._param_shapes)} params, got {len(params_list)}"
            )
        for i, (p, s) in enumerate(zip(params_list, self._param_shapes)):
            if p.shape != s:
                raise ValueError(
                    f"Param {i} shape mismatch: expected {s}, got {p.shape}"
                )

    def update(
        self,
        psi_log_fn,
        x_batch: torch.Tensor,
        params: Sequence[torch.Tensor],
    ) -> dict:
        """
        Estimate Fisher diagonal from a subsample of collocation points.

        Performs n_probes Hutchinson probes, each requiring one backward pass.
        Updates the running EMA of the Fisher diagonal.

        Parameters
        ----------
        psi_log_fn : callable
            Maps (B, N, d) tensor → (B,) log|Ψ| values.
        x_batch : torch.Tensor
            Full collocation batch, shape (n_coll, N, d).
        params : list[torch.Tensor]
            Same parameter list as __init__.

        Returns
        -------
        dict with diagnostic info (fisher_mean, fisher_max, fisher_min, n_samples).
        """
        params_list = list(params)
        self._validate_params(params_list)

        B_full = x_batch.shape[0]
        n_sub = min(self.subsample, B_full)

        # Subsample collocation points (no replacement)
        if n_sub < B_full:
            idx = torch.randperm(B_full, device=x_batch.device)[:n_sub]
            x = x_batch[idx].detach().requires_grad_(True)
        else:
            x = x_batch.detach().requires_grad_(True)

        B = x.shape[0]

        # Forward pass — graph connects x → params → lp
        lp = psi_log_fn(x)  # (B,)
        if lp.dim() > 1:
            lp = lp.view(-1)

        # Accumulate sum of squared per-sample gradients via Hutchinson probes
        fisher_batch = [torch.zeros_like(p) for p in params_list]

        for probe_idx in range(self.n_probes):
            # Rademacher random vector: v_k ∈ {-1, +1}
            v = torch.empty(B, device=lp.device, dtype=lp.dtype)
            v.bernoulli_(0.5).mul_(2).sub_(1)

            # ∂/∂θ [Σ_k v_k lp_k] = Σ_k v_k (∂lp_k/∂θ)
            retain = probe_idx < self.n_probes - 1
            grads = torch.autograd.grad(
                (v * lp).sum(),
                params_list,
                retain_graph=retain,
                allow_unused=True,
                create_graph=False,
            )

            for i, g in enumerate(grads):
                if g is not None:
                    # (Σ_k v_k O_{k,i})² is unbiased estimate of Σ_k O_{k,i}²
                    fisher_batch[i].add_(g.detach().square())

        # Normalize: E[O_i²] ≈ (1/n_probes) (1/B) Σ_probes (Σ_k v_k O_{k,i})²
        # The Hutchinson identity gives Σ_k O_{k,i}² per probe, so divide by B
        # to get the empirical mean E[O_i²], and by n_probes to average probes.
        for fb in fisher_batch:
            fb.div_(self.n_probes * B)
            fb.clamp_(min=0.0, max=self.max_fisher)

        # EMA update
        self._n_updates += 1
        # Bias correction for early epochs (like Adam)
        bias_correction = 1.0 - self.ema_decay ** self._n_updates

        for i, p in enumerate(params_list):
            fb = fisher_batch[i]
            if self._fisher[i] is None:
                self._fisher[i] = fb.clone()
            else:
                self._fisher[i].mul_(self.ema_decay).add_(fb, alpha=1.0 - self.ema_decay)

        # Diagnostics
        all_fisher = torch.cat([f.view(-1) for f in self._fisher if f is not None])
        corrected = all_fisher / bias_correction
        return {
            "fisher_mean": float(corrected.mean().item()),
            "fisher_max": float(corrected.max().item()),
            "fisher_min": float(corrected.min().item()),
            "fisher_median": float(corrected.median().item()),
            "n_samples": B,
            "n_probes": self.n_probes,
            "n_updates": self._n_updates,
        }

    def precondition(self, params: Sequence[torch.Tensor]) -> None:
        """
        Divide .grad of each parameter by (F_ii + damping) in-place.

        Call this AFTER loss.backward() and BEFORE optimizer.step().
        """
        params_list = list(params)
        self._validate_params(params_list)

        if self._n_updates == 0:
            # No Fisher estimate yet — do not precondition
            return

        bias_correction = 1.0 - self.ema_decay ** self._n_updates

        for i, p in enumerate(params_list):
            if p.grad is None or self._fisher[i] is None:
                continue
            # Bias-corrected Fisher
            f_diag = self._fisher[i] / bias_correction
            # Precondition: g_nat = g / (F + λ)
            p.grad.data.div_(f_diag + self.damping)

    def state_dict(self) -> dict:
        """Serialize for checkpointing."""
        return {
            "fisher": [f.cpu().clone() if f is not None else None for f in self._fisher],
            "n_updates": self._n_updates,
            "damping": self.damping,
            "ema_decay": self.ema_decay,
            "n_probes": self.n_probes,
            "subsample": self.subsample,
            "max_fisher": self.max_fisher,
            "param_shapes": self._param_shapes,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore from checkpoint."""
        self._n_updates = state["n_updates"]
        self.damping = state.get("damping", self.damping)
        self.ema_decay = state.get("ema_decay", self.ema_decay)
        saved_fisher = state["fisher"]
        for i, f in enumerate(saved_fisher):
            if f is not None and i < len(self._fisher):
                self._fisher[i] = f.clone()

    def reset(self) -> None:
        """Reset running Fisher estimate (e.g. after architecture change)."""
        self._fisher = [None] * len(self._param_shapes)
        self._n_updates = 0
