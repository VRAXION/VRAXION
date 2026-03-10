"""Hessian-based saddle point detection and escape for VRAXION training.

Core idea:
    In high-dimensional loss landscapes, saddle points vastly outnumber local
    minima. Standard SGD can stall at these points because the gradient is
    near-zero but the curvature has negative directions.

Algorithm:
    1. Compute a small number of Hessian-vector products using random or
       Lanczos probe vectors (fixed compute budget per batch).
    2. Estimate the most negative eigenvalue via the Lanczos tridiagonal
       decomposition.
    3. If gradient norm is small AND a negative eigenvalue is found →
       we're at a saddle point → perturb along the negative eigenvector.
    4. Repeat across batches until the confidence interval on the
       minimum eigenvalue is tight enough (adaptive batch count).

The per-batch budget (number of Hessian-vector products) is FIXED.
Only the number of batches varies to achieve the desired confidence.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn


# ── Hessian-vector product ───────────────────────────────────

def hessian_vector_product(loss: torch.Tensor,
                           params: list[torch.Tensor],
                           vector: list[torch.Tensor]) -> list[torch.Tensor]:
    """Compute H @ v via two backward passes (Pearlmutter trick).

    Args:
        loss:   scalar loss (must have grad_fn — don't call .backward() first)
        params: list of parameter tensors
        vector: list of tensors same shape as params — the direction v

    Returns:
        list of tensors: H @ v, same shapes as params
    """
    # first backward: get gradients
    grads = torch.autograd.grad(loss, params, create_graph=True)

    # dot product g^T v
    gv = sum((g * v).sum() for g, v in zip(grads, vector))

    # second backward: d(g^T v)/d(params) = H @ v
    hvp = torch.autograd.grad(gv, params, retain_graph=True)

    return [h.detach() for h in hvp]


# ── Lanczos iteration ───────────────────────────────────────

def _flatten(tensors: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.reshape(-1) for t in tensors])


def _unflatten(flat: torch.Tensor, shapes: list[torch.Size]) -> list[torch.Tensor]:
    parts, offset = [], 0
    for s in shapes:
        n = 1
        for d in s:
            n *= d
        parts.append(flat[offset:offset + n].reshape(s))
        offset += n
    return parts


def lanczos_extreme_eigenvalue(
    loss: torch.Tensor,
    params: list[torch.Tensor],
    k: int = 10,
    seed: Optional[int] = None,
) -> tuple[float, list[torch.Tensor]]:
    """Lanczos iteration to estimate the most negative eigenvalue of the Hessian.

    Args:
        loss:   scalar loss with grad_fn
        params: model parameters
        k:      number of Lanczos steps (= Hessian-vector products = budget)
        seed:   optional RNG seed for the initial random vector

    Returns:
        (min_eigenvalue, eigenvector_as_param_list)
        The eigenvector corresponding to the most negative eigenvalue,
        split back into param-shaped tensors.
    """
    shapes = [p.shape for p in params]
    device = params[0].device
    n_total = sum(p.numel() for p in params)

    rng = torch.Generator(device=device)
    if seed is not None:
        rng.manual_seed(seed)

    # random starting vector, normalized
    q = torch.randn(n_total, device=device, generator=rng)
    q = q / q.norm()

    # Lanczos tridiagonal: alpha (diagonal), beta (off-diagonal)
    alphas = []
    betas = []
    Q = [q]  # orthonormal basis vectors

    for j in range(k):
        q_params = _unflatten(Q[j], shapes)
        hvp = hessian_vector_product(loss, params, q_params)
        w = _flatten(hvp)

        alpha_j = torch.dot(Q[j], w).item()
        alphas.append(alpha_j)

        # orthogonalize
        w = w - alpha_j * Q[j]
        if j > 0:
            w = w - betas[-1] * Q[j - 1]

        # full re-orthogonalization for numerical stability
        for qi in Q:
            w = w - torch.dot(qi, w) * qi

        beta_j = w.norm().item()

        if beta_j < 1e-12:
            # Krylov subspace exhausted early
            break

        betas.append(beta_j)
        Q.append(w / beta_j)

    # Build tridiagonal matrix and solve eigenvalue problem
    # m = number of completed Lanczos steps (alphas collected)
    # betas may be shorter if we broke early on the last iteration
    m = len(alphas)
    T = torch.zeros(m, m)
    for i in range(m):
        T[i, i] = alphas[i]
    n_betas = min(len(betas), m - 1)
    for i in range(n_betas):
        T[i, i + 1] = betas[i]
        T[i + 1, i] = betas[i]

    eigenvalues, eigenvectors = torch.linalg.eigh(T)

    # most negative eigenvalue
    min_idx = 0
    min_eval = eigenvalues[min_idx].item()

    # map Lanczos eigenvector back to parameter space
    ritz_vec = eigenvectors[:, min_idx]
    Q_matrix = torch.stack(Q[:m])  # (m, n_total)
    full_evec = ritz_vec @ Q_matrix  # (n_total,)
    full_evec = full_evec / full_evec.norm()

    return min_eval, _unflatten(full_evec, shapes)


# ── Saddle point detector ───────────────────────────────────

@dataclass
class SaddleCheckResult:
    """Result of a saddle point check."""
    is_saddle: bool
    min_eigenvalue: float
    grad_norm: float
    escape_direction: Optional[list[torch.Tensor]] = None
    confidence_interval: tuple[float, float] = (0.0, 0.0)
    n_batches_used: int = 0


@dataclass
class HessianSaddleConfig:
    """Configuration for saddle point detection and escape.

    Attributes:
        lanczos_steps:       Fixed budget of Hessian-vector products per batch.
        grad_threshold:      Below this gradient norm, we suspect a critical point.
        eigenvalue_threshold: If min eigenvalue < this, it's a saddle (not a min).
        max_batches:         Maximum batches to check before giving up.
        confidence_width:    Target half-width of the CI on the min eigenvalue.
        escape_lr:           Step size along the negative eigenvector.
        confidence_level:    Z-score multiplier for the CI (1.96 = 95%).
    """
    lanczos_steps: int = 10
    grad_threshold: float = 1e-3
    eigenvalue_threshold: float = -1e-4
    max_batches: int = 20
    confidence_width: float = 0.05
    escape_lr: float = 0.01
    confidence_level: float = 1.96


class HessianSaddleOptimizer:
    """Wraps a base optimizer with Hessian-based saddle point escape.

    Usage:
        optimizer = HessianSaddleOptimizer(
            model=model,
            base_optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
            loss_fn=my_loss_fn,
            config=HessianSaddleConfig(),
        )

        for batch in dataloader:
            loss = optimizer.step(batch)
            # The optimizer handles saddle point detection internally.
    """

    def __init__(
        self,
        model: nn.Module,
        base_optimizer: torch.optim.Optimizer,
        loss_fn,  # callable(model, batch) -> scalar loss
        config: Optional[HessianSaddleConfig] = None,
    ):
        self.model = model
        self.base_optimizer = base_optimizer
        self.loss_fn = loss_fn
        self.config = config or HessianSaddleConfig()
        self.params = [p for p in model.parameters() if p.requires_grad]

        # stats
        self.saddle_escapes = 0
        self.total_checks = 0
        self._eigenvalue_history: list[float] = []

    def check_saddle(self, batch) -> SaddleCheckResult:
        """Run saddle point detection on the current parameter state.

        Uses multiple batches (each with fixed Lanczos budget) until the
        confidence interval on the minimum eigenvalue is tight enough.
        """
        cfg = self.config
        self.total_checks += 1

        # First check: is gradient small enough to suspect a critical point?
        self.model.zero_grad()
        loss = self.loss_fn(self.model, batch)
        loss.backward()
        grad_norm = self._grad_norm()

        if grad_norm > cfg.grad_threshold:
            return SaddleCheckResult(
                is_saddle=False,
                min_eigenvalue=float('nan'),
                grad_norm=grad_norm,
                n_batches_used=0,
            )

        # Gradient is small → probe Hessian across batches
        eigenvalue_samples = []

        for batch_idx in range(cfg.max_batches):
            self.model.zero_grad()
            probe_loss = self.loss_fn(self.model, batch)

            min_eval, escape_dir = lanczos_extreme_eigenvalue(
                probe_loss, self.params, k=cfg.lanczos_steps,
                seed=batch_idx * 137,  # different seed per batch
            )
            eigenvalue_samples.append(min_eval)

            # Check confidence interval
            if len(eigenvalue_samples) >= 2:
                mean_eval = sum(eigenvalue_samples) / len(eigenvalue_samples)
                std_eval = (
                    sum((x - mean_eval) ** 2 for x in eigenvalue_samples)
                    / (len(eigenvalue_samples) - 1)
                ) ** 0.5
                se = std_eval / math.sqrt(len(eigenvalue_samples))
                half_width = cfg.confidence_level * se

                if half_width < cfg.confidence_width:
                    # Confidence interval is tight enough
                    ci = (mean_eval - half_width, mean_eval + half_width)
                    is_saddle = mean_eval < cfg.eigenvalue_threshold

                    result = SaddleCheckResult(
                        is_saddle=is_saddle,
                        min_eigenvalue=mean_eval,
                        grad_norm=grad_norm,
                        confidence_interval=ci,
                        n_batches_used=batch_idx + 1,
                    )
                    if is_saddle:
                        result.escape_direction = escape_dir
                    return result

        # Ran out of batches — use best estimate
        mean_eval = sum(eigenvalue_samples) / len(eigenvalue_samples)
        std_eval = (
            sum((x - mean_eval) ** 2 for x in eigenvalue_samples)
            / (len(eigenvalue_samples) - 1)
        ) ** 0.5 if len(eigenvalue_samples) > 1 else float('inf')
        se = std_eval / math.sqrt(len(eigenvalue_samples)) if eigenvalue_samples else float('inf')
        half_width = cfg.confidence_level * se
        ci = (mean_eval - half_width, mean_eval + half_width)
        is_saddle = mean_eval < cfg.eigenvalue_threshold

        result = SaddleCheckResult(
            is_saddle=is_saddle,
            min_eigenvalue=mean_eval,
            grad_norm=grad_norm,
            confidence_interval=ci,
            n_batches_used=cfg.max_batches,
        )
        if is_saddle:
            result.escape_direction = escape_dir
        return result

    def escape_saddle(self, result: SaddleCheckResult):
        """Perturb parameters along the most negative eigenvector direction."""
        if not result.is_saddle or result.escape_direction is None:
            return

        with torch.no_grad():
            for p, d in zip(self.params, result.escape_direction):
                # Step in the direction that decreases the loss
                # (negative eigenvector = direction of negative curvature)
                p.add_(d, alpha=self.config.escape_lr)

        self.saddle_escapes += 1
        self._eigenvalue_history.append(result.min_eigenvalue)

    def step(self, batch) -> dict:
        """One optimization step with saddle point awareness.

        Returns a dict with step info for logging.
        """
        # Normal gradient step
        self.model.zero_grad()
        loss = self.loss_fn(self.model, batch)
        loss.backward()
        grad_norm = self._grad_norm()
        self.base_optimizer.step()

        info = {
            'loss': loss.item(),
            'grad_norm': grad_norm,
            'saddle_detected': False,
            'saddle_escaped': False,
        }

        # Check for saddle point if gradient is suspiciously small
        if grad_norm < self.config.grad_threshold:
            result = self.check_saddle(batch)
            info['saddle_detected'] = result.is_saddle
            info['min_eigenvalue'] = result.min_eigenvalue
            info['ci'] = result.confidence_interval
            info['n_batches'] = result.n_batches_used

            if result.is_saddle:
                self.escape_saddle(result)
                info['saddle_escaped'] = True

        return info

    def _grad_norm(self) -> float:
        total = 0.0
        for p in self.params:
            if p.grad is not None:
                total += p.grad.data.norm().item() ** 2
        return total ** 0.5

    def stats(self) -> dict:
        return {
            'total_checks': self.total_checks,
            'saddle_escapes': self.saddle_escapes,
            'eigenvalue_history': list(self._eigenvalue_history),
        }
