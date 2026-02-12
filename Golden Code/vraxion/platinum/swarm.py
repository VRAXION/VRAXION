"""Prismion swarm — miniature AbsoluteHallway cells + Fibonacci budget.

A Prismion is a mini ring-memory cell with shared weights. The swarm
runs N Prismions in parallel (bank topology) or chained (loop topology).

Fibonacci halving budget allocates satellite ants at exponentially
decreasing fractions: 0.25, 0.125, 0.0625, ...
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


PHI = (1.0 + 5.0**0.5) / 2.0


def fibonacci_halving_budget(
    total_params: int,
    think_dim: int,
    vocab_size: int,
    *,
    in_dim: int | None = None,
    min_ring_len: int = 4,
    max_ants: int = 8,
) -> list:
    """Compute per-ant sizing for a Fibonacci-halving Prismion swarm.

    Each satellite ant gets a fraction ``0.5^(i+1)`` of the total budget.
    Returns a list of dicts with keys: ring_len, slot_dim, fraction, param_count.

    Args:
        in_dim: input dimension for the ants. If None, defaults to 2*think_dim
                (legacy chrom-fed behavior).
    """
    if in_dim is None:
        in_dim = 2 * think_dim
    head_cost = think_dim * vocab_size + vocab_size
    small_const = 4 + 1 + think_dim

    ants = []
    for i in range(max_ants):
        frac = 0.5 ** (i + 1)
        budget_i = int(total_params * frac)
        coeff = in_dim + 5 + 2 * think_dim + 1
        slot_dim = max(1, (budget_i - head_cost - small_const) // max(1, coeff))
        ring_len = max(min_ring_len, slot_dim // 2)
        if ring_len < min_ring_len:
            break
        if slot_dim < think_dim:
            break
        actual = (
            (in_dim * slot_dim + slot_dim)
            + (slot_dim + 1) * 4
            + ring_len * 2
            + 1
            + (2 * slot_dim * think_dim + think_dim)
            + head_cost
        )
        ants.append({
            "ring_len": int(ring_len),
            "slot_dim": int(slot_dim),
            "fraction": float(frac),
            "param_count": int(actual),
        })
    return ants


@dataclass
class PrismionState:
    """Runtime state for a Prismion (per sample)."""
    ring: torch.Tensor    # [B, L, S]
    ptr: torch.Tensor     # [B] float pointer in [0, L)
    h: torch.Tensor       # [B, S] internal hidden
    msg_prev: torch.Tensor  # [B, D]
    msg_ema: torch.Tensor   # [B, D]
    gain_prev: torch.Tensor # [B]
    gain_ema: torch.Tensor  # [B]


class Prismion(nn.Module):
    """Miniature AbsoluteHallway cell with ring memory + soft pointer.

    Communication: output is a projected summary of (read, write), optionally
    scaled by a learned gain.
    """

    def __init__(
        self,
        in_dim: int,
        msg_dim: int,
        ring_len: int,
        *,
        alpha: float = 1.0,
        msg_ema_beta: float = 0.9,
        gain_enabled: bool = False,
        slot_dim: Optional[int] = None,
        gauss_k: int = 1,
        gauss_tau: float = 2.0,
        context_scale_init: float = 0.2,
        ptr_dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.msg_dim = int(msg_dim)
        self.ring_len = int(max(1, ring_len))
        self.alpha = float(alpha)
        self.msg_ema_beta = float(msg_ema_beta)
        self.gain_enabled = bool(gain_enabled)
        self.gauss_k = int(max(1, gauss_k))
        self.gauss_tau = float(gauss_tau)
        self.ptr_dtype = ptr_dtype

        if slot_dim is None:
            slot_dim = int(max(self.msg_dim, round(float(self.msg_dim) * PHI)))
        self.slot_dim = int(max(1, slot_dim))

        self.input_proj = nn.Linear(self.in_dim, self.slot_dim)
        self.jump_score = nn.Linear(self.slot_dim, 1)
        self.inertia_head = nn.Linear(self.slot_dim, 1)
        self.deadzone_head = nn.Linear(self.slot_dim, 1)
        self.walk_head = nn.Linear(self.slot_dim, 1)

        self.theta_ptr = nn.Parameter(torch.zeros(int(self.ring_len)))
        self.theta_gate = nn.Parameter(torch.zeros(int(self.ring_len)))

        c0 = max(1e-3, min(0.999, float(context_scale_init)))
        logit = math.log(c0 / (1.0 - c0))
        self.context_logit = nn.Parameter(torch.tensor(logit, dtype=torch.float32))

        self.msg_proj = nn.Linear(2 * self.slot_dim, self.msg_dim)
        self.gain_head = nn.Linear(2 * self.slot_dim, 1) if self.gain_enabled else None

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.jump_score.weight)
        nn.init.zeros_(self.jump_score.bias)
        nn.init.xavier_uniform_(self.inertia_head.weight)
        nn.init.zeros_(self.inertia_head.bias)
        nn.init.xavier_uniform_(self.deadzone_head.weight)
        nn.init.zeros_(self.deadzone_head.bias)
        nn.init.xavier_uniform_(self.walk_head.weight)
        nn.init.zeros_(self.walk_head.bias)
        nn.init.zeros_(self.theta_ptr)
        nn.init.zeros_(self.theta_gate)
        nn.init.xavier_uniform_(self.msg_proj.weight)
        nn.init.zeros_(self.msg_proj.bias)
        if self.gain_head is not None:
            nn.init.xavier_uniform_(self.gain_head.weight)
            nn.init.zeros_(self.gain_head.bias)

    def init_state(self, batch: int, *, device: torch.device, dtype: torch.dtype) -> PrismionState:
        ring = torch.zeros(int(batch), self.ring_len, self.slot_dim, device=device, dtype=dtype)
        ptr = torch.zeros(int(batch), device=device, dtype=self.ptr_dtype)
        h = torch.zeros(int(batch), self.slot_dim, device=device, dtype=dtype)
        msg_prev = torch.zeros(int(batch), self.msg_dim, device=device, dtype=dtype)
        msg_ema = torch.zeros(int(batch), self.msg_dim, device=device, dtype=dtype)
        gain_prev = torch.ones(int(batch), device=device, dtype=dtype)
        gain_ema = torch.ones(int(batch), device=device, dtype=dtype)
        return PrismionState(
            ring=ring, ptr=ptr, h=h,
            msg_prev=msg_prev, msg_ema=msg_ema,
            gain_prev=gain_prev, gain_ema=gain_ema,
        )

    @staticmethod
    def _wrap_delta(a: torch.Tensor, b: torch.Tensor, ring_range: int) -> torch.Tensor:
        rr = float(ring_range)
        return torch.remainder(b - a + rr / 2.0, rr) - rr / 2.0

    @staticmethod
    def _circ_lerp(a: torch.Tensor, b: torch.Tensor, w: torch.Tensor, ring_range: int) -> torch.Tensor:
        return torch.remainder(a + w * Prismion._wrap_delta(a, b, ring_range), float(ring_range))

    def _compute_kernel_weights(
        self, ptr_float: torch.Tensor, *, ring_range: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rr = int(ring_range)
        offsets = torch.arange(-self.gauss_k, self.gauss_k + 1, device=ptr_float.device, dtype=ptr_float.dtype)
        base = torch.floor(ptr_float).long().clamp(0, rr - 1)
        centers = torch.remainder(base.unsqueeze(1) + offsets.to(torch.long).unsqueeze(0), rr)
        centers_f = centers.to(ptr_float.dtype)
        delta = self._wrap_delta(ptr_float.unsqueeze(1), centers_f, rr)
        d2 = delta ** 2
        tau = max(float(self.gauss_tau), 1e-4)
        logits = -d2 / tau
        weights = torch.softmax(logits, dim=1)
        return centers, weights

    def step(
        self,
        x: torch.Tensor,
        st: PrismionState,
        active_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, PrismionState]:
        if x.dim() != 2 or x.size(-1) != self.in_dim:
            raise ValueError(f"Prismion.step expects x [B,{self.in_dim}] got {tuple(x.shape)}")

        ring = st.ring
        ptr = st.ptr
        h = st.h

        B = int(ring.size(0))
        rr = int(self.ring_len)
        am_any = bool(active_mask.any())

        ptr = torch.nan_to_num(ptr, nan=0.0, posinf=float(rr - 1), neginf=0.0)
        ptr = torch.remainder(ptr, float(rr))
        pos_idx, weights = self._compute_kernel_weights(ptr, ring_range=rr)
        pos_idx_exp = pos_idx.unsqueeze(-1).expand(-1, -1, self.slot_dim).clamp(0, rr - 1)
        neigh = ring.gather(1, pos_idx_exp)
        cur = (weights.unsqueeze(-1) * neigh.to(weights.dtype)).sum(dim=1).to(ring.dtype)

        inp = self.input_proj(x.to(self.input_proj.weight.dtype))
        inp = torch.tanh(inp).to(cur.dtype)
        context_scale = torch.sigmoid(self.context_logit).to(inp.dtype)
        cue = inp + context_scale * cur

        h_new = torch.tanh(cue + h)
        if am_any:
            h = torch.where(active_mask.unsqueeze(1), h_new, h)

        upd = h
        a = float(self.alpha)
        a = max(0.0, min(1.0, a))
        if a != 1.0:
            upd = (1.0 - a) * cur + a * upd

        upd_exp = upd.unsqueeze(1).expand(-1, weights.size(1), -1)
        contrib = (weights.unsqueeze(-1) * upd_exp).to(ring.dtype)
        contrib = contrib * active_mask.view(B, 1, 1).to(contrib.dtype)
        ring_dtype = ring.dtype
        if ring_dtype in (torch.float16, torch.bfloat16):
            ring_fp32 = ring.float()
            ring_fp32 = ring_fp32.scatter_add(1, pos_idx_exp, contrib.float())
            ring = ring_fp32.to(ring_dtype)
        else:
            ring = ring.scatter_add(1, pos_idx_exp, contrib)

        prev_ptr = ptr.to(self.ptr_dtype)
        if am_any:
            inertia_use = torch.sigmoid(self.inertia_head(upd)).squeeze(1).to(self.ptr_dtype)
            deadzone_use = F.softplus(self.deadzone_head(upd)).squeeze(1).to(self.ptr_dtype)
            walk_use = torch.sigmoid(self.walk_head(upd)).squeeze(1).to(self.ptr_dtype)

            inertia_use = torch.clamp(inertia_use, 0.0, 0.99)
            deadzone_use = torch.clamp(deadzone_use, min=0.0)
            walk_use = torch.clamp(walk_use, 0.0, 1.0)

            ptr_base = torch.floor(prev_ptr).long().clamp(0, rr - 1)
            theta_ptr = torch.sigmoid(self.theta_ptr[ptr_base].to(self.ptr_dtype)) * float(rr - 1)
            theta_gate = self.theta_gate[ptr_base].to(upd.dtype)

            jump_logits = self.jump_score(upd).squeeze(1) + theta_gate
            p = torch.sigmoid(jump_logits).to(self.ptr_dtype)

            target_cont = theta_ptr
            target_ste = (target_cont.round() - target_cont).detach() + target_cont

            walk_ptr = torch.remainder(prev_ptr + 1.0, float(rr))
            stay_ptr = prev_ptr
            non_jump_ptr = self._circ_lerp(stay_ptr, walk_ptr, walk_use, rr)
            ptr_next = self._circ_lerp(non_jump_ptr, target_ste, p, rr)

            ptr_next = self._circ_lerp(prev_ptr, ptr_next, 1.0 - inertia_use, rr)

            delta_raw = self._wrap_delta(prev_ptr, ptr_next, rr)
            tau = 0.25
            move_mask = torch.sigmoid((delta_raw.abs() - deadzone_use) / max(tau, 1e-6))
            ptr_next = torch.remainder(prev_ptr + move_mask * delta_raw, float(rr))

            ptr = torch.where(active_mask, ptr_next, prev_ptr).to(self.ptr_dtype)
        else:
            ptr = prev_ptr.to(self.ptr_dtype)

        rw = torch.cat([cur, upd], dim=1).to(self.msg_proj.weight.dtype)
        msg = torch.tanh(self.msg_proj(rw)).to(cur.dtype)
        if self.gain_head is not None:
            gain = torch.sigmoid(self.gain_head(rw)).view(B).to(cur.dtype)
        else:
            gain = torch.ones(B, device=cur.device, dtype=cur.dtype)
        msg_out = msg * gain.unsqueeze(1)

        msg_prev = st.msg_prev
        msg_ema = st.msg_ema
        if msg_prev.dtype != msg_out.dtype:
            msg_prev = msg_prev.to(msg_out.dtype)
        if msg_ema.dtype != msg_out.dtype:
            msg_ema = msg_ema.to(msg_out.dtype)

        gain_prev = st.gain_prev
        gain_ema = st.gain_ema
        if gain_prev.dtype != gain.dtype:
            gain_prev = gain_prev.to(gain.dtype)
        if gain_ema.dtype != gain.dtype:
            gain_ema = gain_ema.to(gain.dtype)

        if am_any:
            am = active_mask.to(torch.bool).unsqueeze(1)
            msg_prev = torch.where(am, msg_out, msg_prev)
            b = max(0.0, min(1.0, float(self.msg_ema_beta)))
            msg_ema = torch.where(am, (b * msg_ema) + ((1.0 - b) * msg_out), msg_ema)
            gain_prev = torch.where(active_mask, gain, gain_prev)
            gain_ema = torch.where(active_mask, (b * gain_ema) + ((1.0 - b) * gain), gain_ema)

        st = PrismionState(
            ring=ring, ptr=ptr, h=h,
            msg_prev=msg_prev, msg_ema=msg_ema,
            gain_prev=gain_prev, gain_ema=gain_ema,
        )
        return msg_out, st
