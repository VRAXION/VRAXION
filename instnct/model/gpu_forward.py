"""
GPU-accelerated forward pass for VRAXION spiking network.
Uses PyTorch for batched evaluation of multiple mutation candidates.

Key optimization: instead of 12 CPU workers each running 1 forward pass,
run all 12 mutations as a single batched GPU call.
"""

import torch
import numpy as np


# Import WAVE_LUT from graph.py
import os, sys
_model_dir = os.path.dirname(os.path.abspath(__file__))
if _model_dir not in sys.path:
    sys.path.insert(0, _model_dir)
from graph import SelfWiringGraph

WAVE_LUT_NP = SelfWiringGraph.WAVE_LUT  # (9, 8) float32
MAX_CHARGE = 15.0
DECAY_PERIOD = 6


class GPUForward:
    """Batched spiking network forward pass on GPU."""

    def __init__(self, H, device='cuda'):
        self.H = H
        self.device = torch.device(device)
        self.wave_lut = torch.tensor(WAVE_LUT_NP, dtype=torch.float32, device=self.device)

    def rollout_token(self, injected, rows, cols, theta, channel, polarity,
                      ticks=16, input_duration=2, state=None, charge=None):
        """Single forward pass (1 network, 1 token).

        Args:
            injected: (H,) float32 tensor — input activation
            rows, cols: (E,) int64 tensors — sparse edge list
            theta: (H,) float32 — firing thresholds
            channel: (H,) int64 — wave channel [1-8]
            polarity: (H,) float32 — +1/-1
            ticks: int
            input_duration: int
            state, charge: optional (H,) tensors to continue from

        Returns: (act, charge) tensors on GPU
        """
        H = self.H
        dev = self.device
        act = state.clone() if state is not None else torch.zeros(H, device=dev)
        cur_charge = charge.clone() if charge is not None else torch.zeros(H, device=dev)

        for tick in range(ticks):
            # 1. Decay
            if tick % DECAY_PERIOD == 0:
                cur_charge = torch.clamp(cur_charge - 1.0, min=0.0)
            # 2. Input
            if tick < input_duration:
                act = act + injected
            # 3. Propagate (scatter_add)
            raw = torch.zeros(H, device=dev)
            if len(rows) > 0:
                raw.scatter_add_(0, cols, act[rows])
            cur_charge = torch.clamp(cur_charge + raw, 0.0, MAX_CHARGE)
            # 4. Spike with wave gating
            theta_mult = self.wave_lut[channel, tick % 8]
            eff_theta = torch.clamp(theta * theta_mult, 1.0, MAX_CHARGE)
            fired = cur_charge >= eff_theta
            act = fired.float() * polarity
            # 5. Reset
            cur_charge[fired] = 0.0

        return act, cur_charge

    def rollout_token_batched(self, injected_batch, rows, cols, theta, channel,
                              polarity, ticks=16, input_duration=2,
                              state_batch=None, charge_batch=None):
        """Batched forward pass — B networks simultaneously.

        All networks share the same edge list (rows, cols) but can have
        different theta/channel/polarity. For mutation eval where only
        1 param differs per candidate, pass shared tensors.

        Args:
            injected_batch: (B, H) — B candidates
            rows, cols: (E,) — shared edge list
            theta: (H,) or (B, H) — thresholds
            channel: (H,) or (B, H) — channels
            polarity: (H,) or (B, H) — polarities
            state_batch, charge_batch: optional (B, H)

        Returns: (act_batch, charge_batch) as (B, H) tensors
        """
        B = injected_batch.shape[0]
        H = self.H
        dev = self.device

        act = state_batch.clone() if state_batch is not None else torch.zeros(B, H, device=dev)
        cur_charge = charge_batch.clone() if charge_batch is not None else torch.zeros(B, H, device=dev)

        # Expand theta/channel/polarity to (B, H) if needed
        if theta.dim() == 1:
            theta = theta.unsqueeze(0).expand(B, -1)
        if polarity.dim() == 1:
            polarity = polarity.unsqueeze(0).expand(B, -1)
        if channel.dim() == 1:
            channel = channel.unsqueeze(0).expand(B, -1)

        for tick in range(ticks):
            # 1. Decay
            if tick % DECAY_PERIOD == 0:
                cur_charge = torch.clamp(cur_charge - 1.0, min=0.0)
            # 2. Input
            if tick < input_duration:
                act = act + injected_batch
            # 3. Propagate — batched scatter_add
            raw = torch.zeros(B, H, device=dev)
            if len(rows) > 0:
                # act[:, rows] shape (B, E), scatter into raw[:, cols]
                src = act[:, rows]  # (B, E)
                raw.scatter_add_(1, cols.unsqueeze(0).expand(B, -1), src)
            cur_charge = torch.clamp(cur_charge + raw, 0.0, MAX_CHARGE)
            # 4. Spike with wave gating
            # channel is (B, H) int, need per-tick lookup
            theta_mult = self.wave_lut[channel, tick % 8]  # (B, H)
            eff_theta = torch.clamp(theta * theta_mult, 1.0, MAX_CHARGE)
            fired = cur_charge >= eff_theta
            act = fired.float() * polarity
            # 5. Reset
            cur_charge[fired] = 0.0

        return act, cur_charge

    def eval_accuracy_single(self, qdata_np, theta_np, channel_np, pol_f_np,
                             text_bytes, bp_in_np, bp_out_np,
                             in_dim, out_dim, ticks=16, input_duration=2):
        """Evaluate accuracy on a single network (convenience wrapper).

        All inputs are numpy, converted to GPU tensors internally.
        Returns: float accuracy [0, 1]
        """
        from quaternary_mask import QuaternaryMask
        H = self.H
        dev = self.device

        qm = QuaternaryMask(H, qdata_np)
        rows_np, cols_np = qm.to_directed_edges()
        rows = torch.tensor(rows_np, dtype=torch.long, device=dev)
        cols = torch.tensor(cols_np, dtype=torch.long, device=dev)
        theta = torch.tensor(theta_np, dtype=torch.float32, device=dev)
        channel = torch.tensor(channel_np, dtype=torch.long, device=dev)
        polarity = torch.tensor(pol_f_np, dtype=torch.float32, device=dev)
        bp_in = torch.tensor(bp_in_np, dtype=torch.float32, device=dev)
        bp_out = torch.tensor(bp_out_np, dtype=torch.float32, device=dev)

        state = torch.zeros(H, device=dev)
        charge = torch.zeros(H, device=dev)
        correct = 0; total = 0

        for i in range(len(text_bytes) - 1):
            inj = torch.zeros(H, device=dev)
            inj[:in_dim] = bp_in[text_bytes[i]]
            state, charge = self.rollout_token(
                inj, rows, cols, theta, channel, polarity,
                ticks=ticks, input_duration=input_duration,
                state=state, charge=charge)
            logits = bp_out @ charge[H - out_dim:]
            if torch.argmax(logits).item() == text_bytes[i + 1]:
                correct += 1
            total += 1

        return correct / total if total > 0 else 0.0
