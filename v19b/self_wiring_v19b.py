"""
VRAXION v19b — Self-Wiring Graph Network with Reward-Modulated Target Learning
===============================================================================
NO mutation. NO backprop. Pure self-wiring + reward-modulated learning.

Key features:
- Binary activation with per-neuron learnable threshold (neuromodulation)
- 3D+1D address initialization (spatial + functional type)
- Reward-modulated target_W learning (per-sample reward drives wiring)
- Per-output reward-modulated Hebbian/anti-Hebbian weight learning
- Threshold neuromodulation (active neurons adjust sensitivity based on reward)
- 32-class A→B association task, 160 neurons

This is the critical test: can the network learn ENTIRELY through self-wiring
guided by a global reward signal, without any mutation?

Core learning mechanism:
- Self-wiring proposes new topology (exploration)
- Weight learning steers signal to correct outputs (exploitation)
- Threshold modulation controls neuron selectivity
- All driven by per-sample reward: did THIS input map to correct output?

Author: Daniel (researcher) + Claude (advisor)
Date: 2026-03-11
"""

import torch
import time
import math
import random
from collections import defaultdict


# =============================================================================
# Network Monitor — tracks internal dynamics
# =============================================================================

class NetworkMonitor:
    """Tracks all internal dynamics of the self-wiring network."""

    def __init__(self, log_interval=500):
        self.log_interval = log_interval
        self.history = defaultdict(list)
        self.step = 0

    def log(self, net, extra=None):
        self.step += 1
        if self.step % self.log_interval != 0:
            return

        mask = net.mask
        n = net.n_neurons
        n_conns = mask.sum().item()
        density = n_conns / (n * n)

        in_deg = mask.sum(dim=0).float()
        out_deg = mask.sum(dim=1).float()
        max_in = in_deg.max().item()
        max_out = out_deg.max().item()
        avg_deg = n_conns / n if n > 0 else 0
        hubs_in = (in_deg > 2 * avg_deg).sum().item()
        hubs_out = (out_deg > 2 * avg_deg).sum().item()

        sw_conns = net.conn_origin.sum().item()
        total = n_conns

        ages = net.conn_age[mask.bool()]
        if len(ages) > 0:
            mean_age = ages.float().mean().item()
            max_age = ages.max().item()
            young = (ages < 100).sum().item()
            old = (ages >= 100).sum().item()
        else:
            mean_age = max_age = young = old = 0

        activated = net._last_activations
        active_frac = (activated > 0).float().mean().item()
        addr_range = net.addresses.max(dim=0).values - net.addresses.min(dim=0).values
        target_mag = net.target_W.abs().mean().item()
        thresh_mean = net.thresholds.mean().item()
        thresh_std = net.thresholds.std().item()
        thresh_min = net.thresholds.min().item()
        thresh_max = net.thresholds.max().item()

        sw_proposals = net._sw_proposals
        sw_accepts = net._sw_accepts
        accept_rate = sw_accepts / max(sw_proposals, 1)
        rw_positive = net._reward_positive
        rw_negative = net._reward_negative
        rw_neutral = net._reward_neutral

        active_W = net.W[mask.bool()]
        if len(active_W) > 0:
            w_mean = active_W.mean().item()
            w_std = active_W.std().item()
            w_pos = (active_W > 0).sum().item()
            w_neg = (active_W < 0).sum().item()
        else:
            w_mean = w_std = 0
            w_pos = w_neg = 0

        print(f"\n{'='*70}")
        print(f"[MONITOR] Step {self.step}")
        print(f"{'='*70}")
        print(f"[GRAPH]  conns={n_conns:.0f}  density={density:.4f}  "
              f"max_in={max_in:.0f}  max_out={max_out:.0f}  "
              f"hubs_in={hubs_in}  hubs_out={hubs_out}")
        print(f"[ORIGIN] self-wired={sw_conns:.0f}/{total:.0f} "
              f"({sw_conns/max(total,1)*100:.1f}%)")
        print(f"[WIRE]   proposals={sw_proposals}  accepts={sw_accepts}  "
              f"rate={accept_rate:.3f}")
        print(f"[AGE]    mean={mean_age:.1f}  max={max_age:.0f}  "
              f"young(<100)={young}  old(>=100)={old}")
        print(f"[ACTIV]  sparsity={1-active_frac:.3f}  "
              f"active_frac={active_frac:.3f}")
        print(f"[ADDR]   range=[{addr_range[0]:.2f},{addr_range[1]:.2f},"
              f"{addr_range[2]:.2f},{addr_range[3]:.2f}]  "
              f"target_mag={target_mag:.4f}")
        print(f"[THRESH] mean={thresh_mean:.3f}  std={thresh_std:.3f}  "
              f"min={thresh_min:.3f}  max={thresh_max:.3f}")
        print(f"[WEIGHT] mean={w_mean:.3f}  std={w_std:.3f}  "
              f"pos={w_pos}  neg={w_neg}")
        print(f"[REWARD] positive={rw_positive}  negative={rw_negative}  "
              f"neutral={rw_neutral}")
        if extra:
            for k, v in extra.items():
                print(f"[EXTRA]  {k}={v}")
        print(f"{'='*70}")

        self.history['step'].append(self.step)
        self.history['n_conns'].append(n_conns)
        self.history['density'].append(density)
        self.history['sw_rate'].append(accept_rate)
        self.history['active_frac'].append(active_frac)
        self.history['target_mag'].append(target_mag)
        self.history['thresh_mean'].append(thresh_mean)
        self.history['thresh_std'].append(thresh_std)


# =============================================================================
# v19b Self-Wiring Graph Network
# =============================================================================

class SelfWiringNetV19b:
    """
    Self-wiring graph network with reward-modulated target learning.

    NO mutation. NO backprop. The network learns ONLY through:
    1. Self-wiring: active neurons propose new connections (topology exploration)
    2. Reward-modulated weight learning: per-output Hebbian/anti-Hebbian
    3. Reward-modulated target_W: steers WHERE self-wiring reaches
    4. Threshold neuromodulation: controls neuron selectivity

    The key learning mechanism: for each sample, strengthen weights on paths
    to the correct output and weaken weights on paths to the wrong (predicted)
    output. This creates input-specific routing through weight differentiation
    even though topology is shared.
    """

    def __init__(self, n_neurons=160, n_in=32, n_out=32, n_ticks=8,
                 decay=0.5, top_k_wire=5, target_lr=0.01, threshold_lr=0.003,
                 weight_lr=0.015, device='cpu'):
        self.n_neurons = n_neurons
        self.n_in = n_in
        self.n_out = n_out
        self.n_ticks = n_ticks
        self.decay = decay
        self.top_k_wire = top_k_wire
        self.target_lr = target_lr
        self.threshold_lr = threshold_lr
        self.weight_lr = weight_lr
        self.device = device

        # --- Neuron properties ---
        self.addresses = self._init_3d_1d_addresses()
        self.target_W = torch.randn(n_neurons, 4, device=device) * 0.2
        self.thresholds = torch.full((n_neurons,), 0.3, device=device)

        # --- Connection structure ---
        self.mask = torch.zeros(n_neurons, n_neurons, device=device)
        self._init_connections()

        # Weight matrix: larger init for signal propagation, biased positive
        self.W = torch.randn(n_neurons, n_neurons, device=device) * 0.3 + 0.2
        self.W *= self.mask

        # --- Tracking ---
        self.conn_origin = torch.zeros(n_neurons, n_neurons, device=device)
        self.conn_age = torch.zeros(n_neurons, n_neurons, dtype=torch.long,
                                    device=device)
        self.state = torch.zeros(n_neurons, device=device)
        self._sw_proposals = 0
        self._sw_accepts = 0
        self._reward_positive = 0
        self._reward_negative = 0
        self._reward_neutral = 0
        self._last_activations = torch.zeros(n_neurons, device=device)
        # Per-tick activation history for credit assignment
        self._tick_activations = []

    def _init_3d_1d_addresses(self):
        """4D addresses: dims 0-2 spatial random, dim 3 functional type."""
        n = self.n_neurons
        addresses = torch.zeros(n, 4, device=self.device)
        addresses[:, :3] = torch.rand(n, 3, device=self.device)
        for i in range(n):
            if i < self.n_in:
                addresses[i, 3] = 0.0
            elif i >= n - self.n_out:
                addresses[i, 3] = 1.0
            else:
                addresses[i, 3] = 0.5
        return addresses

    def _init_connections(self):
        """
        Sparse seed connections ensuring paths from every input to every output.
        Each input → 3 random internals, each internal → 2 others + 2 outputs.
        Plus 2% random diversity.
        """
        n = self.n_neurons
        n_in = self.n_in
        n_out = self.n_out
        internal_start = n_in
        internal_end = n - n_out
        n_internal = internal_end - internal_start

        if n_internal == 0:
            return

        # Input → 3 random internals
        for i in range(n_in):
            k = min(3, n_internal)
            targets = random.sample(range(internal_start, internal_end), k)
            for j in targets:
                self.mask[i, j] = 1.0

        # Internal → 2 other internals + 2 outputs
        for i in range(internal_start, internal_end):
            others = [x for x in range(internal_start, internal_end) if x != i]
            if others:
                for j in random.sample(others, min(2, len(others))):
                    self.mask[i, j] = 1.0
            # 2 random outputs
            for _ in range(2):
                out_j = random.randint(n - n_out, n - 1)
                self.mask[i, out_j] = 1.0

        # 2% random
        n_extra = int(n * n * 0.02)
        for _ in range(n_extra):
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i != j:
                self.mask[i, j] = 1.0

    def forward(self, input_vec):
        """Forward pass: inject input every tick (sustained), propagate, read output."""
        self.state = torch.zeros(self.n_neurons, device=self.device)

        effective_W = self.W * self.mask
        self._tick_activations = []

        for tick in range(self.n_ticks):
            # Sustained input: re-inject every tick so input neurons stay active
            self.state[:self.n_in] = torch.max(self.state[:self.n_in], input_vec)

            activated = (self.state > self.thresholds).float()
            self._tick_activations.append(activated.clone())
            incoming = activated @ effective_W
            self.state = self.state * self.decay + incoming

        self._last_activations = (self.state > self.thresholds).float()
        output = self.state[-self.n_out:]
        return output

    def self_wire(self):
        """
        Self-wiring: top-k most active neurons propose new connections.
        Each computes target = address + activation * target_W.
        """
        activated = self._last_activations
        active_mask = activated > 0
        n_active = int(active_mask.sum().item())
        if n_active == 0:
            return []

        active_states = self.state.abs() * active_mask.float()
        k = min(self.top_k_wire, n_active)
        _, top_indices = torch.topk(active_states, k)

        new_connections = []
        for idx in top_indices:
            i = idx.item()
            self._sw_proposals += 1
            target_addr = self.addresses[i] + activated[i] * self.target_W[i]
            dists = torch.norm(self.addresses - target_addr.unsqueeze(0), dim=1)
            dists[i] = float('inf')

            for _ in range(5):
                j = dists.argmin().item()
                if self.mask[i, j] == 0:
                    self.mask[i, j] = 1.0
                    self.W[i, j] = torch.randn(1, device=self.device).item() * 0.1
                    self.conn_origin[i, j] = 1.0
                    self.conn_age[i, j] = 0
                    new_connections.append((i, j))
                    self._sw_accepts += 1
                    break
                else:
                    dists[j] = float('inf')

        return new_connections

    def reward_update(self, target_idx, predicted_idx, new_connections):
        """
        Reward-modulated learning based on per-sample correctness.

        Core mechanism — per-output credit assignment:
        - For CORRECT output neuron: strengthen weights from active pre-synaptic
          neurons TO this output (reward pathway)
        - For WRONG predicted output: weaken weights from active pre-synaptic
          neurons TO this output (punishment pathway)
        - This creates input-specific weight patterns without backprop

        Also updates:
        - target_W for self-wiring neurons (three-factor Hebbian)
        - Thresholds via neuromodulation
        """
        correct = (target_idx == predicted_idx)

        if correct:
            self._reward_positive += 1
            reward = 1.0
        else:
            self._reward_negative += 1
            reward = -1.0

        activated = self._last_activations
        n = self.n_neurons
        n_out = self.n_out
        out_start = n - n_out

        # --- Per-output weight adjustment (THE key mechanism) ---
        if not correct:
            # Get which neurons were active (potential pre-synaptic sources)
            active_indices = (activated > 0).nonzero(as_tuple=True)[0]

            # STRENGTHEN paths to correct output
            target_neuron = out_start + target_idx
            for pre_idx in active_indices:
                i = pre_idx.item()
                if self.mask[i, target_neuron] > 0:
                    # Existing connection: make more excitatory
                    self.W[i, target_neuron] += self.weight_lr

            # WEAKEN paths to wrong predicted output
            pred_neuron = out_start + predicted_idx
            for pre_idx in active_indices:
                i = pre_idx.item()
                if self.mask[i, pred_neuron] > 0:
                    # Existing connection: make less excitatory / more inhibitory
                    self.W[i, pred_neuron] -= self.weight_lr

            # Also adjust weights deeper in the network using tick activations
            # Trace back: which internals were active and connected to active neurons
            # that connected to the target/predicted outputs?
            if len(self._tick_activations) >= 2:
                late_act = self._tick_activations[-1]  # near-output activation
                early_act = self._tick_activations[len(self._tick_activations)//2]  # mid

                # Strengthen mid→late connections where late feeds target
                for pre_idx in (early_act > 0).nonzero(as_tuple=True)[0]:
                    i = pre_idx.item()
                    for post_idx in (late_act > 0).nonzero(as_tuple=True)[0]:
                        j = post_idx.item()
                        if self.mask[i, j] > 0 and self.mask[j, target_neuron] > 0:
                            self.W[i, j] += self.weight_lr * 0.1
                        if self.mask[i, j] > 0 and self.mask[j, pred_neuron] > 0:
                            self.W[i, j] -= self.weight_lr * 0.1

        # Small weight decay to prevent runaway (shrink toward zero)
        self.W *= 0.9999
        # Clamp weights
        self.W.clamp_(-2.0, 2.0)
        self.W *= self.mask

        # --- Target_W update for wiring direction ---
        for (i, j) in new_connections:
            direction = self.addresses[j] - self.addresses[i]
            direction = direction / (direction.norm() + 1e-8)
            self.target_W[i] += self.target_lr * activated[i] * direction * reward

        # --- Threshold neuromodulation ---
        active_mask = activated > 0
        if active_mask.any():
            if correct:
                # Good outcome: active neurons lower threshold (more sensitive)
                self.thresholds[active_mask] -= self.threshold_lr
            else:
                # Bad outcome: active neurons raise threshold (more selective)
                self.thresholds[active_mask] += self.threshold_lr * 0.5
            self.thresholds.clamp_(0.01, 5.0)

    def remove_connection(self, i, j):
        self.mask[i, j] = 0.0
        self.W[i, j] = 0.0
        self.conn_origin[i, j] = 0.0
        self.conn_age[i, j] = 0

    def prune_dead_connections(self, max_age=10000, min_conns=100):
        n_conns = self.mask.sum().item()
        if n_conns <= min_conns:
            return 0
        old = (self.conn_age > max_age) & self.mask.bool()
        n_pruned = min(int(old.sum().item()), int(n_conns - min_conns))
        if n_pruned > 0:
            old_ages = self.conn_age.clone()
            old_ages[~old] = 0
            flat = old_ages.flatten()
            _, prune_idx = torch.topk(flat, n_pruned)
            rows = prune_idx // self.n_neurons
            cols = prune_idx % self.n_neurons
            for r, c in zip(rows.tolist(), cols.tolist()):
                self.remove_connection(r, c)
        return n_pruned

    def age_connections(self):
        self.conn_age += self.mask.long()


# =============================================================================
# Task
# =============================================================================

def make_association_task(n_classes=32, device='cpu'):
    indices = list(range(n_classes))
    targets = indices.copy()
    random.shuffle(targets)
    pairs = []
    mapping = {}
    for a_idx, b_idx in zip(indices, targets):
        inp = torch.zeros(n_classes, device=device)
        inp[a_idx] = 1.0
        pairs.append((inp, b_idx))
        mapping[a_idx] = b_idx
    return pairs, mapping


def evaluate_accuracy(net, pairs):
    correct = 0
    for inp, target in pairs:
        output = net.forward(inp)
        if output.argmax().item() == target:
            correct += 1
    return correct / len(pairs)


# =============================================================================
# Main training loop — Per-Sample Reward with Credit Assignment
# =============================================================================

def train_v19b(n_classes=32, n_neurons=160, max_attempts=200_000,
               log_interval=500, target_acc=1.0, device='cpu'):
    """
    Train v19b with per-sample reward + per-output credit assignment.

    Each step:
    1. Pick sample (input_a, target_b)
    2. Forward pass → output, activations
    3. Self-wire based on activations (topology exploration)
    4. Reward update: strengthen path to target, weaken path to prediction
    5. Threshold neuromodulation
    """
    print("=" * 70)
    print("VRAXION v19b — Self-Wiring with Reward-Modulated Target Learning")
    print("=" * 70)
    print(f"Config: {n_classes} classes, {n_neurons} neurons, {8} ticks")
    print(f"Activation: binary, learnable threshold (init=0.3, [0.01, 5.0])")
    print(f"Addresses: 3D+1D (spatial + functional type)")
    print(f"Learning: per-sample per-output reward-modulated Hebbian")
    print(f"  target_W lr={0.01}, weight lr={0.015}, threshold lr={0.003}")
    print(f"Mutation: NONE")
    print("=" * 70)

    net = SelfWiringNetV19b(
        n_neurons=n_neurons, n_in=n_classes, n_out=n_classes,
        n_ticks=8, decay=0.5, top_k_wire=5,
        target_lr=0.01, threshold_lr=0.003, weight_lr=0.015,
        device=device,
    )

    pairs, mapping = make_association_task(n_classes, device)
    print(f"\nMapping (first 8): {dict(list(mapping.items())[:8])}")

    monitor = NetworkMonitor(log_interval=log_interval)

    acc = evaluate_accuracy(net, pairs)
    best_acc = acc
    best_acc_step = 0
    stale_count = 0
    stale_limit = 30_000

    print(f"\nInitial accuracy: {acc*100:.1f}%")
    print(f"Initial connections: {net.mask.sum().item():.0f}")
    print(f"\nStarting per-sample reward training...\n")

    start_time = time.time()

    for attempt in range(1, max_attempts + 1):
        # Pick random sample
        inp, target = random.choice(pairs)

        # Forward pass
        output = net.forward(inp)
        predicted = output.argmax().item()

        # Self-wire (topology exploration)
        new_conns = net.self_wire()

        # Reward-modulated learning (weight adjustment + target_W + thresholds)
        net.reward_update(target, predicted, new_conns)

        # Age connections
        net.age_connections()

        # Pruning
        if attempt % 5000 == 0:
            net.prune_dead_connections(max_age=10000)

        # Periodic full accuracy
        if attempt % 100 == 0:
            acc = evaluate_accuracy(net, pairs)
            if acc > best_acc:
                best_acc = acc
                best_acc_step = attempt
                stale_count = 0
            else:
                stale_count += 100

        # Logging
        if attempt % log_interval == 0:
            if attempt % 100 != 0:
                acc = evaluate_accuracy(net, pairs)
                if acc > best_acc:
                    best_acc = acc
                    best_acc_step = attempt
            elapsed = time.time() - start_time

            print(f"[Step {attempt:>7d}]  acc={acc*100:>5.1f}%  "
                  f"best={best_acc*100:.1f}% (@{best_acc_step})  "
                  f"conns={net.mask.sum().item():.0f}  "
                  f"stale={stale_count}  "
                  f"time={elapsed:.1f}s")

            monitor.log(net, extra={
                'accuracy': f'{acc*100:.1f}%',
                'best_accuracy': f'{best_acc*100:.1f}%',
            })

        # Success
        if attempt % 100 == 0 and acc >= target_acc:
            elapsed = time.time() - start_time
            print(f"\n{'='*70}")
            print(f"SUCCESS! 100% accuracy at step {attempt} ({elapsed:.1f}s)")
            print(f"Connections: {net.mask.sum().item():.0f} "
                  f"(self-wired: {net.conn_origin.sum().item():.0f})")
            print(f"Threshold: [{net.thresholds.min():.3f}, {net.thresholds.max():.3f}]")
            print(f"Target_W mag: {net.target_W.abs().mean():.4f}")
            print(f"{'='*70}")
            return net, monitor

        # Stale-out recovery
        if stale_count >= stale_limit:
            elapsed = time.time() - start_time
            print(f"\n[STALE-OUT step {attempt}] best={best_acc*100:.1f}% "
                  f"(@{best_acc_step}) time={elapsed:.1f}s")
            net.top_k_wire = min(net.top_k_wire + 2, 12)
            net.target_W += torch.randn_like(net.target_W) * 0.05
            net.thresholds += torch.randn_like(net.thresholds) * 0.02
            net.thresholds.clamp_(0.01, 5.0)
            print(f"  Recovery: top_k={net.top_k_wire}, target+threshold noise")
            stale_count = 0
            stale_limit += 10_000

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"MAX ATTEMPTS ({max_attempts}). Best: {best_acc*100:.1f}% (@{best_acc_step})")
    print(f"Time: {elapsed:.1f}s")
    print(f"{'='*70}")
    return net, monitor


if __name__ == '__main__':
    torch.manual_seed(42)
    random.seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    net, monitor = train_v19b(
        n_classes=32, n_neurons=160, max_attempts=200_000,
        log_interval=500, target_acc=1.0, device=device,
    )
