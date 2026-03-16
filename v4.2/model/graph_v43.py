"""
Self-Wiring Graph Network v4.3 — Dynamic Growth
=================================================
Everything from v4.2 PLUS:
- Dynamic N (pre-allocated N_MAX, active sub-region)
- Energy-guided chain detection (follow the signal highways)
- Chain spawn: clone/anti/split variants of high-energy chains
- Neuron sculpting: kill low-energy neurons
- Q3 growth_type bit in decision tree

EXPERIMENTAL — not yet sweep-validated.
"""

import numpy as np
import random


class SelfWiringGraph:

    # Fixed constants (all sweep-validated in v4.2)
    NV_RATIO = 3
    NV_MAX = 8        # pre-allocate up to 8× vocab
    GAIN = 2
    CHARGE_RATE = 0.3
    SELF_CONN = 0.05
    THRESHOLD = 0.5
    CLIP_BOUND = 1.0
    PATIENCE = 0.35
    LOSS_DRIFT = 0.2

    def __init__(self, *args, density=0.06):
        if len(args) == 1:
            vocab = args[0]
        else:
            _, vocab = args[0], args[1]
        self.V = vocab
        self.N = vocab * self.NV_RATIO       # active neurons
        self.N_MAX = vocab * self.NV_MAX     # pre-allocated capacity

        # Split I/O: first V = input, last V = output
        self.out_start = self.N - vocab if self.N >= 2 * vocab else 0

        # Pre-allocated mask (N_MAX × N_MAX), only [0:N, 0:N] active
        r = np.random.rand(self.N, self.N)
        self.mask = np.zeros((self.N_MAX, self.N_MAX), dtype=np.int8)
        self.mask[:self.N, :self.N][r < density / 2] = -1
        self.mask[:self.N, :self.N][r > 1 - density / 2] = 1
        np.fill_diagonal(self.mask, 0)

        # Alive edges
        rows, cols = np.where(self.mask != 0)
        self.alive = list(zip(rows.tolist(), cols.tolist()))
        self.alive_set = set(self.alive)

        # Persistent state (pre-allocated to N_MAX)
        self.state = np.zeros(self.N_MAX, dtype=np.float32)
        self.charge = np.zeros(self.N_MAX, dtype=np.float32)

        # Energy per neuron (updated after each forward_batch)
        self.energy = np.zeros(self.N_MAX, dtype=np.float32)

        # Co-evolved learned params (all int8)
        self.loss_pct = np.int8(15)
        self.signal = np.int8(0)       # Q1: 0=structural, 1=signal
        self.grow = np.int8(1)         # Q2: 0=shrink, 1=grow
        self.growth_type = np.int8(0)  # Q3: 0=micro, 1=split, 2=anti
        self.intensity = np.int8(7)

    def reset(self):
        self.state[:self.N] *= 0
        self.charge[:self.N] *= 0

    @property
    def retention(self):
        return np.float32((100 - int(self.loss_pct)) * 0.01)

    # Backward compat properties (same as v4.2)
    @property
    def loss(self):
        return int(self.loss_pct)

    @loss.setter
    def loss(self, value):
        self.loss_pct = np.int8(max(1, min(50, int(value))))

    @property
    def leak(self):
        return float(self.retention)

    @leak.setter
    def leak(self, value):
        if isinstance(value, (float, np.floating)):
            loss = int(round((1.0 - float(value)) * 100.0))
        else:
            iv = int(value)
            loss = 100 - iv if 50 <= iv <= 99 else iv
        self.loss_pct = np.int8(max(1, min(50, loss)))

    def forward_batch(self, ticks=8):
        """Batch forward using ACTIVE sub-matrix only."""
        V, N = self.V, self.N
        charges = np.zeros((V, N), dtype=np.float32)
        acts = np.zeros((V, N), dtype=np.float32)
        retain = float(self.retention)
        for t in range(ticks):
            if t == 0:
                acts[:, :V] = np.eye(V, dtype=np.float32)
            raw = acts @ self.mask[:N, :N] * self.GAIN + acts * self.SELF_CONN
            np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            charges += raw * self.CHARGE_RATE
            charges *= retain
            acts = np.maximum(charges - self.THRESHOLD, 0.0)
            charges = np.clip(charges, -self.CLIP_BOUND, self.CLIP_BOUND)

        # Update energy (FREE — charges already computed)
        self.energy[:N] = np.abs(charges).sum(axis=0)

        self.out_start = N - V if N >= 2 * V else 0
        return charges[:, self.out_start:self.out_start + V]

    def forward(self, world, ticks=8):
        """Single-input forward pass."""
        N = self.N
        act = self.state[:N].copy()
        retain = float(self.retention)
        for t in range(ticks):
            if t == 0:
                act[:self.V] = world
            raw = act @ self.mask[:N, :N] * self.GAIN + act * self.SELF_CONN
            np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            self.charge[:N] += raw * self.CHARGE_RATE
            self.charge[:N] *= retain
            act = np.maximum(self.charge[:N] - self.THRESHOLD, 0.0)
            self.charge[:N] = np.clip(self.charge[:N], -self.CLIP_BOUND, self.CLIP_BOUND)
        self.state[:N] = act
        self.out_start = N - self.V if N >= 2 * self.V else 0
        return self.charge[self.out_start:self.out_start + self.V]

    # --- Chain analytics ---

    def chain_stats(self):
        """Compute full chain/path statistics via BFS from every neuron.

        Returns dict with:
            avg_path:     mean shortest path length (connected pairs)
            diameter:     longest shortest path (graph diameter)
            avg_degree:   mean out-degree of active neurons
            dead_neurons: neurons with zero in+out edges
            bottlenecks:  edges whose removal disconnects components
            clustering:   average local clustering coefficient
            reachable:    fraction of neuron pairs that are connected
        """
        from collections import deque
        N = self.N
        # Build adjacency list
        adj = [[] for _ in range(N)]
        for r, c in self.alive:
            if r < N and c < N:
                adj[r].append(c)

        # BFS from every neuron
        total_dist = 0
        total_pairs = 0
        diameter = 0

        for start in range(N):
            dist = [-1] * N
            dist[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
                        total_dist += dist[v]
                        total_pairs += 1
                        if dist[v] > diameter:
                            diameter = dist[v]

        avg_path = total_dist / total_pairs if total_pairs > 0 else 0
        reachable = total_pairs / (N * (N - 1)) if N > 1 else 0

        # Degree stats
        out_deg = np.array([len(adj[i]) for i in range(N)])
        in_deg = np.zeros(N, dtype=int)
        for r, c in self.alive:
            if c < N:
                in_deg[c] += 1

        dead = int(((out_deg == 0) & (in_deg == 0)).sum())

        # Local clustering coefficient
        # For directed graph: fraction of neighbor pairs that are connected
        cluster_sum = 0.0
        cluster_count = 0
        for u in range(N):
            neighbors = set(adj[u])
            k = len(neighbors)
            if k < 2:
                continue
            links = 0
            for v in neighbors:
                for w in neighbors:
                    if v != w and w in set(adj[v]):
                        links += 1
            cluster_sum += links / (k * (k - 1))
            cluster_count += 1

        clustering = cluster_sum / cluster_count if cluster_count > 0 else 0

        # Bottleneck detection: edges on all shortest paths between I/O
        # Simplified: find edges with highest "betweenness" approximation
        edge_usage = {}
        # Sample BFS from input neurons to output neurons
        for src in range(min(self.V, 8)):  # sample up to 8 input neurons
            dist = [-1] * N
            parent = [[] for _ in range(N)]
            dist[src] = 0
            q = deque([src])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = [u]
                        q.append(v)
                    elif dist[v] == dist[u] + 1:
                        parent[v].append(u)
            # Trace back from output neurons
            for out_n in range(self.out_start, min(self.out_start + self.V, N)):
                if dist[out_n] == -1:
                    continue
                # Walk parents back
                frontier = [out_n]
                while frontier:
                    nxt = []
                    for node in frontier:
                        for p in parent[node]:
                            key = (p, node)
                            edge_usage[key] = edge_usage.get(key, 0) + 1
                            nxt.append(p)
                    frontier = nxt

        # Top bottleneck edges (highest betweenness)
        bottlenecks = sorted(edge_usage.items(), key=lambda x: -x[1])[:10]

        return {
            'avg_path': round(avg_path, 2),
            'diameter': diameter,
            'avg_out_degree': round(float(out_deg.mean()), 2),
            'avg_in_degree': round(float(in_deg.mean()), 2),
            'dead_neurons': dead,
            'clustering': round(clustering, 4),
            'reachable': round(reachable, 4),
            'total_edges': len(self.alive),
            'total_neurons': N,
            'bottlenecks': bottlenecks,
        }

    # --- Energy-guided chain detection ---

    def find_chain(self, length=3):
        """Follow energy gradient to find the highest-energy chain."""
        N = self.N
        if N == 0:
            return []
        start = int(np.argmax(self.energy[:N]))
        chain = [start]
        visited = {start}
        for _ in range(length - 1):
            row = self.mask[chain[-1], :N]
            candidates = np.where(row != 0)[0]
            candidates = [c for c in candidates if c not in visited]
            if not candidates:
                break
            best = max(candidates, key=lambda c: self.energy[c])
            chain.append(best)
            visited.add(best)
        return chain

    # --- Alive management ---

    def resync_alive(self):
        rows, cols = np.where(self.mask[:self.N, :self.N] != 0)
        self.alive = list(zip(rows.tolist(), cols.tolist()))
        self.alive_set = set(self.alive)

    def count_connections(self):
        return len(self.alive)

    def pos_neg_ratio(self):
        pos = sum(1 for r, c in self.alive if self.mask[r, c] > 0)
        return pos, len(self.alive) - pos

    # --- State management ---

    def save_state(self):
        N = self.N
        return {
            'mask': self.mask[:N, :N].copy(),
            'alive': self.alive.copy(),
            'alive_set': self.alive_set.copy(),
            'state': self.state[:N].copy(),
            'charge': self.charge[:N].copy(),
            'loss_pct': np.int8(self.loss_pct),
            'N': N,
        }

    def restore_state(self, s):
        old_N = s['N']
        # Clear any neurons beyond saved N
        if self.N > old_N:
            self.mask[old_N:self.N, :] = 0
            self.mask[:, old_N:self.N] = 0
            self.state[old_N:self.N] = 0
            self.charge[old_N:self.N] = 0
        self.N = old_N
        self.mask[:old_N, :old_N] = s['mask']
        self.alive = s['alive'].copy()
        self.alive_set = s.get('alive_set', set(self.alive)).copy()
        self.state[:old_N] = s['state']
        self.charge[:old_N] = s['charge']
        self.loss_pct = np.int8(s.get('loss_pct', 15))

    def replay(self, log):
        """Undo logged ops. Handles neuron growth/kill too."""
        has_structural = False
        for entry in reversed(log):
            op = entry[0]
            if op == 'F':
                self.mask[entry[1], entry[2]] *= -1
            elif op == 'A':
                self.mask[entry[1], entry[2]] = 0
                self.alive_set.discard((entry[1], entry[2]))
                has_structural = True
            elif op == 'R':
                self.mask[entry[1], entry[2]] = entry[3]
                self.alive_set.add((entry[1], entry[2]))
                has_structural = True
            elif op == 'W':
                _, r, c_old, c_new = entry
                sign = self.mask[r, c_new]
                self.mask[r, c_new] = 0
                self.mask[r, c_old] = sign
                self.alive_set.discard((r, c_new))
                self.alive_set.add((r, c_old))
                has_structural = True
            elif op == 'SPLIT':
                _, r, c, old_sign, n = entry
                self.mask[r, n] = 0
                self.mask[n, c] = 0
                self.mask[r, c] = old_sign
                self.N -= 1
                has_structural = True
            elif op == 'ANTI':
                _, n = entry
                self.mask[n, :] = 0
                self.mask[:, n] = 0
                self.N -= 1
                has_structural = True
            elif op == 'KILL':
                _, n, saved_row, saved_col, old_N = entry
                self.N = old_N
                self.mask[n, :old_N] = saved_row
                self.mask[:old_N, n] = saved_col
                has_structural = True
        if has_structural:
            self.resync_alive()

    # --- Mutation ---

    def mutate(self):
        """Extended decision tree with Q3 growth_type."""
        # Intensity drift
        if random.random() < self.PATIENCE:
            self.intensity = np.int8(max(1, min(15, int(self.intensity) + random.choice([-1, 1]))))

        # Loss step
        if random.random() < self.LOSS_DRIFT:
            self.loss_pct = np.int8(max(1, min(50, int(self.loss_pct) + random.randint(-3, 3))))

        undo = []
        for _ in range(int(self.intensity)):
            if self.signal:         # Q1: signal → flip
                self._flip(undo)
            else:                   # structural
                if self.grow:       # Q2: grow
                    gt = int(self.growth_type)
                    if gt == 0:
                        self._add(undo)        # MICRO
                    elif gt == 1:
                        self._split(undo)      # SPLIT (NEAT-style)
                    else:
                        self._anti(undo)       # ANTI (original)
                else:               # shrink
                    if random.random() < 0.5:
                        self._remove(undo)
                    elif random.random() < 0.5:
                        self._rewire(undo)
                    else:
                        self._kill(undo)       # KILL neuron
        return undo

    def mutate_with_mood(self):
        return self.mutate()

    # --- Mutation ops ---

    def _add(self, undo):
        N = self.N
        r, c = random.randint(0, N-1), random.randint(0, N-1)
        if r != c and self.mask[r, c] == 0:
            self.mask[r, c] = random.choice([-1, 1])
            self.alive.append((r, c))
            self.alive_set.add((r, c))
            undo.append(('A', r, c))

    def _flip(self, undo):
        if self.alive:
            idx = random.randint(0, len(self.alive)-1)
            r, c = self.alive[idx]
            self.mask[r, c] *= -1
            undo.append(('F', r, c))

    def _remove(self, undo):
        if self.alive:
            idx = random.randint(0, len(self.alive)-1)
            r, c = self.alive[idx]
            old_sign = self.mask[r, c]
            self.mask[r, c] = 0
            self.alive[idx] = self.alive[-1]
            self.alive.pop()
            self.alive_set.discard((r, c))
            undo.append(('R', r, c, old_sign))

    def _rewire(self, undo):
        if self.alive:
            idx = random.randint(0, len(self.alive)-1)
            r, c = self.alive[idx]
            nc = random.randint(0, self.N-1)
            if nc != r and nc != c and self.mask[r, nc] == 0:
                old = self.mask[r, c]
                self.mask[r, c] = 0
                self.mask[r, nc] = old
                self.alive[idx] = (r, nc)
                self.alive_set.discard((r, c))
                self.alive_set.add((r, nc))
                undo.append(('W', r, c, nc))

    def _split(self, undo):
        """NEAT-style: split alive edge, insert new neuron."""
        if not self.alive or self.N >= self.N_MAX:
            return
        idx = random.randint(0, len(self.alive)-1)
        r, c = self.alive[idx]
        old_sign = self.mask[r, c]

        n = self.N  # new neuron index
        self.mask[r, c] = 0           # disable original
        self.mask[r, n] = 1           # A→N weight=+1
        self.mask[n, c] = old_sign    # N→B weight=original sign
        self.N += 1

        self.alive_set.discard((r, c))
        self.alive_set.add((r, n))
        self.alive_set.add((n, c))
        self.alive = list(self.alive_set)
        undo.append(('SPLIT', r, c, old_sign, n))

    def _anti(self, undo):
        """Create sign-inverted clone of a high-energy neuron."""
        if self.N >= self.N_MAX or self.N < 2:
            return
        # Pick high-energy internal neuron (not I/O boundary)
        candidates = list(range(self.V, self.N - self.V))
        if not candidates:
            candidates = list(range(self.N))
        # Bias toward high energy
        energies = self.energy[candidates]
        if energies.sum() > 0:
            probs = energies / energies.sum()
            src = np.random.choice(candidates, p=probs)
        else:
            src = random.choice(candidates)

        n = self.N
        # Copy row (outgoing) with flipped signs
        self.mask[n, :self.N] = -self.mask[src, :self.N]
        # Copy column (incoming) with flipped signs
        self.mask[:self.N, n] = -self.mask[:self.N, src]
        self.mask[n, n] = 0  # no self-connection
        self.N += 1

        self.resync_alive()
        undo.append(('ANTI', n))

    # --- Pruning ---

    def prune(self, targets, ticks=8, tolerance=0.005, strategy='energy',
              batch_size=1, verbose=True):
        """Iteratively remove edges while accuracy holds.

        Args:
            targets:    target permutation array (length V)
            ticks:      forward pass ticks
            tolerance:  max allowed score drop from baseline
            strategy:   'energy' (weakest first) or 'random'
            batch_size: edges to remove per iteration (1 = safest)
            verbose:    print progress

        Returns:
            dict with pruning stats
        """
        V, N = self.V, self.N

        def evaluate():
            logits = self.forward_batch(ticks)
            e = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = e / e.sum(axis=1, keepdims=True)
            acc = (np.argmax(probs, axis=1)[:V] == targets[:V]).mean()
            tp = probs[np.arange(V), targets[:V]].mean()
            return 0.5 * acc + 0.5 * tp

        baseline = evaluate()
        threshold = baseline - tolerance
        init_edges = len(self.alive)
        removed = 0
        history = [(0, init_edges, baseline)]

        if verbose:
            print(f"  Prune start: {init_edges} edges, "
                  f"score={baseline*100:.1f}%, threshold={threshold*100:.1f}%")

        while self.alive:
            # Rank edges by weakness
            if strategy == 'energy':
                # Score each edge: low energy at both endpoints = weak
                scored = []
                for r, c in self.alive:
                    edge_score = self.energy[r] + self.energy[c]
                    scored.append((edge_score, r, c))
                scored.sort()  # weakest first
                candidates = [(r, c) for _, r, c in scored[:batch_size]]
            else:
                # Random removal
                indices = random.sample(range(len(self.alive)),
                                        min(batch_size, len(self.alive)))
                candidates = [self.alive[i] for i in indices]

            # Save state for rollback
            saved_edges = []
            for r, c in candidates:
                saved_edges.append((r, c, self.mask[r, c]))
                self.mask[r, c] = 0
                self.alive_set.discard((r, c))

            self.alive = list(self.alive_set)

            new_score = evaluate()

            if new_score >= threshold:
                # Accept: edges were dispensable
                removed += len(candidates)
                history.append((removed, len(self.alive), new_score))
                if verbose and removed % 50 == 0:
                    print(f"    removed {removed:4d} | "
                          f"edges={len(self.alive):4d} | "
                          f"score={new_score*100:.1f}%")
            else:
                # Reject: restore edges, stop pruning
                for r, c, val in saved_edges:
                    self.mask[r, c] = val
                    self.alive_set.add((r, c))
                self.alive = list(self.alive_set)
                if verbose:
                    print(f"    STOP at {removed} removed | "
                          f"would drop to {new_score*100:.1f}%")
                break

        final_edges = len(self.alive)
        final_score = evaluate()
        compression = 1.0 - (final_edges / init_edges) if init_edges > 0 else 0

        result = {
            'init_edges': init_edges,
            'final_edges': final_edges,
            'removed': removed,
            'compression': compression,
            'baseline_score': baseline,
            'final_score': final_score,
            'tolerance': tolerance,
            'strategy': strategy,
            'history': history,
        }

        if verbose:
            print(f"  Prune done: {init_edges} → {final_edges} edges "
                  f"({compression*100:.1f}% compression) | "
                  f"score: {baseline*100:.1f}% → {final_score*100:.1f}%")

        return result

    def prune_neurons(self, targets, ticks=8, tolerance=0.005, verbose=True):
        """Kill neurons with zero remaining edges after edge pruning."""
        V, N = self.V, self.N

        def evaluate():
            logits = self.forward_batch(ticks)
            e = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = e / e.sum(axis=1, keepdims=True)
            acc = (np.argmax(probs, axis=1)[:V] == targets[:V]).mean()
            tp = probs[np.arange(V), targets[:V]].mean()
            return 0.5 * acc + 0.5 * tp

        baseline = evaluate()
        threshold = baseline - tolerance
        killed = 0

        # Find internal neurons with no connections
        for n in range(self.V, self.N - self.V):
            row_sum = np.abs(self.mask[n, :self.N]).sum()
            col_sum = np.abs(self.mask[:self.N, n]).sum()
            if row_sum == 0 and col_sum == 0:
                killed += 1

        # Also try killing lowest-energy neurons
        internals = list(range(self.V, self.N - self.V))
        if internals:
            energies = [(self.energy[n], n) for n in internals]
            energies.sort()  # lowest energy first
            for _, n in energies:
                saved_row = self.mask[n, :self.N].copy()
                saved_col = self.mask[:self.N, n].copy()
                self.mask[n, :] = 0
                self.mask[:, n] = 0
                self.resync_alive()

                if evaluate() >= threshold:
                    killed += 1
                    if verbose:
                        print(f"    killed neuron {n} (energy={self.energy[n]:.4f})")
                else:
                    self.mask[n, :self.N] = saved_row
                    self.mask[:self.N, n] = saved_col
                    self.resync_alive()

        if verbose:
            print(f"  Neuron prune: killed {killed} neurons")
        return killed

    def _kill(self, undo):
        """Remove lowest-energy internal neuron."""
        if self.N <= self.V * 2:  # minimum N/V=2
            return
        # Find lowest energy internal neuron
        internals = list(range(self.V, self.N - self.V))
        if not internals:
            return
        victim = min(internals, key=lambda n: self.energy[n])

        saved_row = self.mask[victim, :self.N].copy()
        saved_col = self.mask[:self.N, victim].copy()
        old_N = self.N

        # Zero all connections
        self.mask[victim, :] = 0
        self.mask[:, victim] = 0

        # Swap with last active neuron to keep contiguous
        last = self.N - 1
        if victim != last:
            self.mask[victim, :] = self.mask[last, :]
            self.mask[:, victim] = self.mask[:, last]
            self.mask[last, :] = 0
            self.mask[:, last] = 0
            self.mask[victim, victim] = 0  # no self-conn after swap
            self.energy[victim] = self.energy[last]
            self.state[victim] = self.state[last]
            self.charge[victim] = self.charge[last]

        self.N -= 1
        self.resync_alive()
        undo.append(('KILL', victim, saved_row, saved_col, old_N))


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def train(net, targets, vocab, max_attempts=8000, ticks=8,
          stale_limit=6000, verbose=True):
    """Train with extended decision tree (Q1+Q2+Q3)."""

    def evaluate():
        logits = net.forward_batch(ticks)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        V = min(vocab, net.V)
        acc = (np.argmax(probs, axis=1)[:V] == targets[:V]).mean()
        tp = probs[np.arange(V), targets[:V]].mean()
        return 0.5 * acc + 0.5 * tp

    score = evaluate()
    best = score
    stale = 0

    for att in range(max_attempts):
        old_loss = int(net.loss_pct)
        old_N = net.N
        undo = net.mutate()
        new_score = evaluate()

        if new_score > score:
            score = new_score
            best = max(best, score)
            stale = 0
        else:
            net.replay(undo)
            net.loss_pct = np.int8(old_loss)
            if net.N != old_N:
                # replay should have fixed it, but safety
                net.N = old_N
            stale += 1
            # Strategy flip on reject
            if random.random() < net.PATIENCE:
                net.signal = np.int8(1 - int(net.signal))
            if random.random() < net.PATIENCE:
                net.grow = np.int8(1 - int(net.grow))
            if random.random() < net.PATIENCE:
                net.growth_type = np.int8(random.randint(0, 2))

        if verbose and (att + 1) % 1000 == 0:
            gt = ['MICRO', 'SPLIT', 'ANTI'][int(net.growth_type)]
            mode = "SIGNAL" if net.signal else ("GROW/" + gt if net.grow else "SHRINK")
            print(f"  [{att+1:5d}] Score: {best*100:5.1f}% | "
                  f"N={net.N} Conns={net.count_connections()} | {mode} | "
                  f"Loss: {int(net.loss_pct)}%")

        if best >= 0.99 or stale >= stale_limit:
            break

    return best
