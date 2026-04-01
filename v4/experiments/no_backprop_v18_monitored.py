"""
No-Backprop v18 — Self-Wiring Graph Network (MONITORED)
=========================================================
Original v18 + comprehensive internal monitoring.
All output is print-based (Modal sandbox compatible).

Monitors:
- Connection dynamics per phase
- Hub neuron detection (degree distribution)
- Connection lifetime tracking
- Neuron activation sparsity
- 4D address space dynamics (target magnitudes, short vs long range)
- Self-wiring proposal acceptance rate
- Graph topology (clustering, components)
- Plateau internal activity detection
"""

import modal
import time

app = modal.App("vraxion-selfwire-monitored")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch")
)


@app.function(cpu=2, memory=1024, timeout=3600, image=image)
def run_experiment():
    import torch
    import math
    import random
    from collections import defaultdict

    # =========================================================================
    # MONITOR CLASS — tracks internal dynamics
    # =========================================================================
    class NetworkMonitor:
        def __init__(self):
            self.conn_birth = {}        # (i,j) -> attempt when connection was born
            self.conn_deaths = []       # list of lifetimes of removed connections
            self.wire_proposals = 0     # total self-wire proposals
            self.wire_accepted = 0      # total self-wire accepted (new conn created)
            self.activation_history = []  # sparsity per eval
            self.conn_history = []      # connection count per log
            self.hub_history = []       # max degree per log
            self.magnitude_history = [] # target magnitudes per log
            self.phase_transitions = [] # (attempt, from_phase, to_phase, metrics)

            # Track which connections were added by self-wiring vs mutation
            self.selfwire_conns = set()
            self.mutation_conns = set()

        def record_mask(self, mask, attempt, source="init"):
            """Track connection births from current mask state."""
            current = set()
            nonzero = (mask == 1).nonzero(as_tuple=False)
            for k in range(len(nonzero)):
                i, j = nonzero[k][0].item(), nonzero[k][1].item()
                current.add((i, j))
                if (i, j) not in self.conn_birth:
                    self.conn_birth[(i, j)] = attempt

            # Record deaths
            dead = set(self.conn_birth.keys()) - current
            for conn in dead:
                lifetime = attempt - self.conn_birth[conn]
                self.conn_deaths.append(lifetime)
                del self.conn_birth[conn]
                self.selfwire_conns.discard(conn)
                self.mutation_conns.discard(conn)

        def record_selfwire(self, neuron_from, neuron_to, accepted):
            self.wire_proposals += 1
            if accepted:
                self.wire_accepted += 1
                self.selfwire_conns.add((neuron_from, neuron_to))

        def record_activation(self, activations, n_in):
            """Track activation sparsity."""
            internal = activations[n_in:].detach()
            active = (internal > 0.1).sum().item()
            total = len(internal)
            self.activation_history.append(active / total if total > 0 else 0)

        def compute_degree_stats(self, mask):
            """Compute in-degree and out-degree statistics."""
            out_degree = mask.sum(dim=1)  # row sums = outgoing
            in_degree = mask.sum(dim=0)   # col sums = incoming
            return out_degree, in_degree

        def compute_graph_stats(self, mask):
            """Compute basic graph topology metrics."""
            n = mask.shape[0]
            total_conns = int(mask.sum().item())
            max_possible = n * (n - 1)
            density = total_conns / max_possible if max_possible > 0 else 0

            out_deg, in_deg = self.compute_degree_stats(mask)

            # Hub detection: neurons with degree > 2*mean
            mean_out = out_deg.mean().item()
            mean_in = in_deg.mean().item()
            hub_out = (out_deg > 2 * mean_out).sum().item()
            hub_in = (in_deg > 2 * mean_in).sum().item()

            # Isolated neurons (no connections at all)
            isolated = ((out_deg + in_deg) == 0).sum().item()

            # Top-5 hubs by total degree
            total_deg = out_deg + in_deg
            top5_vals, top5_idx = total_deg.topk(min(5, n))

            return {
                "total_conns": total_conns,
                "density": density,
                "mean_out_deg": mean_out,
                "mean_in_deg": mean_in,
                "max_out_deg": int(out_deg.max().item()),
                "max_in_deg": int(in_deg.max().item()),
                "hub_out": hub_out,
                "hub_in": hub_in,
                "isolated": isolated,
                "top5_hubs": [(top5_idx[k].item(), int(top5_vals[k].item()))
                              for k in range(len(top5_idx))],
            }

        def compute_address_stats(self, addresses, target_W, activations):
            """Analyze 4D address space and target magnitudes."""
            # Target magnitudes (how far neurons "reach")
            magnitudes = target_W.norm(dim=1)

            # Address spread
            addr_std = addresses.std(dim=0)

            # Effective range: activation * target magnitude
            act_mag = activations.detach().abs()
            effective_range = act_mag * magnitudes

            return {
                "target_mag_mean": magnitudes.mean().item(),
                "target_mag_max": magnitudes.max().item(),
                "target_mag_std": magnitudes.std().item(),
                "addr_spread": addr_std.tolist(),
                "effective_range_mean": effective_range.mean().item(),
                "effective_range_max": effective_range.max().item(),
            }

        def compute_connection_age_stats(self, attempt):
            """Age distribution of current connections."""
            if not self.conn_birth:
                return {"n_alive": 0}

            ages = [attempt - birth for birth in self.conn_birth.values()]
            ages_t = torch.tensor(ages, dtype=torch.float32)

            return {
                "n_alive": len(ages),
                "mean_age": ages_t.mean().item(),
                "max_age": ages_t.max().item(),
                "median_age": ages_t.median().item(),
                "young_pct": (ages_t < 100).sum().item() / len(ages) * 100,
                "old_pct": (ages_t > 1000).sum().item() / len(ages) * 100,
            }

        def compute_selfwire_vs_mutation(self, mask):
            """How many current connections came from self-wiring vs mutation."""
            current = set()
            nonzero = (mask == 1).nonzero(as_tuple=False)
            for k in range(len(nonzero)):
                i, j = nonzero[k][0].item(), nonzero[k][1].item()
                current.add((i, j))

            sw = len(self.selfwire_conns & current)
            mt = len(self.mutation_conns & current)
            other = len(current) - sw - mt  # from init or untracked

            return {"selfwire": sw, "mutation": mt, "other": other}

        def lifetime_distribution(self):
            """Stats on dead connections' lifetimes."""
            if not self.conn_deaths:
                return {"n_dead": 0}

            lt = torch.tensor(self.conn_deaths, dtype=torch.float32)
            return {
                "n_dead": len(self.conn_deaths),
                "mean_lifetime": lt.mean().item(),
                "median_lifetime": lt.median().item(),
                "max_lifetime": lt.max().item(),
                "died_young_pct": (lt < 50).sum().item() / len(lt) * 100,
            }

        def print_report(self, net, attempt, phase, activations):
            """Print comprehensive monitoring report."""
            print(f"\n  {'='*65}")
            print(f"  MONITOR @ attempt {attempt} | Phase: {phase}")
            print(f"  {'='*65}")

            # --- Graph topology ---
            gs = self.compute_graph_stats(net.mask)
            print(f"  [GRAPH]  Conns: {gs['total_conns']} | "
                  f"Density: {gs['density']:.4f} | "
                  f"Isolated neurons: {gs['isolated']}")
            print(f"  [GRAPH]  Degree — Out: mean={gs['mean_out_deg']:.1f} "
                  f"max={gs['max_out_deg']} | "
                  f"In: mean={gs['mean_in_deg']:.1f} max={gs['max_in_deg']}")
            print(f"  [GRAPH]  Hubs (>2x mean) — Out: {gs['hub_out']} | "
                  f"In: {gs['hub_in']}")
            print(f"  [GRAPH]  Top-5 hubs (neuron, total_degree): "
                  f"{gs['top5_hubs']}")

            # --- Connection origins ---
            origins = self.compute_selfwire_vs_mutation(net.mask)
            total_o = origins['selfwire'] + origins['mutation'] + origins['other']
            if total_o > 0:
                print(f"  [ORIGIN] Self-wired: {origins['selfwire']} "
                      f"({origins['selfwire']/total_o*100:.0f}%) | "
                      f"Mutation: {origins['mutation']} "
                      f"({origins['mutation']/total_o*100:.0f}%) | "
                      f"Init/other: {origins['other']}")

            # --- Self-wiring efficiency ---
            if self.wire_proposals > 0:
                rate = self.wire_accepted / self.wire_proposals * 100
                print(f"  [WIRE]   Proposals: {self.wire_proposals} | "
                      f"Accepted: {self.wire_accepted} ({rate:.1f}%)")

            # --- Connection age ---
            age = self.compute_connection_age_stats(attempt)
            if age['n_alive'] > 0:
                print(f"  [AGE]    Alive: {age['n_alive']} | "
                      f"Mean: {age['mean_age']:.0f} | "
                      f"Median: {age['median_age']:.0f} | "
                      f"Max: {age['max_age']:.0f}")
                print(f"  [AGE]    Young (<100): {age['young_pct']:.0f}% | "
                      f"Old (>1000): {age['old_pct']:.0f}%")

            # --- Connection lifetime (dead connections) ---
            lt = self.lifetime_distribution()
            if lt['n_dead'] > 0:
                print(f"  [LIFE]   Dead connections: {lt['n_dead']} | "
                      f"Mean lifetime: {lt['mean_lifetime']:.0f} | "
                      f"Died young (<50): {lt['died_young_pct']:.0f}%")

            # --- Activation sparsity ---
            if self.activation_history:
                recent = self.activation_history[-10:]
                avg_sparsity = sum(recent) / len(recent)
                print(f"  [ACTIV]  Sparsity (recent): {avg_sparsity:.2%} "
                      f"neurons active (>0.1)")

            # --- 4D Address space ---
            addr = self.compute_address_stats(
                net.addresses, net.target_W, activations)
            print(f"  [ADDR]   Target magnitude — "
                  f"mean: {addr['target_mag_mean']:.3f} | "
                  f"max: {addr['target_mag_max']:.3f} | "
                  f"std: {addr['target_mag_std']:.3f}")
            print(f"  [ADDR]   Addr spread per dim: "
                  f"[{', '.join(f'{s:.2f}' for s in addr['addr_spread'])}]")
            print(f"  [ADDR]   Effective range — "
                  f"mean: {addr['effective_range_mean']:.3f} | "
                  f"max: {addr['effective_range_max']:.3f}")

            # --- Rewiring rate (plateau detector) ---
            if hasattr(self, '_last_mask') and self._last_mask is not None:
                changed = (net.mask != self._last_mask).sum().item()
                print(f"  [PLAT]   Connections changed since last report: "
                      f"{int(changed)} "
                      f"({'ACTIVE' if changed > 10 else 'QUIET'})")
            self._last_mask = net.mask.clone()

            print(f"  {'='*65}\n")

    # =========================================================================
    # MODIFIED NETWORK with monitor hooks
    # =========================================================================
    class SelfWiringNet:
        def __init__(self, n_neurons, n_in, n_out, addr_dim=4, density=0.06,
                     monitor=None):
            self.N = n_neurons
            self.n_in = n_in
            self.n_out = n_out
            self.addr_dim = addr_dim
            self.monitor = monitor

            s = math.sqrt(2.0 / n_neurons)
            self.W = torch.randn(n_neurons, n_neurons) * s
            self.mask = (torch.rand(n_neurons, n_neurons) < density).float()
            self.mask.fill_diagonal_(0)

            self.addresses = torch.randn(n_neurons, addr_dim)
            self.target_W = torch.randn(n_neurons, addr_dim) * 0.1

            self.state = torch.zeros(n_neurons)
            self.decay = 0.5

        def reset_state(self):
            self.state = torch.zeros(self.N)

        def forward(self, world, diff, ticks=8):
            inp = torch.cat([world, diff])
            act = self.state.clone()
            Weff = self.W * self.mask

            for t in range(ticks):
                act = act * self.decay
                act[:self.n_in] = inp
                act = torch.relu(act @ Weff + act * 0.1)
                act[:self.n_in] = inp

            self.state = act.detach()
            self._self_wire(act)

            # Monitor activation sparsity
            if self.monitor:
                self.monitor.record_activation(act, self.n_in)

            return act[-self.n_out:]

        def _self_wire(self, activations, top_k=3, max_new=2):
            internal_start = self.n_in
            act = activations[internal_start:].detach()

            if act.sum() < 0.01:
                return

            n_candidates = min(top_k, len(act))
            _, top_idx = act.topk(n_candidates)
            top_idx = top_idx + internal_start

            new_connections = 0
            for neuron_idx in top_idx:
                ni = neuron_idx.item()
                if activations[ni] < 0.1:
                    continue

                target = self.addresses[ni] + activations[ni] * self.target_W[ni]
                dists = ((self.addresses - target.unsqueeze(0)) ** 2).sum(dim=1)
                dists[ni] = float('inf')
                nearest = dists.argmin().item()

                if self.mask[ni, nearest] == 0:
                    self.mask[ni, nearest] = 1
                    self.W[ni, nearest] = random.gauss(0, math.sqrt(2.0 / self.N))
                    new_connections += 1

                    if self.monitor:
                        self.monitor.record_selfwire(ni, nearest, True)
                else:
                    if self.monitor:
                        self.monitor.record_selfwire(ni, nearest, False)

                if new_connections >= max_new:
                    break

        def count_connections(self):
            return int(self.mask.sum().item())

        def save_state(self):
            return (
                self.W.clone(), self.mask.clone(),
                self.state.clone(),
                self.addresses.clone(), self.target_W.clone()
            )

        def restore_state(self, saved):
            self.W = saved[0].clone()
            self.mask = saved[1].clone()
            self.state = saved[2].clone()
            self.addresses = saved[3].clone()
            self.target_W = saved[4].clone()

        def mutate_structure(self, rate=0.05):
            action = random.choice(["add", "remove", "rewire"])
            if action == "add":
                dead = (self.mask == 0).nonzero(as_tuple=False)
                dead = dead[dead[:, 0] != dead[:, 1]]
                if len(dead) > 0:
                    n = max(1, int(len(dead) * rate))
                    idx = dead[torch.randperm(len(dead))[:n]]
                    for j in range(len(idx)):
                        r, c = idx[j][0].item(), idx[j][1].item()
                        self.mask[r, c] = 1
                        self.W[r, c] = random.gauss(0, math.sqrt(2.0 / self.N))
                        if self.monitor:
                            self.monitor.mutation_conns.add((r, c))
            elif action == "remove":
                alive = (self.mask == 1).nonzero(as_tuple=False)
                if len(alive) > 3:
                    n = max(1, int(len(alive) * rate))
                    idx = alive[torch.randperm(len(alive))[:n]]
                    for j in range(len(idx)):
                        self.mask[idx[j][0], idx[j][1]] = 0
            elif action == "rewire":
                alive = (self.mask == 1).nonzero(as_tuple=False)
                if len(alive) > 0:
                    n = max(1, int(len(alive) * rate))
                    idx = alive[torch.randperm(len(alive))[:n]]
                    for j in range(len(idx)):
                        r, c = idx[j][0].item(), idx[j][1].item()
                        self.mask[r, c] = 0
                        new_c = random.randint(0, self.N - 1)
                        while new_c == r:
                            new_c = random.randint(0, self.N - 1)
                        self.mask[r, new_c] = 1
                        self.W[r, new_c] = self.W[r, c]
                        if self.monitor:
                            self.monitor.mutation_conns.add((r, new_c))

        def mutate_weights(self, scale=0.05):
            noise = torch.randn_like(self.W) * scale
            self.W += noise * self.mask
            self.target_W += torch.randn_like(self.target_W) * scale * 0.5
            self.addresses += torch.randn_like(self.addresses) * scale * 0.2

    def evaluate(net, inputs, targets, vocab, ticks=8, n_passes=2):
        total_score = 0.0
        correct = 0
        n_items = len(inputs)
        net.reset_state()
        prev_diff = torch.zeros(vocab)

        last_activations = None

        for pass_num in range(n_passes):
            for idx in range(n_items):
                world = torch.zeros(vocab)
                world[inputs[idx]] = 1.0
                logits = net.forward(world, prev_diff, ticks=ticks)
                probs = torch.softmax(logits, dim=-1)
                predicted = probs.argmax().item()
                target = targets[idx].item()

                if pass_num == n_passes - 1:
                    total_score += probs[target].item()
                    if predicted == target:
                        correct += 1

                target_vec = torch.zeros(vocab)
                target_vec[target] = 1.0
                prev_diff = target_vec - probs.detach()

            if pass_num == n_passes - 1:
                # Capture final activation state for monitoring
                last_activations = net.state.clone()

        acc = correct / n_items
        avg_prob = total_score / n_items
        return 0.5 * acc + 0.5 * avg_prob, acc, last_activations

    # =========================================================================
    # MAIN EXPERIMENT
    # =========================================================================
    print(f"v18 Self-Wiring Graph Network — MONITORED")
    print(f"=" * 65)

    torch.manual_seed(42); random.seed(42)
    v = 32
    perm = torch.randperm(v)
    inputs = list(range(v))
    n_neurons = 160
    n_in = v * 2
    n_out = v
    ticks = 8

    monitor = NetworkMonitor()
    net = SelfWiringNet(n_neurons, n_in, n_out, addr_dim=4, density=0.06,
                        monitor=monitor)

    # Record initial mask
    monitor.record_mask(net.mask, 0, "init")

    print(f"  Neurons: {n_neurons} | Ticks: {ticks} | Addr dim: 4")
    print(f"  Self-wiring: top-3 active → propose connections each forward")
    print(f"  MONITORING: graph topology, hub detection, connection ages,")
    print(f"              activation sparsity, 4D address dynamics,")
    print(f"              self-wire vs mutation origin tracking")

    current_score, current_acc, last_act = evaluate(
        net, inputs, perm, v, ticks)
    best_score = current_score
    phase = "STRUCTURE"
    kept = 0
    stale = 0
    phase_switched = False
    stale_limit = 20000
    max_attempts = 200000
    t0 = time.time()

    # --- Monitoring intervals ---
    MONITOR_EVERY = 2500     # full report
    LOG_EVERY = 1000         # short log line

    print(f"  Start: Acc={current_acc*100:.0f}% | Score: {current_score:.3f} | "
          f"Conns: {net.count_connections()}")
    print(f"  Monitor reports every {MONITOR_EVERY} attempts, "
          f"short log every {LOG_EVERY}")

    # Initial full report
    monitor.print_report(net, 0, phase, last_act)

    for attempt in range(max_attempts):
        state = net.save_state()

        if phase == "STRUCTURE":
            net.mutate_structure(rate=0.05)
        else:
            if random.random() < 0.3:
                net.mutate_structure(rate=0.02)
            else:
                net.mutate_weights(scale=0.05)

        new_score, new_acc, new_act = evaluate(
            net, inputs, perm, v, ticks)

        if new_score > current_score:
            current_score = new_score
            kept += 1
            stale = 0
            if current_score > best_score:
                best_score = current_score
            # Track mask changes on accepted mutations
            monitor.record_mask(net.mask, attempt + 1)
        else:
            net.restore_state(state)
            stale += 1

        # --- Phase transition ---
        if phase == "STRUCTURE" and stale > 3000 and not phase_switched:
            phase = "BOTH"
            phase_switched = True
            _, acc, act = evaluate(net, inputs, perm, v, ticks)
            elapsed = time.time() - t0
            print(f"\n  {'*'*65}")
            print(f"  PHASE TRANSITION: STRUCTURE -> BOTH @ attempt {attempt+1} "
                  f"({elapsed:.1f}s)")
            print(f"  Acc: {acc*100:.0f}% | Score: {current_score:.3f} | "
                  f"Conns: {net.count_connections()}")
            print(f"  {'*'*65}")
            monitor.print_report(net, attempt + 1, phase, act)
            stale = 0

        # --- Short log ---
        if (attempt + 1) % LOG_EVERY == 0:
            _, acc, _ = evaluate(net, inputs, perm, v, ticks)
            elapsed = time.time() - t0
            conns = net.count_connections()
            wire_rate = (monitor.wire_accepted / max(1, monitor.wire_proposals)
                         * 100)
            print(f"  [{phase:9s}] {attempt+1:6d} | "
                  f"Acc: {acc*100:.0f}% | Score: {current_score:.3f} | "
                  f"Kept: {kept} | Conns: {conns} | "
                  f"SelfWire: {wire_rate:.0f}% | "
                  f"Stale: {stale} | {elapsed:.1f}s")

        # --- Full monitoring report ---
        if (attempt + 1) % MONITOR_EVERY == 0:
            _, _, act = evaluate(net, inputs, perm, v, ticks)
            monitor.record_mask(net.mask, attempt + 1)
            monitor.print_report(net, attempt + 1, phase, act)

        # --- Solved ---
        if best_score > 0.99:
            elapsed = time.time() - t0
            _, _, act = evaluate(net, inputs, perm, v, ticks)
            print(f"\n  -> SOLVED at {attempt+1}! ({elapsed:.1f}s)")
            monitor.record_mask(net.mask, attempt + 1)
            monitor.print_report(net, attempt + 1, phase, act)
            break

        # --- Stale stop ---
        if stale >= stale_limit:
            elapsed = time.time() - t0
            _, acc, act = evaluate(net, inputs, perm, v, ticks)
            print(f"\n  -> STOPPED at {attempt+1}: stale {stale_limit} "
                  f"({elapsed:.1f}s)")
            print(f"     Acc: {acc*100:.0f}% | Score: {current_score:.3f}")
            monitor.record_mask(net.mask, attempt + 1)
            monitor.print_report(net, attempt + 1, phase, act)
            break

    # =========================================================================
    # FINAL REPORT
    # =========================================================================
    elapsed = time.time() - t0
    _, final_acc, final_act = evaluate(net, inputs, perm, v, ticks)

    print(f"\n{'='*65}")
    print(f"  FINAL RESULTS")
    print(f"{'='*65}")
    print(f"  Accuracy: {final_acc*100:.0f}% | Score: {best_score:.3f} | "
          f"Conns: {net.count_connections()} | Time: {elapsed:.1f}s")

    # Final full monitor report
    monitor.record_mask(net.mask, max_attempts)
    monitor.print_report(net, max_attempts, phase, final_act)

    # Per-input details
    net.reset_state()
    prev_diff = torch.zeros(v)
    for idx in range(len(inputs)):
        world = torch.zeros(v)
        world[inputs[idx]] = 1.0
        logits = net.forward(world, prev_diff, ticks=ticks)
        probs = torch.softmax(logits, dim=-1)
        target_vec = torch.zeros(v)
        target_vec[perm[idx].item()] = 1.0
        prev_diff = target_vec - probs.detach()
    print(f"\n  Per-input (warm pass):")
    for idx in range(32):
        world = torch.zeros(v)
        world[inputs[idx]] = 1.0
        logits = net.forward(world, prev_diff, ticks=ticks)
        probs = torch.softmax(logits, dim=-1)
        pred = probs.argmax().item()
        tgt = perm[idx].item()
        conf = probs[pred].item()
        mark = "ok" if pred == tgt else "X "
        print(f"    {inputs[idx]:2d} -> {pred:2d} (exp:{tgt:2d}) {mark}  "
              f"conf:{conf:.3f}")
        target_vec = torch.zeros(v)
        target_vec[tgt] = 1.0
        prev_diff = target_vec - probs.detach()

    print(f"\n  Compare 32-class:")
    print(f"  v12 (layered):          91%")
    print(f"  v16 (graph):            81%")
    print(f"  v17 (graph+recurrent):  88% (timeout)")
    print(f"  v18 (self-wiring):      {final_acc*100:.0f}%")

    return {"acc": final_acc, "score": best_score, "time": elapsed,
            "conns": net.count_connections()}


@app.local_entrypoint()
def main():
    print("Running self-wiring network on Modal CPU (MONITORED)...")
    results = run_experiment.remote()
    print(f"\nDone! 32-class: {results['acc']*100:.0f}%")
