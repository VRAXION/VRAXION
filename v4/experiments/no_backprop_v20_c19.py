"""
No-Backprop v20 — C19 Activation in Self-Wiring Graph
======================================================
Same as v18 (self-wiring graph, the 100% winner) but replaces ReLU
with the C19 periodic parabolic wave activation from instnct.py.

C19 is dense (never zeros out neurons) vs ReLU which is sparse.
Key question: does dense activation help or hurt try→keep/revert?

Also: rho and C are evolvable per-neuron parameters (mutated like weights).

32-class test.
"""

import modal
import time

app = modal.App("vraxion-v20-c19")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch")
)


@app.function(cpu=2, memory=1024, timeout=600, image=image)
def run_experiment():
    import torch
    import math
    import random

    # ── C19 activation (standalone, no nn.Module needed) ──────────
    def c19_activation(x, rho, C):
        """
        Periodic parabolic wave.
        rho: curvature per neuron (vector)
        C: period per neuron (vector)
        """
        inv_c = 1.0 / C
        scaled = x * inv_c
        n = torch.floor(scaled)
        t = scaled - n
        h = t * (1.0 - t)
        is_even = torch.remainder(n, 2.0) < 1.0
        sgn = torch.where(is_even, 1.0, -1.0)
        core = C * (sgn * h + (rho * h * h))
        l = 6.0 * C
        result = torch.where(x <= -l, x + l, core)
        return torch.where(x >= l, x - l, result)

    class SelfWiringC19Net:
        """
        v18 self-wiring graph but with C19 activation instead of ReLU.
        Per-neuron rho and C are evolvable.
        """

        def __init__(self, n_neurons, n_in, n_out, addr_dim=4, density=0.06):
            self.N = n_neurons
            self.n_in = n_in
            self.n_out = n_out
            self.addr_dim = addr_dim

            s = math.sqrt(2.0 / n_neurons)
            self.W = torch.randn(n_neurons, n_neurons) * s
            self.mask = (torch.rand(n_neurons, n_neurons) < density).float()
            self.mask.fill_diagonal_(0)

            self.addresses = torch.randn(n_neurons, addr_dim)
            self.target_W = torch.randn(n_neurons, addr_dim) * 0.1

            # C19 per-neuron params
            self.rho = torch.full((n_neurons,), 4.0)
            self.C = torch.full((n_neurons,), math.pi)

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
                raw = act @ Weff + act * 0.1
                act = c19_activation(raw, self.rho, self.C)
                act[:self.n_in] = inp

            self.state = act.detach()
            self._self_wire(act)
            return act[-self.n_out:]

        def _self_wire(self, activations, top_k=3, max_new=2):
            internal_start = self.n_in
            act = activations[internal_start:].detach()

            if act.abs().sum() < 0.01:  # abs() because C19 can be negative
                return

            # Use absolute activation for ranking (C19 outputs can be negative)
            n_candidates = min(top_k, len(act))
            _, top_idx = act.abs().topk(n_candidates)
            top_idx = top_idx + internal_start

            new_connections = 0
            for neuron_idx in top_idx:
                ni = neuron_idx.item()
                if activations[ni].abs() < 0.1:
                    continue

                target = self.addresses[ni] + activations[ni] * self.target_W[ni]
                dists = ((self.addresses - target.unsqueeze(0)) ** 2).sum(dim=1)
                dists[ni] = float('inf')
                nearest = dists.argmin().item()

                if self.mask[ni, nearest] == 0:
                    self.mask[ni, nearest] = 1
                    self.W[ni, nearest] = random.gauss(0, math.sqrt(2.0 / self.N))
                    new_connections += 1

                if new_connections >= max_new:
                    break

        def count_connections(self):
            return int(self.mask.sum().item())

        def save_state(self):
            return (
                self.W.clone(), self.mask.clone(),
                self.state.clone(),
                self.addresses.clone(), self.target_W.clone(),
                self.rho.clone(), self.C.clone()
            )

        def restore_state(self, saved):
            self.W = saved[0].clone()
            self.mask = saved[1].clone()
            self.state = saved[2].clone()
            self.addresses = saved[3].clone()
            self.target_W = saved[4].clone()
            self.rho = saved[5].clone()
            self.C = saved[6].clone()

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

        def mutate_weights(self, scale=0.05):
            noise = torch.randn_like(self.W) * scale
            self.W += noise * self.mask
            self.target_W += torch.randn_like(self.target_W) * scale * 0.5
            self.addresses += torch.randn_like(self.addresses) * scale * 0.2
            # Evolve C19 params
            self.rho += torch.randn_like(self.rho) * scale * 0.3
            self.rho.clamp_(0.5, 8.0)
            self.C += torch.randn_like(self.C) * scale * 0.3
            self.C.clamp_(1.0, 50.0)

    def evaluate(net, inputs, targets, vocab, ticks=8, n_passes=2):
        total_score = 0.0
        correct = 0
        n_items = len(inputs)
        net.reset_state()
        prev_diff = torch.zeros(vocab)

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

        acc = correct / n_items
        avg_prob = total_score / n_items
        return 0.5 * acc + 0.5 * avg_prob, acc

    print(f"v20 C19 Activation — Self-Wiring Graph")
    print(f"=" * 60)

    torch.manual_seed(42); random.seed(42)
    v = 32
    perm = torch.randperm(v)
    inputs = list(range(v))
    n_neurons = 160
    n_in = v * 2
    n_out = v
    ticks = 8

    net = SelfWiringC19Net(n_neurons, n_in, n_out, addr_dim=4, density=0.06)

    print(f"  Neurons: {n_neurons} | Ticks: {ticks}")
    print(f"  Activation: C19 (periodic parabola) — DENSE, never zeros out")
    print(f"  Per-neuron rho (init=4.0) and C (init=π) — evolvable")
    print(f"  Self-wiring: same as v18")

    current_score, current_acc = evaluate(net, inputs, perm, v, ticks)
    best_score = current_score
    phase = "STRUCTURE"
    kept = 0
    stale = 0
    phase_switched = False
    stale_limit = 5000
    max_attempts = 50000
    t0 = time.time()

    print(f"  Start: Acc={current_acc*100:.0f}% | Score: {current_score:.3f} | "
          f"Conns: {net.count_connections()}")

    for attempt in range(max_attempts):
        state = net.save_state()

        if phase == "STRUCTURE":
            net.mutate_structure(rate=0.05)
        else:
            if random.random() < 0.3:
                net.mutate_structure(rate=0.02)
            else:
                net.mutate_weights(scale=0.05)

        new_score, new_acc = evaluate(net, inputs, perm, v, ticks)

        if new_score > current_score:
            current_score = new_score
            kept += 1
            stale = 0
            if current_score > best_score:
                best_score = current_score
        else:
            net.restore_state(state)
            stale += 1

        if phase == "STRUCTURE" and stale > 3000 and not phase_switched:
            phase = "BOTH"
            phase_switched = True
            _, acc = evaluate(net, inputs, perm, v, ticks)
            elapsed = time.time() - t0
            print(f"  -- PLATEAU at {attempt+1} ({elapsed:.1f}s) --")
            print(f"  -> BOTH | Acc: {acc*100:.0f}% | Score: {current_score:.3f} | "
                  f"Conns: {net.count_connections()}")
            stale = 0

        if (attempt + 1) % 500 == 0:
            _, acc = evaluate(net, inputs, perm, v, ticks)
            elapsed = time.time() - t0
            rho_mean = net.rho.mean().item()
            C_mean = net.C.mean().item()
            print(f"  [{phase:9s}] {attempt+1:6d} | "
                  f"Acc: {acc*100:.0f}% | Score: {current_score:.3f} | "
                  f"Kept: {kept} | Conns: {net.count_connections()} | "
                  f"rho: {rho_mean:.2f} | C: {C_mean:.2f} | "
                  f"Stale: {stale} | {elapsed:.1f}s")

        if best_score > 0.99:
            elapsed = time.time() - t0
            print(f"  -> Solved at {attempt+1}! ({elapsed:.1f}s)")
            break

        if stale >= stale_limit:
            elapsed = time.time() - t0
            _, acc = evaluate(net, inputs, perm, v, ticks)
            print(f"  -> STOPPED at {attempt+1}: stale {stale_limit} ({elapsed:.1f}s)")
            print(f"     Acc: {acc*100:.0f}% | Score: {current_score:.3f}")
            break

    elapsed = time.time() - t0
    _, final_acc = evaluate(net, inputs, perm, v, ticks)
    rho_final = net.rho.mean().item()
    C_final = net.C.mean().item()
    rho_std = net.rho.std().item()
    C_std = net.C.std().item()

    print(f"\n  Final: {final_acc*100:.0f}% | Score: {best_score:.3f} | "
          f"Conns: {net.count_connections()} | {elapsed:.1f}s")
    print(f"  rho: {rho_final:.3f} ± {rho_std:.3f} (init 4.0)")
    print(f"  C:   {C_final:.3f} ± {C_std:.3f} (init π={math.pi:.3f})")

    print(f"\n  Compare 32-class:")
    print(f"  v18 (ReLU + self-wire):  100% at ~10k")
    print(f"  v20 (C19 + self-wire):   {final_acc*100:.0f}%")

    return {"acc": final_acc, "score": best_score, "time": elapsed,
            "conns": net.count_connections(),
            "rho_mean": rho_final, "C_mean": C_final}


@app.local_entrypoint()
def main():
    print("Running v20 C19 activation in self-wiring graph...")
    results = run_experiment.remote()
    print(f"\nDone! {results['acc']*100:.0f}% | "
          f"rho={results['rho_mean']:.2f} | C={results['C_mean']:.2f}")
