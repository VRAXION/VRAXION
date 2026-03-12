"""
No-Backprop v20h — 64-CLASS with Simulated Annealing
=====================================================
v20e/f/g all plateau at 55-56%. Pure hill-climbing gets stuck.
Changes from v20g:
  1. Simulated annealing: accept worse mutations with decreasing probability
  2. Interleaved phases: alternate structure/weight blocks
  3. Adaptive mutation scale
  4. 320 neurons (faster per step, more iterations in budget)
  5. Reheat when stuck
"""

import modal
import time
import sys

app = modal.App("vraxion-v20h-anneal")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch")
)


@app.function(cpu=2, memory=1024, timeout=3600, image=image)
def run_experiment():
    import torch
    import torch.nn.functional as F
    import math
    import random

    class SelfWiringLeakyNet:
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
                act = F.leaky_relu(act @ Weff + act * 0.1)
                act[:self.n_in] = inp

            self.state = act.detach()
            self._self_wire(act)
            return act[-self.n_out:]

        def _self_wire(self, activations, top_k=5, max_new=3):
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

    print(f"v20h — 64-CLASS, Simulated Annealing + Interleaved Phases")
    print(f"=" * 60)
    sys.stdout.flush()

    torch.manual_seed(42); random.seed(42)
    v = 64
    perm = torch.randperm(v)
    inputs = list(range(v))
    n_neurons = 320
    n_in = v * 2
    n_out = v
    ticks = 8

    net = SelfWiringLeakyNet(n_neurons, n_in, n_out, addr_dim=4, density=0.06)

    print(f"  Neurons: {n_neurons} (internal: {n_neurons - n_in - n_out})")
    print(f"  Ticks: {ticks} | Eval passes: 2")
    print(f"  SA: T0=0.05, cooling=0.9997")
    print(f"  Interleaved: 300 structure / 300 weight blocks")
    sys.stdout.flush()

    current_score, current_acc = evaluate(net, inputs, perm, v, ticks)
    best_score = current_score
    best_acc = current_acc
    best_state = net.save_state()
    kept = 0
    sa_accepted = 0
    reheats = 0
    stale = 0
    stale_limit = 15000
    max_attempts = 300000
    t0 = time.time()

    # SA parameters
    temperature = 0.05
    cooling_rate = 0.9997
    min_temp = 0.001

    print(f"  Start: Acc={current_acc*100:.0f}% | Score: {current_score:.3f} | "
          f"Conns: {net.count_connections()}")
    sys.stdout.flush()

    for attempt in range(max_attempts):
        state = net.save_state()

        # Interleaved: 300 structure / 300 weight
        block = (attempt // 300) % 2
        if block == 0:
            net.mutate_structure(rate=0.05)
        else:
            if random.random() < 0.2:
                net.mutate_structure(rate=0.02)
            else:
                scale = 0.05 * (1.0 - current_score * 0.4)
                net.mutate_weights(scale=max(0.01, scale))

        new_score, new_acc = evaluate(net, inputs, perm, v, ticks)

        if new_score > current_score:
            current_score = new_score
            kept += 1
            stale = 0
            if current_score > best_score:
                best_score = current_score
                best_state = net.save_state()
            if new_acc > best_acc:
                best_acc = new_acc
        elif temperature > min_temp:
            delta = current_score - new_score
            accept_prob = math.exp(-delta / max(temperature, 1e-10))
            if random.random() < accept_prob:
                current_score = new_score
                sa_accepted += 1
                stale = 0
            else:
                net.restore_state(state)
                stale += 1
        else:
            net.restore_state(state)
            stale += 1

        temperature *= cooling_rate

        if (attempt + 1) % 2000 == 0:
            _, acc = evaluate(net, inputs, perm, v, ticks)
            elapsed = time.time() - t0
            phase = "STRUCT" if block == 0 else "WEIGHT"
            print(f"  {attempt+1:6d} | "
                  f"Acc: {acc*100:.0f}% | Score: {current_score:.3f} | "
                  f"Kept: {kept} | SA: {sa_accepted} | T: {temperature:.5f} | "
                  f"Conns: {net.count_connections()} | Stale: {stale} | {elapsed:.1f}s")
            sys.stdout.flush()

        if new_acc >= 1.0:
            elapsed = time.time() - t0
            print(f"  -> Solved at {attempt+1}! ({elapsed:.1f}s)")
            sys.stdout.flush()
            break

        if stale >= stale_limit:
            if reheats < 3:
                reheats += 1
                temperature = 0.03 / reheats
                net.restore_state(best_state)
                current_score = best_score
                stale = 0
                elapsed = time.time() - t0
                print(f"  -> REHEAT #{reheats} at {attempt+1} ({elapsed:.1f}s) | "
                      f"T={temperature:.4f} | Best: {best_score:.3f}")
                sys.stdout.flush()
            else:
                elapsed = time.time() - t0
                _, acc = evaluate(net, inputs, perm, v, ticks)
                print(f"  -> STOPPED at {attempt+1}: stale ({elapsed:.1f}s)")
                print(f"     Acc: {acc*100:.0f}% | Score: {current_score:.3f}")
                sys.stdout.flush()
                break

    net.restore_state(best_state)
    elapsed = time.time() - t0
    _, final_acc = evaluate(net, inputs, perm, v, ticks)
    print(f"\n  Final: {final_acc*100:.0f}% | Best: {best_acc*100:.0f}% | "
          f"Score: {best_score:.3f} | Conns: {net.count_connections()} | {elapsed:.1f}s")
    print(f"  SA accepted: {sa_accepted} | Kept: {kept} | Reheats: {reheats}")
    sys.stdout.flush()

    return {"acc": final_acc, "best_acc": best_acc, "score": best_score,
            "time": elapsed, "conns": net.count_connections(),
            "sa_accepted": sa_accepted, "kept": kept, "reheats": reheats}


@app.local_entrypoint()
def main():
    print("Running v20h SA + interleaved...")
    results = run_experiment.remote()
    print(f"\nDone! {results['acc']*100:.0f}% | Best: {results['best_acc']*100:.0f}% | "
          f"SA: {results['sa_accepted']} | Conns: {results['conns']}")
