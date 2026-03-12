"""
No-Backprop v20i — 64-CLASS with Curriculum Learning
=====================================================
Instead of trying 64 classes at once, start with 16 and
gradually expand. The network builds capacity incrementally.
Each stage reuses the topology from the previous one.
"""

import modal
import time
import sys

app = modal.App("vraxion-v20i-curriculum")

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

    def evaluate(net, active_inputs, targets, vocab, ticks=8, n_passes=2):
        """Evaluate only on active_inputs subset."""
        total_score = 0.0
        correct = 0
        n_items = len(active_inputs)
        net.reset_state()
        prev_diff = torch.zeros(vocab)

        for pass_num in range(n_passes):
            for idx in range(n_items):
                world = torch.zeros(vocab)
                world[active_inputs[idx]] = 1.0
                logits = net.forward(world, prev_diff, ticks=ticks)
                probs = torch.softmax(logits, dim=-1)
                predicted = probs.argmax().item()
                target = targets[active_inputs[idx]].item()

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

    def evaluate_full(net, all_inputs, targets, vocab, ticks=8):
        """Evaluate on ALL 64 classes."""
        return evaluate(net, all_inputs, targets, vocab, ticks, n_passes=2)

    print(f"v20i — 64-CLASS, Curriculum Learning")
    print(f"=" * 60)
    sys.stdout.flush()

    torch.manual_seed(42); random.seed(42)
    v = 64
    perm = torch.randperm(v)
    all_inputs = list(range(v))
    n_neurons = 320
    n_in = v * 2
    n_out = v
    ticks = 8

    net = SelfWiringLeakyNet(n_neurons, n_in, n_out, addr_dim=4, density=0.06)

    # Curriculum stages
    stages = [16, 32, 48, 64]
    stage_thresholds = [0.90, 0.80, 0.70, 1.0]  # acc needed to advance

    print(f"  Neurons: {n_neurons} (internal: {n_neurons - n_in - n_out})")
    print(f"  Ticks: {ticks}")
    print(f"  Curriculum: {stages}")
    print(f"  Advance thresholds: {stage_thresholds}")

    stage_idx = 0
    active_n = stages[stage_idx]
    active_inputs = all_inputs[:active_n]

    current_score, current_acc = evaluate(net, active_inputs, perm, v, ticks)
    best_score = current_score
    best_acc_full = 0.0
    kept = 0
    stale = 0
    phase = "STRUCTURE"
    phase_switched = False
    stale_limit = 15000
    max_attempts = 300000
    total_attempts = 0
    t0 = time.time()

    print(f"  Stage 1: {active_n} classes")
    print(f"  Start: Acc={current_acc*100:.0f}% | Score: {current_score:.3f} | "
          f"Conns: {net.count_connections()}")
    sys.stdout.flush()

    for attempt in range(max_attempts):
        total_attempts += 1
        state = net.save_state()

        if phase == "STRUCTURE":
            net.mutate_structure(rate=0.05)
        else:
            if random.random() < 0.3:
                net.mutate_structure(rate=0.02)
            else:
                net.mutate_weights(scale=0.05)

        new_score, new_acc = evaluate(net, active_inputs, perm, v, ticks)

        if new_score > current_score:
            current_score = new_score
            kept += 1
            stale = 0
            if current_score > best_score:
                best_score = current_score
        else:
            net.restore_state(state)
            stale += 1

        # Phase switch
        if phase == "STRUCTURE" and stale > 2000 and not phase_switched:
            phase = "BOTH"
            phase_switched = True
            elapsed = time.time() - t0
            print(f"  -- PLATEAU at {total_attempts} ({elapsed:.1f}s), switching to BOTH --")
            sys.stdout.flush()
            stale = 0

        # Stage advancement check
        if new_acc >= stage_thresholds[stage_idx] and stage_idx < len(stages) - 1:
            stage_idx += 1
            active_n = stages[stage_idx]
            active_inputs = all_inputs[:active_n]

            # Re-evaluate on new wider set
            current_score, current_acc = evaluate(net, active_inputs, perm, v, ticks)
            best_score = current_score
            phase = "STRUCTURE"
            phase_switched = False
            stale = 0

            elapsed = time.time() - t0
            _, full_acc = evaluate_full(net, all_inputs, perm, v, ticks)
            print(f"\n  === STAGE {stage_idx+1}: {active_n} classes (at {total_attempts}, {elapsed:.1f}s) ===")
            print(f"  Acc on {active_n}: {current_acc*100:.0f}% | Full 64: {full_acc*100:.0f}% | "
                  f"Conns: {net.count_connections()}")
            sys.stdout.flush()

        if (total_attempts) % 2000 == 0:
            _, acc = evaluate(net, active_inputs, perm, v, ticks)
            _, full_acc = evaluate_full(net, all_inputs, perm, v, ticks)
            elapsed = time.time() - t0
            print(f"  [{phase:9s}] {total_attempts:6d} | "
                  f"Stage: {active_n}cl | Acc: {acc*100:.0f}% | Full: {full_acc*100:.0f}% | "
                  f"Score: {current_score:.3f} | Kept: {kept} | "
                  f"Conns: {net.count_connections()} | Stale: {stale} | {elapsed:.1f}s")
            sys.stdout.flush()
            if full_acc > best_acc_full:
                best_acc_full = full_acc

        if stage_idx == len(stages) - 1 and new_acc >= 1.0:
            elapsed = time.time() - t0
            print(f"  -> Solved ALL 64 at {total_attempts}! ({elapsed:.1f}s)")
            break

        if stale >= stale_limit:
            elapsed = time.time() - t0
            _, acc = evaluate(net, active_inputs, perm, v, ticks)
            print(f"  -> STALE at {total_attempts} ({elapsed:.1f}s) | "
                  f"Stage: {active_n}cl | Acc: {acc*100:.0f}%")
            if stage_idx < len(stages) - 1:
                # Force advance to next stage anyway
                stage_idx += 1
                active_n = stages[stage_idx]
                active_inputs = all_inputs[:active_n]
                current_score, current_acc = evaluate(net, active_inputs, perm, v, ticks)
                best_score = current_score
                phase = "STRUCTURE"
                phase_switched = False
                stale = 0
                print(f"  -> Forcing advance to {active_n} classes")
            else:
                break

    elapsed = time.time() - t0
    _, final_acc = evaluate_full(net, all_inputs, perm, v, ticks)
    if final_acc > best_acc_full:
        best_acc_full = final_acc
    print(f"\n  Final (64-class): {final_acc*100:.0f}% | Best: {best_acc_full*100:.0f}% | "
          f"Score: {best_score:.3f} | Conns: {net.count_connections()} | {elapsed:.1f}s")
    print(f"  Stages reached: {stage_idx+1}/{len(stages)}")
    sys.stdout.flush()

    return {"acc": final_acc, "best_acc": best_acc_full, "score": best_score,
            "time": elapsed, "conns": net.count_connections(),
            "stage": stage_idx + 1}


@app.local_entrypoint()
def main():
    print("Running v20i curriculum learning...")
    results = run_experiment.remote()
    print(f"\nDone! {results['acc']*100:.0f}% | Best: {results['best_acc']*100:.0f}% | "
          f"Stage: {results['stage']}/4 | Conns: {results['conns']}")
