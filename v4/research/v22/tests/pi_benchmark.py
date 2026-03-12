"""
BENCHMARK — How fast is a single forward pass?
Estimate if Raspberry Pi swarm is viable.

RPi 4 is roughly 5-10x slower than a modern x86 core for numpy.
"""

import numpy as np
import time, math

def benchmark_forward(n_neurons, ticks=8, iterations=1000):
    """Time raw forward pass without any learning."""
    s = math.sqrt(2.0 / n_neurons)
    W = np.random.randn(n_neurons, n_neurons) * s
    mask = (np.random.rand(n_neurons, n_neurons) < 0.1).astype(np.float64)
    Weff = W * mask
    state = np.zeros(n_neurons)
    inp = np.random.randn(n_neurons)

    # Warmup
    for _ in range(10):
        act = state.copy()
        for t in range(ticks):
            act = act * 0.5
            act[:32] = inp[:32]
            raw = act @ Weff + act * 0.1
            act = np.where(raw > 0, raw, 0.01 * raw)

    # Timed
    t0 = time.time()
    for _ in range(iterations):
        act = state.copy()
        for t in range(ticks):
            act = act * 0.5
            act[:32] = inp[:32]
            raw = act @ Weff + act * 0.1
            act = np.where(raw > 0, raw, 0.01 * raw)
    elapsed = time.time() - t0

    per_forward = elapsed / iterations * 1000  # ms
    per_second = iterations / elapsed
    return per_forward, per_second


def benchmark_full_eval(n_neurons, n_classes, ticks=8, iterations=100):
    """Time a full evaluation (all classes, 2 passes)."""
    s = math.sqrt(2.0 / n_neurons)
    W = np.random.randn(n_neurons, n_neurons) * s
    mask = (np.random.rand(n_neurons, n_neurons) < 0.1).astype(np.float64)
    Weff = W * mask
    n_in = n_classes * 2
    n_out = n_classes

    t0 = time.time()
    for _ in range(iterations):
        state = np.zeros(n_neurons)
        prev_diff = np.zeros(n_classes)
        for p in range(2):
            for idx in range(n_classes):
                world = np.zeros(n_classes)
                world[idx] = 1.0
                inp = np.concatenate([world, prev_diff])
                act = state.copy()
                for t in range(ticks):
                    act = act * 0.5
                    act[:n_in] = inp
                    raw = act @ Weff + act * 0.1
                    act = np.where(raw > 0, raw, 0.01 * raw)
                state = act.copy()
                logits = act[-n_out:]
                e = np.exp(logits - logits.max())
                probs = e / e.sum()
                tv = np.zeros(n_classes)
                tv[idx] = 1.0
                prev_diff = tv - probs
    elapsed = time.time() - t0

    per_eval = elapsed / iterations * 1000  # ms
    evals_per_second = iterations / elapsed
    return per_eval, evals_per_second


def benchmark_mutation_cycle(n_neurons, n_classes, ticks=8, iterations=100):
    """Time a complete mutation+eval+revert cycle."""
    s = math.sqrt(2.0 / n_neurons)
    W = np.random.randn(n_neurons, n_neurons) * s
    mask = (np.random.rand(n_neurons, n_neurons) < 0.1).astype(np.float64)
    n_in = n_classes * 2; n_out = n_classes

    t0 = time.time()
    for _ in range(iterations):
        # Save
        W_saved = W.copy(); mask_saved = mask.copy()

        # Mutate
        action = np.random.randint(3)
        if action == 0:  # add
            dead = np.argwhere(mask == 0)
            if len(dead) > 0:
                n = max(1, int(len(dead) * 0.05))
                idx = dead[np.random.choice(len(dead), min(n, len(dead)), replace=False)]
                for j in range(len(idx)):
                    mask[int(idx[j][0]), int(idx[j][1])] = 1
        elif action == 1:  # remove
            alive = np.argwhere(mask == 1)
            if len(alive) > 3:
                n = max(1, int(len(alive) * 0.05))
                idx = alive[np.random.choice(len(alive), min(n, len(alive)), replace=False)]
                for j in range(len(idx)):
                    mask[int(idx[j][0]), int(idx[j][1])] = 0

        # Eval
        Weff = W * mask
        state = np.zeros(n_neurons)
        prev_diff = np.zeros(n_classes)
        for p in range(2):
            for idx_c in range(n_classes):
                world = np.zeros(n_classes); world[idx_c] = 1.0
                inp = np.concatenate([world, prev_diff])
                act = state.copy()
                for t in range(ticks):
                    act = act * 0.5; act[:n_in] = inp
                    raw = act @ Weff + act * 0.1
                    act = np.where(raw > 0, raw, 0.01 * raw)
                state = act.copy()
                logits = act[-n_out:]
                e = np.exp(logits - logits.max()); probs = e / e.sum()
                tv = np.zeros(n_classes); tv[idx_c] = 1.0
                prev_diff = tv - probs

        # Revert (50% of time, simulating bad mutation)
        if np.random.rand() < 0.5:
            W = W_saved; mask = mask_saved

    elapsed = time.time() - t0
    per_cycle = elapsed / iterations * 1000
    cycles_per_second = iterations / elapsed
    return per_cycle, cycles_per_second


print("="*65)
print("v18 PERFORMANCE BENCHMARK")
print("="*65)
print("Testing on this machine. RPi 4 is ~5-10x slower.\n")

# Forward pass speed
print("--- Single Forward Pass ---")
for n in [80, 160, 320, 640]:
    ms, fps = benchmark_forward(n)
    rpi_ms = ms * 7  # rough estimate
    print(f"  {n:4d} neurons: {ms:.3f}ms ({fps:.0f}/sec) | "
          f"RPi est: {rpi_ms:.2f}ms ({fps/7:.0f}/sec)")

# Full eval speed
print("\n--- Full Evaluation (all classes, 2 passes) ---")
for nc, nn in [(16, 80), (32, 160), (64, 320), (128, 640)]:
    ms, eps = benchmark_full_eval(nn, nc)
    rpi_ms = ms * 7
    print(f"  {nc:3d}-class, {nn:3d} neurons: {ms:.1f}ms ({eps:.0f}/sec) | "
          f"RPi est: {rpi_ms:.0f}ms ({eps/7:.0f}/sec)")

# Full mutation cycle
print("\n--- Full Mutation+Eval+Revert Cycle ---")
for nc, nn in [(16, 80), (32, 160), (64, 320)]:
    ms, cps = benchmark_mutation_cycle(nn, nc)
    rpi_ms = ms * 7
    print(f"  {nc:3d}-class, {nn:3d} neurons: {ms:.1f}ms ({cps:.0f}/sec) | "
          f"RPi est: {rpi_ms:.0f}ms ({cps/7:.0f}/sec)")

# Swarm estimates
print("\n--- SWARM ESTIMATES ---")
print("Assumptions: RPi 4, 7x slower than this machine, island model\n")

for n_pis, nc, nn in [(4, 32, 160), (8, 32, 160), (4, 64, 320), (8, 64, 320), (16, 64, 320)]:
    _, cps_here = benchmark_mutation_cycle(nn, nc)
    cps_pi = cps_here / 7  # per Pi
    total_cps = cps_pi * n_pis  # all Pis combined

    # Time estimates
    attempts_32class = 8000  # what v18 needs for 32-class
    attempts_64class = 50000  # estimate for 64-class
    target = attempts_64class if nc == 64 else attempts_32class

    time_single = target / cps_pi
    time_swarm = target / total_cps

    print(f"  {n_pis:2d}x RPi4, {nc}-class ({nn}n): "
          f"{cps_pi:.0f} cycles/sec/Pi, {total_cps:.0f} total → "
          f"single:{time_single:.0f}s, swarm:{time_swarm:.0f}s "
          f"({time_swarm/60:.0f}min)")

print("\n--- MEMORY USAGE ---")
for nn in [80, 160, 320, 640, 1280]:
    # W + mask + state + addresses + target_W
    mem_bytes = (nn*nn*8*2 + nn*8 + nn*4*8*2)  # float64
    mem_mb = mem_bytes / 1024 / 1024
    print(f"  {nn:4d} neurons: {mem_mb:.1f} MB (RPi4 has 1-8GB RAM)")
