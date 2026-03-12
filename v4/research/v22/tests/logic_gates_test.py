"""
v22 Logic Test: Boolean Gates (AND/OR/XOR on 4-bit inputs)
==========================================================
Tests basic logical operations. XOR is the hard one —
requires nonlinear combination that can't be solved by a single connection.

Input: 4-bit binary vector (one-hot per bit = 8 neurons)
Output: 2 classes (0 or 1)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import time
from v22_best_config import SelfWiringGraph, softmax


def make_gate_dataset(gate_fn, n_bits=4):
    """Generate all 2^n_bits inputs and apply gate_fn pairwise."""
    inputs = []
    targets = []
    for val in range(2 ** n_bits):
        bits = [(val >> i) & 1 for i in range(n_bits)]
        inputs.append(bits)
        # Apply gate between first two bits
        targets.append(gate_fn(bits[0], bits[1]))
    return inputs, targets


def train_gate(net, bit_inputs, targets, n_bits, max_attempts=8000, ticks=6):
    """Train on gate task. Input = one-hot per bit (2*n_bits neurons). Output = 2 classes."""
    input_dim = n_bits * 2  # one-hot per bit: [bit0=0, bit0=1, bit1=0, bit1=1, ...]
    V = 2  # binary output

    def encode(bits):
        world = np.zeros(input_dim, dtype=np.float32)
        for i, b in enumerate(bits):
            world[i * 2 + b] = 1.0
        return world

    def evaluate():
        net.reset()
        correct = 0
        for p in range(2):
            for i in range(len(bit_inputs)):
                world = encode(bit_inputs[i])
                logits = net.forward(world, ticks)
                probs = softmax(logits[:V])
                if p == 1 and np.argmax(probs) == targets[i]:
                    correct += 1
        acc = correct / len(bit_inputs)
        net.last_acc = acc
        return acc

    score = evaluate()
    best = score
    phase = "STRUCTURE"
    kept = 0
    stale = 0
    switched = False

    for att in range(max_attempts):
        state = net.save_state()

        if phase == "STRUCTURE":
            net.mutate_structure(0.05)
        else:
            if random.random() < 0.3:
                net.mutate_structure(0.02)
            else:
                net.mutate_weights()

        net.self_wire()
        new_score = evaluate()

        if new_score > score:
            score = new_score
            kept += 1
            stale = 0
            best = max(best, score)
        else:
            net.restore_state(state)
            stale += 1

        if phase == "STRUCTURE" and stale > 2500 and not switched:
            phase = "BOTH"
            switched = True
            stale = 0

        if (att + 1) % 1000 == 0:
            print(f"  [{att+1:5d}] Acc: {best*100:5.1f}% | "
                  f"Conns: {net.count_connections():4d} | "
                  f"Kept: {kept:3d} | Phase: {phase}")

        if best >= 0.99 or stale >= 6000:
            break

    return best, kept


if __name__ == "__main__":
    SEED = 42
    N_NEURONS = 256
    N_BITS = 4

    gates = [
        ("AND",  lambda a, b: a & b),
        ("OR",   lambda a, b: a | b),
        ("XOR",  lambda a, b: a ^ b),
        ("NAND", lambda a, b: 1 - (a & b)),
        ("XNOR", lambda a, b: 1 - (a ^ b)),
    ]

    print("=" * 60)
    print(f"v22 Logic Test: Boolean Gates ({N_BITS}-bit inputs)")
    print("=" * 60)

    results = []
    for name, fn in gates:
        np.random.seed(SEED)
        random.seed(SEED)

        bit_inputs, targets = make_gate_dataset(fn, N_BITS)
        input_dim = N_BITS * 2
        net = SelfWiringGraph(N_NEURONS, input_dim)

        print(f"\n--- {name} gate ---")
        print(f"  Dataset: {len(bit_inputs)} inputs, 2 classes")
        # Show class balance
        n1 = sum(targets)
        print(f"  Balance: {n1} ones / {len(targets)-n1} zeros")

        t0 = time.time()
        best, kept = train_gate(net, bit_inputs, targets, N_BITS)
        elapsed = time.time() - t0

        majority = max(n1, len(targets)-n1) / len(targets)
        print(f"\n  Result: {best*100:.1f}% (majority: {majority*100:.1f}%) "
              f"| {elapsed:.1f}s | kept: {kept}")
        results.append((name, best, majority, elapsed))

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    fmt = "  {:<10s} {:>8s} {:>10s} {:>8s}"
    print(fmt.format("Gate", "Acc", "Majority", "Time"))
    print(fmt.format("-"*10, "-"*8, "-"*10, "-"*8))
    for name, best, maj, t in results:
        marker = " ***" if best > maj + 0.05 else ""
        print(fmt.format(name, f"{best*100:.1f}%", f"{maj*100:.1f}%", f"{t:.1f}s") + marker)
