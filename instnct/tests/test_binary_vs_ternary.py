"""
Test: does the network actually need negative edges?

Since act >= 0 after ReLU and charge >= 0 always, negative edges provide
inhibition (reduce charge). This test checks:
1. What's the actual pos/neg ratio in evolved networks?
2. Does removing negative edges hurt accuracy?
3. Can a binary {0, 1} mask match ternary {-1, 0, +1}?
"""
import sys, os, random
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.graph import SelfWiringGraph
from lib.data import TEXT

SEED = 777
VOCAB = 32
HIDDEN_RATIO = 3
TICKS = 6
STEPS = 300
N_TRIALS = 3


def make_data(vocab):
    raw = [b for b in TEXT.encode('ascii') if b < vocab + 32]
    return np.array(raw, dtype=np.uint8) % vocab


def eval_accuracy(net, data, ticks=TICKS):
    correct = 0
    total = 0
    net.state *= 0
    net.charge *= 0
    for i in range(len(data) - 1):
        world = np.zeros(net.V, dtype=np.float32)
        world[data[i]] = 1.0
        logits = net.forward(world, ticks=ticks)
        if np.argmax(logits) == data[i + 1]:
            correct += 1
        total += 1
    return correct / total if total else 0.0


def run_evo(seed, steps=STEPS, binary_only=False):
    """Run evolutionary loop. If binary_only, force all edges to +1."""
    rng = random.Random(seed)
    np.random.seed(seed)
    net = SelfWiringGraph(VOCAB, hidden_ratio=HIDDEN_RATIO, seed=seed)
    data = make_data(VOCAB)

    if binary_only:
        # Convert all -1 edges to +1
        net.mask[net.mask == -1] = 1
        net.resync_alive()

    best_acc = eval_accuracy(net, data)
    for step in range(steps):
        undo = net.mutate()

        if binary_only:
            # After mutation, force any -1 to +1
            flipped = False
            for entry in undo:
                if entry[0] == 'A':
                    r, c = entry[1], entry[2]
                    if net.mask[r, c] == -1:
                        net.mask[r, c] = 1
                        flipped = True
                elif entry[0] == 'F':
                    # Flip would create -1 from +1, undo it
                    r, c = entry[1], entry[2]
                    if net.mask[r, c] == -1:
                        net.mask[r, c] = 1
                        flipped = True
            if flipped:
                net.resync_alive()

        net.state *= 0
        net.charge *= 0
        acc = eval_accuracy(net, data)
        if acc >= best_acc:
            best_acc = acc
        else:
            net.replay(undo)
            if binary_only:
                # Ensure binary after replay too
                net.mask[net.mask == -1] = 1
                net.resync_alive()

    # Report pos/neg ratio
    pos = int(np.sum(net.mask > 0))
    neg = int(np.sum(net.mask < 0))
    return best_acc, pos, neg


def test_binary_vs_ternary():
    print("=== Binary {0,1} vs Ternary {-1,0,+1} ===\n")

    ternary_accs = []
    binary_accs = []

    for trial in range(N_TRIALS):
        seed = SEED + trial * 100

        acc_t, pos_t, neg_t = run_evo(seed, binary_only=False)
        acc_b, pos_b, neg_b = run_evo(seed, binary_only=True)

        ternary_accs.append(acc_t)
        binary_accs.append(acc_b)

        print(f"  trial {trial+1}:")
        print(f"    ternary: {acc_t*100:.1f}%  (pos={pos_t}, neg={neg_t}, ratio={neg_t/(pos_t+neg_t+1e-10)*100:.0f}% neg)")
        print(f"    binary:  {acc_b*100:.1f}%  (pos={pos_b}, neg={neg_b})")

    mean_t = np.mean(ternary_accs)
    mean_b = np.mean(binary_accs)
    print(f"\n  MEAN ternary: {mean_t*100:.1f}%")
    print(f"  MEAN binary:  {mean_b*100:.1f}%")
    print(f"  delta:        {(mean_b - mean_t)*100:+.1f}%")

    if mean_b >= mean_t - 0.02:
        print("\n  → Binary mask is viable! Negative edges are not critical.")
    else:
        print(f"\n  → Ternary wins by {(mean_t - mean_b)*100:.1f}%. Inhibition matters.")


def test_check_activations():
    """Check if negative values actually flow through the network."""
    np.random.seed(42)
    net = SelfWiringGraph(VOCAB, hidden_ratio=HIDDEN_RATIO, seed=42)
    data = make_data(VOCAB)

    # Evolve a bit
    for _ in range(100):
        undo = net.mutate()
        net.replay(undo)  # just exercise the network

    net.state *= 0
    net.charge *= 0

    act_mins = []
    charge_mins = []
    raw_mins = []

    # Manual rollout to inspect values
    for i in range(min(20, len(data) - 1)):
        world = np.zeros(net.V, dtype=np.float32)
        world[data[i]] = 1.0
        injected = world @ net.input_projection

        for tick in range(TICKS):
            if tick < 1:
                net.state = net.state + injected
            act_mins.append(float(net.state.min()))

            raw = net._sparse_mul_1d(net.state)
            raw_mins.append(float(raw.min()))

            net.charge += raw
            net.charge *= (1.0 - net.decay)
            net.state = np.maximum(net.charge - net.theta, 0.0)
            net.charge = np.maximum(net.charge, 0.0)
            charge_mins.append(float(net.charge.min()))

    print("\n=== Activation analysis ===\n")
    print(f"  act min:    {min(act_mins):.4f}  (before ReLU clamp, after injection)")
    print(f"  raw min:    {min(raw_mins):.4f}  (sparse mul output, before charge add)")
    print(f"  charge min: {min(charge_mins):.4f}  (after ReLU clamp)")
    print(f"  act < 0?    {'YES' if min(act_mins) < 0 else 'NO'}")
    print(f"  charge < 0? {'NO (clamped)'}")


if __name__ == "__main__":
    test_check_activations()
    print()
    test_binary_vs_ternary()
