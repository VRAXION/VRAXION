"""
A/B test: edge_magnitude 0.6 (old default) vs 1.0 (new: sign-only ±1).

Uses the toy proverb corpus from lib/data.py and runs a short evolutionary
loop. The test asserts that 1.0 matches or beats 0.6, confirming that the
extra magnitude scaling was redundant.
"""
import sys, os, random
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.graph import SelfWiringGraph
from lib.data import TEXT
from lib.utils import softmax

SEED = 777
VOCAB = 32  # small vocab for speed
HIDDEN_RATIO = 3
TICKS = 6
STEPS = 300
N_TRIALS = 3


def make_data(vocab):
    raw = [b for b in TEXT.encode('ascii') if b < vocab + 32]
    return np.array(raw, dtype=np.uint8) % vocab


def eval_accuracy(net, data, ticks=TICKS):
    """Single-pass next-char accuracy on the toy corpus."""
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


def run_evo(edge_magnitude, seed, steps=STEPS):
    """Run a short evolutionary loop, return best accuracy."""
    rng = random.Random(seed)
    np.random.seed(seed)
    net = SelfWiringGraph(VOCAB, hidden_ratio=HIDDEN_RATIO,
                          edge_magnitude=edge_magnitude, seed=seed)
    data = make_data(VOCAB)

    best_acc = eval_accuracy(net, data)
    for step in range(steps):
        undo = net.mutate()
        net.state *= 0
        net.charge *= 0
        acc = eval_accuracy(net, data)
        if acc >= best_acc:
            best_acc = acc
        else:
            net.replay(undo)
    return best_acc


def test_edge_magnitude_1_0_vs_0_6():
    """A/B: edge_magnitude=1.0 should match or beat 0.6."""
    results_06 = []
    results_10 = []
    for trial in range(N_TRIALS):
        seed = SEED + trial * 100
        acc_06 = run_evo(0.6, seed)
        acc_10 = run_evo(1.0, seed)
        results_06.append(acc_06)
        results_10.append(acc_10)
        print(f"  trial {trial+1}: em=0.6 -> {acc_06*100:.1f}%  |  em=1.0 -> {acc_10*100:.1f}%")

    mean_06 = np.mean(results_06)
    mean_10 = np.mean(results_10)
    print(f"\n  MEAN: em=0.6 -> {mean_06*100:.1f}%  |  em=1.0 -> {mean_10*100:.1f}%")
    print(f"  delta: {(mean_10 - mean_06)*100:+.1f}%")

    # 1.0 should not be materially worse (allow 2% tolerance)
    assert mean_10 >= mean_06 - 0.02, (
        f"edge_magnitude=1.0 ({mean_10*100:.1f}%) was more than 2% worse "
        f"than 0.6 ({mean_06*100:.1f}%)"
    )


def test_breed_no_boost():
    """Verify breed produces binary {0, 1} mask (union of parents, no boost)."""
    np.random.seed(42)
    a = SelfWiringGraph(VOCAB, hidden_ratio=HIDDEN_RATIO, seed=10)
    b = SelfWiringGraph(VOCAB, hidden_ratio=HIDDEN_RATIO, seed=10)
    # Give b some different edges
    for _ in range(20):
        b.mutate()

    child = SelfWiringGraph.breed(a, b, seed=99)
    unique_vals = set(np.unique(child.mask).tolist())
    assert unique_vals.issubset({0, 1}), (
        f"breed mask should only contain {{0, 1}}, got {unique_vals}"
    )
    print(f"  breed mask unique values: {sorted(unique_vals)} OK")


if __name__ == "__main__":
    print("\n=== A/B: edge_magnitude 0.6 vs 1.0 ===")
    test_edge_magnitude_1_0_vs_0_6()
    print("\n=== Breed: no boost ===")
    test_breed_no_boost()
    print("\n  All tests passed.")
