"""
Smoke test: Does three-factor surprise learning work at all?
Compares: mutation+selection vs surprise-only vs hybrid on V=16
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from model.graph import SelfWiringGraph
from model.surprise_graph import SurpriseGraph, train_surprise, train_hybrid
from lib.utils import score_batch

V = 16
TICKS = 6
SEEDS = [42, 77, 123]
MAX_STEPS = 3000


def make_targets(V, seed):
    rng = np.random.RandomState(seed)
    return rng.randint(0, V, size=V)


def test_mutation_baseline(V, targets, seed):
    """Original: mutation + selection."""
    np.random.seed(seed)
    import random; random.seed(seed)
    net = SelfWiringGraph(V)
    sc, acc = score_batch(net, targets, V, TICKS)
    best_sc = sc

    for att in range(MAX_STEPS):
        state = net.save_state()
        old_loss = int(net.loss_pct)
        undo = net.mutate()
        sc, acc = score_batch(net, targets, V, TICKS)
        if sc > best_sc:
            best_sc = sc
        else:
            net.replay(undo)
            net.loss_pct = np.int8(old_loss)

    _, final_acc = score_batch(net, targets, V, TICKS)
    return final_acc


def test_surprise_only(V, targets, seed):
    """New: three-factor local learning only."""
    np.random.seed(seed)
    import random; random.seed(seed)
    net = SurpriseGraph(V)
    best_acc = train_surprise(net, targets, V, ticks=TICKS,
                              max_steps=MAX_STEPS, lr=0.01, verbose=False)
    return best_acc


def test_hybrid(V, targets, seed):
    """Hybrid: surprise + exploratory mutation."""
    np.random.seed(seed)
    import random; random.seed(seed)
    net = SurpriseGraph(V)
    best_acc = train_hybrid(net, targets, V, ticks=TICKS,
                            max_steps=MAX_STEPS, lr=0.01,
                            explore_every=50, verbose=False)
    return best_acc


if __name__ == '__main__':
    print(f"Surprise Learning Smoke Test — V={V}, ticks={TICKS}, steps={MAX_STEPS}")
    print(f"{'='*60}")

    results = {'mutation': [], 'surprise': [], 'hybrid': []}

    for seed in SEEDS:
        targets = make_targets(V, seed + 1000)
        print(f"\nSeed {seed}:")

        acc = test_mutation_baseline(V, targets, seed)
        results['mutation'].append(acc)
        print(f"  Mutation+selection: {acc*100:.1f}%")

        acc = test_surprise_only(V, targets, seed)
        results['surprise'].append(acc)
        print(f"  Surprise-only:     {acc*100:.1f}%")

        acc = test_hybrid(V, targets, seed)
        results['hybrid'].append(acc)
        print(f"  Hybrid:            {acc*100:.1f}%")

    print(f"\n{'='*60}")
    print(f"Mean accuracy across {len(SEEDS)} seeds:")
    for name, accs in results.items():
        mean = np.mean(accs) * 100
        print(f"  {name:20s}: {mean:.1f}%")
