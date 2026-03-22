"""A/B test: MT19937 vs Phi-Pi RNG on 64-class SWG training."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random as stdlib_random
from model.graph import SelfWiringGraph, train
from model.phipi_rng import PhiPiRNG

SEEDS = [0, 1, 2, 10, 42]
VOCAB = 64
BUDGET = 16000


def run_mt(seed):
    """Standard MT19937 run."""
    np.random.seed(seed)
    stdlib_random.seed(seed)
    net = SelfWiringGraph(VOCAB)
    targets = np.arange(VOCAB)
    np.random.shuffle(targets)
    best = train(net, targets, VOCAB, max_attempts=BUDGET, verbose=False)
    return best * 100


def run_phipi(seed):
    """Phi-Pi RNG replacing stdlib random for mutations."""
    np.random.seed(seed)
    stdlib_random.seed(seed)
    net = SelfWiringGraph(VOCAB)
    targets = np.arange(VOCAB)
    np.random.shuffle(targets)

    # Monkey-patch: replace random module functions with PhiPiRNG
    rng = PhiPiRNG(seed)
    old_randint = stdlib_random.randint
    old_choice = stdlib_random.choice
    stdlib_random.randint = rng.randint
    stdlib_random.choice = rng.choice

    try:
        best = train(net, targets, VOCAB, max_attempts=BUDGET, verbose=False)
    finally:
        stdlib_random.randint = old_randint
        stdlib_random.choice = old_choice

    return best * 100


print(f"A/B Test: MT19937 vs Phi-Pi RNG | V={VOCAB} budget={BUDGET}")
print(f"{'seed':>6s}  {'MT19937':>8s}  {'Phi-Pi':>8s}  {'diff':>8s}")
print("-" * 38)

mt_scores = []
pp_scores = []

for seed in SEEDS:
    t0 = time.time()
    mt = run_mt(seed)
    t_mt = time.time() - t0

    t0 = time.time()
    pp = run_phipi(seed)
    t_pp = time.time() - t0

    mt_scores.append(mt)
    pp_scores.append(pp)
    diff = pp - mt
    print(f"{seed:6d}  {mt:7.1f}%  {pp:7.1f}%  {diff:+7.1f}%  (MT:{t_mt:.0f}s PP:{t_pp:.0f}s)")
    sys.stdout.flush()

mt_avg = sum(mt_scores) / len(mt_scores)
pp_avg = sum(pp_scores) / len(pp_scores)
print("-" * 38)
print(f"{'avg':>6s}  {mt_avg:7.1f}%  {pp_avg:7.1f}%  {pp_avg - mt_avg:+7.1f}%")
