"""Quick smoke test for cyclic training."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from model.graph import SelfWiringGraph
from lib.utils import score_batch, train_cyclic

V = 16
np.random.seed(42)
targets = np.random.randint(0, V, size=V)
net = SelfWiringGraph(V)

print(f"V={V}, N={net.N}, initial conns={net.count_connections()}")
print(f"Targets: {targets}")

best_sc, best_acc, kept, cycles = train_cyclic(
    net, targets, V,
    score_fn=score_batch,
    ticks=8,
    max_att=5000,
    stale_limit=1500,
    add_every=50,
    crystal_plateau=300,
    verbose=True,
)

print(f"\nResult: score={best_sc*100:.1f}% acc={best_acc*100:.1f}% "
      f"kept={kept} cycles={cycles} conns={net.count_connections()}")
