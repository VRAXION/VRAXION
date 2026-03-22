"""Export network snapshots during training for HTML visualization."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
from model.graph import SelfWiringGraph

V = 16
BUDGET = 4000
SNAPSHOT_EVERY = 200
SEED = 42
TICKS = 8
OUT_DIR = os.path.join(os.path.dirname(__file__), 'viz')
os.makedirs(OUT_DIR, exist_ok=True)


def classify_edge(src, dst, V, N):
    out_start = N - V
    s_zone = 'input' if src < V else ('output' if src >= out_start else 'compute')
    d_zone = 'input' if dst < V else ('output' if dst >= out_start else 'compute')
    if s_zone == d_zone:
        return 'lateral'
    zone_order = {'input': 0, 'compute': 1, 'output': 2}
    return 'fwd' if zone_order[s_zone] < zone_order[d_zone] else 'fb'


def snapshot(net, step, score):
    V, N = net.V, net.N
    out_start = net.out_start
    # Run forward to get activations
    logits = net.forward_batch(TICKS)
    # Neuron data (use first input's activations as representative)
    neurons = []
    for i in range(N):
        zone = 'input' if i < V else ('output' if i >= out_start else 'compute')
        neurons.append({
            'id': i,
            'zone': zone,
            'charge': float(net.charge[i]),
            'state': float(net.state[i]),
        })
    # Edges
    edges = []
    bidir = 0
    for r, c in net.alive:
        sign = 1 if net.mask[r, c] > 0 else -1
        zone = classify_edge(r, c, V, N)
        is_bidir = net.mask[c, r] != 0
        if is_bidir:
            bidir += 1
        edges.append({
            'src': int(r), 'dst': int(c),
            'sign': sign, 'zone': zone,
            'bidir': bool(is_bidir),
        })
    # Count triangles (sample)
    triangles = 0
    for r, c in net.alive[:200]:
        for c2 in range(N):
            if net.mask[c, c2] != 0 and net.mask[c2, r] != 0:
                triangles += 1
    return {
        'V': V, 'N': N, 'step': step,
        'score': round(float(score) * 100, 1),
        'conns': len(net.alive),
        'neurons': neurons,
        'edges': edges,
        'loops': {'bidirectional': bidir // 2, 'triangles': triangles},
    }


# Train with snapshots
np.random.seed(SEED)
random.seed(SEED)
net = SelfWiringGraph(V)
targets = np.arange(V)
np.random.shuffle(targets)

def evaluate():
    logits = net.forward_batch(TICKS)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    acc = (np.argmax(probs, axis=1)[:V] == targets[:V]).mean()
    tp = probs[np.arange(V), targets[:V]].mean()
    return 0.5 * acc + 0.5 * tp

snapshots = []
score = evaluate()
best = score
stale = 0
snapshots.append(snapshot(net, 0, best))
print(f"V={V} N={net.N} budget={BUDGET}")

random.seed(SEED * 1000 + 1)
for att in range(BUDGET):
    old_loss = int(net.loss_pct)
    old_drive = int(net.drive)
    undo = net.mutate()
    new_score = evaluate()
    if new_score > score:
        score = new_score
        best = max(best, score)
        stale = 0
    else:
        net.replay(undo)
        net.loss_pct = np.int8(old_loss)
        net.drive = np.int8(old_drive)
        stale += 1
    if (att + 1) % SNAPSHOT_EVERY == 0:
        snapshots.append(snapshot(net, att + 1, best))
        print(f"  [{att+1:5d}] {best*100:.1f}% conns={len(net.alive)}")
    if best >= 0.99 or stale >= 6000:
        break

# Final snapshot
if (att + 1) % SNAPSHOT_EVERY != 0:
    snapshots.append(snapshot(net, att + 1, best))

out_path = os.path.join(OUT_DIR, 'snapshots.json')
with open(out_path, 'w') as f:
    json.dump(snapshots, f)
print(f"\nExported {len(snapshots)} snapshots to {out_path}")
