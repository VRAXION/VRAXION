"""Frozen mutation test: Python dumps mutations, C replays them.
This isolates whether the gap is in mutation SELECTION or forward/eval."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from pathlib import Path
import numpy as np, random, json
from model.graph import SelfWiringGraph

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"

V, N = 64, 192
np.random.seed(42); random.seed(42)
net = SelfWiringGraph(V)
perm = np.random.permutation(V)

# Dump init mask
FIXTURE_DIR.mkdir(exist_ok=True)
net.mask.tofile(FIXTURE_DIR / 'frozen_mask_v64.bin')
perm.astype(np.int32).tofile(FIXTURE_DIR / 'frozen_targets_v64.bin')

# Run 2000 steps, log every mutation + accept/reject + score
def evaluate(net, perm):
    logits = net.forward_batch(8)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    acc = (np.argmax(probs, axis=1)[:V] == perm[:V]).mean()
    tp = probs[np.arange(V), perm[:V]].mean()
    return 0.5 * acc + 0.5 * tp

score = evaluate(net, perm)
log = []

for att in range(2000):
    old_loss = int(net.loss_pct)
    old_mask = net.mask.copy()
    undo = net.mutate()
    new_score = evaluate(net, perm)

    accepted = new_score > score

    log.append({
        'att': att,
        'undo_len': len(undo),
        'undo_ops': [u[0] for u in undo],
        'score_before': float(score),
        'score_after': float(new_score),
        'accepted': bool(accepted),
        'loss_pct': int(net.loss_pct),
        'signal': int(net.signal),
        'grow': int(net.grow),
        'intensity': int(net.intensity),
        'conns': net.count_connections(),
    })

    if accepted:
        score = new_score
    else:
        net.replay(undo)
        net.loss_pct = np.int8(old_loss)
        if random.randint(1, 20) <= 7:
            net.signal = np.int8(1 - int(net.signal))
        if random.randint(1, 20) <= 7:
            net.grow = np.int8(1 - int(net.grow))

# Save log
with open(FIXTURE_DIR / 'frozen_mutation_log.json', 'w') as f:
    json.dump(log, f, indent=1)

# Summary
accepts = sum(1 for l in log if l['accepted'])
print(f'V={V} N={N} 2000 steps')
print(f'Accepts: {accepts}/2000 ({100*accepts/2000:.1f}%)')
print(f'Final score: {score*100:.1f}%')
print(f'Final conns: {net.count_connections()}')
print(f'Score trajectory (every 200):')
for i in range(0, 2000, 200):
    print(f'  step {i:4d}: score={log[i]["score_before"]*100:.1f}% conns={log[i]["conns"]} '
          f'sig={log[i]["signal"]} grow={log[i]["grow"]} int={log[i]["intensity"]} loss={log[i]["loss_pct"]}')
