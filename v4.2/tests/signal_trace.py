"""Trace: signal amplitude per tick — WHY does the signal die?"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from model.graph import SelfWiringGraph
from lib.utils import score_batch, train_cyclic

V = 16
TICKS = 16

# Train a decent network first
np.random.seed(42)
targets = np.random.randint(0, V, size=V)
net = SelfWiringGraph(V)
train_cyclic(net, targets, V, score_fn=score_batch, ticks=6,
             max_att=20000, stale_limit=2000,
             crystal_budget=3000, crystal_window=200,
             crystal_min_rate=0.003, verbose=False)

print(f"Trained net: conns={net.count_connections()}, retain={net.retention:.3f}, loss_pct={int(net.loss_pct)}")
print()

# Now trace forward_batch step by step
N = net.N
charges = np.zeros((V, N), dtype=np.float32)
acts = np.zeros((V, N), dtype=np.float32)
retain = float(net.retention)

print(f"{'tick':>4s} | {'mean|charge|':>12s} {'max|charge|':>12s} {'active':>8s} "
      f"{'mean|act|':>10s} {'out_energy':>11s} {'correct':>8s}")
print(f"{'-'*4}-+-{'-'*12}-{'-'*12}-{'-'*8}-{'-'*10}-{'-'*11}-{'-'*8}")

out_s = net.out_start
for t in range(TICKS):
    if t == 0:
        acts[:, :V] = np.eye(V, dtype=np.float32)
    raw = acts @ net.mask
    np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    charges += raw
    charges *= retain
    acts = np.maximum(charges - net.THRESHOLD, 0.0)
    charges = np.clip(charges, -1.0, 1.0)

    # Diagnostics
    abs_charge = np.abs(charges)
    mean_ch = abs_charge.mean()
    max_ch = abs_charge.max()
    active = (acts > 0).sum()  # total active neurons across all V inputs
    mean_act = acts[acts > 0].mean() if active > 0 else 0
    out_logits = charges[:, out_s:out_s + V]
    out_energy = np.abs(out_logits).mean()
    preds = np.argmax(out_logits, axis=1)
    correct = (preds == targets).sum()

    bar_ch = '#' * int(mean_ch * 80)
    bar_out = '*' * int(out_energy * 80)
    print(f"  {t:2d}  | {mean_ch:11.4f}  {max_ch:11.4f}  {active:7d}  "
          f"{mean_act:9.4f}  {out_energy:10.4f}  {correct:3d}/{V}  "
          f"|{bar_ch}")

# What the score would be at each tick cutoff
print(f"\n{'tick':>4s} | {'score':>8s}")
print(f"{'-'*4}-+-{'-'*8}")
for t_cut in [2, 4, 6, 8, 10, 12, 16]:
    charges2 = np.zeros((V, N), dtype=np.float32)
    acts2 = np.zeros((V, N), dtype=np.float32)
    for t in range(t_cut):
        if t == 0:
            acts2[:, :V] = np.eye(V, dtype=np.float32)
        raw = acts2 @ net.mask
        np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        charges2 += raw
        charges2 *= retain
        acts2 = np.maximum(charges2 - net.THRESHOLD, 0.0)
        charges2 = np.clip(charges2, -1.0, 1.0)
    sc, acc = score_batch(net, targets, V, ticks=t_cut)
    print(f"  {t_cut:2d}  | {sc*100:5.1f}%  acc={acc*100:.0f}%")
