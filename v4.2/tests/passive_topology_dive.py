"""
PassiveIO Topology Deep Dive
==============================
Same analysis as topology_deep_dive.py but for PassiveIOGraph.
Key question: do hidden neurons self-organize into functional clusters
when forced to do ALL the computation?
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random as pyrandom
from model.passive_io import PassiveIOGraph, train_passive


def analyze_passive_topology(net, label=""):
    H = net.H
    V = net.V
    mask = net.mask
    W_in = net.W_in
    W_out = net.W_out

    print(f"\n{'='*75}")
    print(f"  TOPOLOGY: {label}")
    print(f"  V={V}, H={H} (hidden only, no IN/OUT neurons)")
    print(f"{'='*75}")

    # ── 1. GLOBAL STATS ──
    alive = (mask != 0)
    n_edges = int(alive.sum())
    n_excit = int((mask > 0).sum())
    n_inhib = int((mask < 0).sum())
    max_possible = H * H - H
    density = n_edges / max_possible * 100

    print(f"\n  GLOBAL:")
    print(f"    Edges:  {n_edges} / {max_possible} ({density:.1f}%)")
    print(f"    Excit:  {n_excit} ({n_excit/max(n_edges,1)*100:.0f}%)")
    print(f"    Inhib:  {n_inhib} ({n_inhib/max(n_edges,1)*100:.0f}%)")

    # ── 2. PER-NEURON DEGREE ──
    in_deg = alive.sum(axis=0)
    out_deg = alive.sum(axis=1)
    in_excit = (mask > 0).sum(axis=0)
    in_inhib = (mask < 0).sum(axis=0)
    out_excit = (mask > 0).sum(axis=1)
    out_inhib = (mask < 0).sum(axis=1)
    total_deg = in_deg + out_deg

    print(f"\n  PER-NEURON DEGREE (all {H} hidden neurons):")
    print(f"    {'':>12s}  {'min':>4s}  {'max':>4s}  {'mean':>5s}  {'median':>6s}  {'std':>5s}")
    for name, arr in [("in-degree", in_deg), ("out-degree", out_deg),
                      ("in-excit", in_excit), ("in-inhib", in_inhib),
                      ("out-excit", out_excit), ("out-inhib", out_inhib)]:
        print(f"    {name:>12s}  {int(arr.min()):4d}  {int(arr.max()):4d}  "
              f"{arr.mean():5.1f}  {np.median(arr):6.1f}  {arr.std():5.1f}")

    # ── 3. INPUT COUPLING: mennyire "hallja" minden neuron az inputot ──
    # W_in[v, h] = how much input v activates hidden neuron h
    # High abs value = strong coupling to that input
    print(f"\n  INPUT COUPLING (W_in amplitúdó per hidden neuron):")
    input_strength = np.abs(W_in).sum(axis=0)  # per hidden neuron: total input sensitivity
    print(f"    Total input sensitivity per hidden neuron:")
    print(f"      min={input_strength.min():.3f}  max={input_strength.max():.3f}  "
          f"mean={input_strength.mean():.3f}  std={input_strength.std():.3f}")

    # How many inputs strongly couple to each neuron?
    threshold_strong = np.abs(W_in).mean() + np.abs(W_in).std()
    strong_inputs_per_neuron = (np.abs(W_in) > threshold_strong).sum(axis=0)
    print(f"    Strongly coupled inputs per neuron (threshold={threshold_strong:.3f}):")
    print(f"      min={int(strong_inputs_per_neuron.min())}  max={int(strong_inputs_per_neuron.max())}  "
          f"mean={strong_inputs_per_neuron.mean():.1f}")

    # ── 4. OUTPUT COUPLING: mennyire "beszél" minden neuron az outputba ──
    output_strength = np.abs(W_out).sum(axis=1)  # per hidden neuron: total output influence
    print(f"\n  OUTPUT COUPLING (W_out amplitúdó per hidden neuron):")
    print(f"    Total output influence per hidden neuron:")
    print(f"      min={output_strength.min():.3f}  max={output_strength.max():.3f}  "
          f"mean={output_strength.mean():.3f}  std={output_strength.std():.3f}")

    strong_outputs_per_neuron = (np.abs(W_out) > threshold_strong).sum(axis=1)
    print(f"    Strongly coupled outputs per neuron:")
    print(f"      min={int(strong_outputs_per_neuron.min())}  max={int(strong_outputs_per_neuron.max())}  "
          f"mean={strong_outputs_per_neuron.mean():.1f}")

    # ── 5. FUNCTIONAL ROLE: input-facing vs output-facing vs relay ──
    # Classify neurons by their coupling
    input_facing = input_strength > np.median(input_strength)
    output_facing = output_strength > np.median(output_strength)
    both = input_facing & output_facing
    relay = ~input_facing & ~output_facing  # weakly coupled to both

    n_in_facing = int(input_facing.sum())
    n_out_facing = int(output_facing.sum())
    n_both = int(both.sum())
    n_relay = int(relay.sum())

    print(f"\n  FUNKCIONÁLIS SZEREPEK (medián felett = erős):")
    print(f"    Input-facing (erős input coupling):   {n_in_facing}/{H}")
    print(f"    Output-facing (erős output coupling):  {n_out_facing}/{H}")
    print(f"    Both (mindkettő erős):                 {n_both}/{H}")
    print(f"    Relay (mindkettő gyenge):              {n_relay}/{H}")

    # ── 6. DEGREE vs FUNCTIONAL ROLE ──
    roles = np.zeros(H, dtype='U10')
    for i in range(H):
        if input_facing[i] and output_facing[i]:
            roles[i] = 'BOTH'
        elif input_facing[i]:
            roles[i] = 'IN-face'
        elif output_facing[i]:
            roles[i] = 'OUT-face'
        else:
            roles[i] = 'RELAY'

    print(f"\n  DEGREE BY FUNCTIONAL ROLE:")
    print(f"    {'Role':<10s} {'Count':>5s} │ {'In-deg':>7s} {'Out-deg':>8s} │ {'Excit%':>7s}")
    print(f"    {'─'*48}")
    for role in ['IN-face', 'OUT-face', 'BOTH', 'RELAY']:
        idx = np.where(roles == role)[0]
        if len(idx) == 0:
            continue
        avg_in = in_deg[idx].mean()
        avg_out = out_deg[idx].mean()
        exc_pct = in_excit[idx].sum() / max(1, in_deg[idx].sum()) * 100
        print(f"    {role:<10s} {len(idx):>5d} │ {avg_in:>7.1f} {avg_out:>8.1f} │ {exc_pct:>6.0f}%")

    # ── 7. HUB DETECTION ──
    top_k = 10
    top_idx = np.argsort(total_deg)[::-1][:top_k]

    print(f"\n  TOP {top_k} HUBOK:")
    print(f"    {'#':>3s} {'Neuron':>6s} {'Role':>9s} {'Total':>6s} │ "
          f"{'In':>3s} {'Out':>4s} │ {'InStr':>6s} {'OutStr':>7s}")
    print(f"    {'─'*58}")
    for rank, idx in enumerate(top_idx):
        print(f"    {rank+1:3d} {idx:6d} {roles[idx]:>9s} {int(total_deg[idx]):6d} │ "
              f"{int(in_deg[idx]):3d} {int(out_deg[idx]):4d} │ "
              f"{input_strength[idx]:>6.2f} {output_strength[idx]:>7.2f}")

    # ── 8. ISOLATED / WEAK NEURONS ──
    isolated = int((total_deg == 0).sum())
    weak = int((total_deg <= 2).sum())
    print(f"\n  IZOLÁLT / GYENGE NEURONOK:")
    print(f"    Teljesen izolált (deg=0): {isolated}/{H}")
    print(f"    Gyenge (deg≤2):           {weak}/{H}")

    # ── 9. CLUSTER DETECTION: connected components ──
    adj = {i: set() for i in range(H)}
    rows, cols = np.where(mask != 0)
    for r, c in zip(rows.tolist(), cols.tolist()):
        adj[r].add(c)
        adj[c].add(r)  # undirected for component analysis

    visited = set()
    components = []
    for start in range(H):
        if start in visited:
            continue
        comp = set()
        queue = [start]
        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            comp.add(node)
            for nb in adj[node]:
                if nb not in visited:
                    queue.append(nb)
        components.append(comp)

    components.sort(key=len, reverse=True)
    print(f"\n  ÖSSZEFÜGGŐ KOMPONENSEK (undirected):")
    print(f"    Összes komponens: {len(components)}")
    for i, comp in enumerate(components[:5]):
        roles_in_comp = [roles[n] for n in comp]
        role_counts = {}
        for r in roles_in_comp:
            role_counts[r] = role_counts.get(r, 0) + 1
        role_str = ", ".join(f"{k}:{v}" for k, v in sorted(role_counts.items()))
        print(f"    #{i+1}: {len(comp)} neuron ({role_str})")

    # ── 10. SIGNAL FLOW SIMULATION: which hidden neurons actually fire? ──
    print(f"\n  SIGNAL FLOW SZIMULÁCIÓ (8 tick, all inputs):")
    charges = np.zeros((V, H), dtype=np.float32)
    acts = np.zeros((V, H), dtype=np.float32)
    retain = float(net.retention)
    projected = np.eye(V, dtype=np.float32) @ W_in

    fired_per_tick = []
    for t in range(8):
        if t == 0:
            acts += projected
        raw = acts @ mask
        np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        charges += raw
        charges *= retain
        acts = np.maximum(charges - net.THRESHOLD, 0.0)
        charges = np.clip(charges, -1.0, 1.0)
        # How many neurons fire across all inputs?
        firing = (acts > 0).any(axis=0)  # per neuron: did it fire for ANY input?
        fired_per_tick.append(int(firing.sum()))

    for t, cnt in enumerate(fired_per_tick):
        bar = "█" * (cnt // 2)
        print(f"    tick {t}: {cnt:3d}/{H} neurons active  {bar}")

    # Per-neuron: how many ticks did it fire?
    # Re-simulate to track per-neuron activity
    charges2 = np.zeros((V, H), dtype=np.float32)
    acts2 = np.zeros((V, H), dtype=np.float32)
    neuron_active_ticks = np.zeros(H, dtype=np.int32)
    for t in range(8):
        if t == 0:
            acts2 += projected
        raw2 = acts2 @ mask
        np.nan_to_num(raw2, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        charges2 += raw2
        charges2 *= retain
        acts2 = np.maximum(charges2 - net.THRESHOLD, 0.0)
        charges2 = np.clip(charges2, -1.0, 1.0)
        firing2 = (acts2 > 0).any(axis=0)
        neuron_active_ticks += firing2.astype(np.int32)

    print(f"\n    Per-neuron: hány tick-ben aktív (any input):")
    for t in range(9):
        cnt = int((neuron_active_ticks == t).sum())
        if cnt > 0:
            print(f"      {t} tick: {cnt:3d} neuron {'█' * cnt}")

    # Which functional roles are most active?
    print(f"\n    Átlagos aktív tick-ek szerep szerint:")
    for role in ['IN-face', 'OUT-face', 'BOTH', 'RELAY']:
        idx = np.where(roles == role)[0]
        if len(idx) == 0:
            continue
        avg_ticks = neuron_active_ticks[idx].mean()
        never = int((neuron_active_ticks[idx] == 0).sum())
        print(f"      {role:<10s}: avg={avg_ticks:.1f} ticks active, "
              f"never-fired={never}/{len(idx)}")

    # ── 11. OUTPUT CONTRIBUTION: which hidden neurons drive the output? ──
    print(f"\n  OUTPUT KONTRIBÚCIÓ:")
    final_charges = charges2  # (V, H) — final hidden charges
    output_logits = final_charges @ W_out  # (V, V)

    # Per-neuron contribution to correct output
    targets_argmax = np.argmax(output_logits, axis=1)
    acc = (targets_argmax == np.arange(V)).mean()
    print(f"    Forward accuracy (no training context): {acc*100:.1f}%")

    # Which neurons contribute most to output magnitude?
    contribution = np.abs(final_charges).mean(axis=0) * np.abs(W_out).sum(axis=1)
    top_contributors = np.argsort(contribution)[::-1][:10]
    print(f"\n    Top 10 output contributors:")
    print(f"    {'#':>3s} {'Neuron':>6s} {'Role':>9s} {'Contrib':>8s} {'Degree':>7s} {'ActiveT':>8s}")
    print(f"    {'─'*50}")
    for rank, idx in enumerate(top_contributors):
        print(f"    {rank+1:3d} {idx:6d} {roles[idx]:>9s} {contribution[idx]:>8.3f} "
              f"{int(total_deg[idx]):>7d} {neuron_active_ticks[idx]:>8d}")


# ══════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════

if __name__ == '__main__':
    V = 27

    for proj_type in ['hadamard', 'random']:
        for seed in [42, 123]:
            np.random.seed(seed)
            pyrandom.seed(seed)

            net = PassiveIOGraph(V, h_ratio=3, proj=proj_type)
            targets = np.random.permutation(V)

            print(f"\n\n{'▓'*75}")
            print(f"  Training PassiveIO-{proj_type} seed={seed} (10k budget)...")
            print(f"{'▓'*75}")

            train_passive(net, targets, V, max_attempts=10_000, ticks=8,
                          stale_limit=8_000, verbose=True)

            analyze_passive_topology(net, f"PassiveIO-{proj_type} seed={seed} TRAINED")

    print(f"\n{'='*75}")
    print("DONE")
