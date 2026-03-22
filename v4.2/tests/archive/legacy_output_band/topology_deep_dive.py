"""Network Topology Deep Dive
==============================
Analyzes the actual connection structure of SelfWiringGraph networks:
- Per-neuron degree distributions (in/out, excitatory/inhibitory)
- Zone-to-zone connectivity matrix
- Hub detection
- Degree vs zone position
- Comparison: random init vs trained network
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random as pyrandom
from model.graph import SelfWiringGraph, train

# ══════════════════════════════════════════════════════
#  TOPOLOGY ANALYZER
# ══════════════════════════════════════════════════════

def analyze_topology(net, label=""):
    V, N = net.V, net.N
    mask = net.mask
    out_start = net.out_start

    # Zone boundaries
    zones = {
        "IN":  (0, V),
        "HID": (V, out_start),
        "OUT": (out_start, N),
    }

    print(f"\n{'='*75}")
    print(f"  TOPOLOGY: {label}")
    print(f"  V={V}, N={N}, out_start={out_start}")
    print(f"{'='*75}")

    # ── 1. GLOBAL STATS ──
    alive = (mask != 0)
    n_edges = int(alive.sum())
    n_excit = int((mask > 0).sum())
    n_inhib = int((mask < 0).sum())
    max_possible = N * N - N  # no self-loops
    density = n_edges / max_possible * 100

    print(f"\n  GLOBAL:")
    print(f"    Edges:  {n_edges} / {max_possible} ({density:.1f}%)")
    print(f"    Excit:  {n_excit} ({n_excit/max(n_edges,1)*100:.0f}%)")
    print(f"    Inhib:  {n_inhib} ({n_inhib/max(n_edges,1)*100:.0f}%)")

    # ── 2. PER-NEURON DEGREE ──
    in_deg = alive.sum(axis=0)    # column sums = incoming
    out_deg = alive.sum(axis=1)   # row sums = outgoing
    in_excit = (mask > 0).sum(axis=0)
    in_inhib = (mask < 0).sum(axis=0)
    out_excit = (mask > 0).sum(axis=1)
    out_inhib = (mask < 0).sum(axis=1)

    print(f"\n  PER-NEURON DEGREE (across all {N} neurons):")
    print(f"    {'':>12s}  {'min':>4s}  {'max':>4s}  {'mean':>5s}  {'median':>6s}  {'std':>5s}")
    for name, arr in [("in-degree", in_deg), ("out-degree", out_deg),
                      ("in-excit", in_excit), ("in-inhib", in_inhib),
                      ("out-excit", out_excit), ("out-inhib", out_inhib)]:
        print(f"    {name:>12s}  {int(arr.min()):4d}  {int(arr.max()):4d}  "
              f"{arr.mean():5.1f}  {np.median(arr):6.1f}  {arr.std():5.1f}")

    # ── 3. DEGREE BY ZONE ──
    print(f"\n  DEGREE BY ZONE:")
    print(f"    {'Zone':<5s} {'Neurons':>7s} │ {'In-deg':>7s} {'Out-deg':>8s} │ "
          f"{'In-exc':>7s} {'In-inh':>7s} │ {'Out-exc':>8s} {'Out-inh':>8s}")
    print(f"    {'─'*72}")
    for zname, (zstart, zend) in zones.items():
        zn = zend - zstart
        zi = in_deg[zstart:zend]
        zo = out_deg[zstart:zend]
        print(f"    {zname:<5s} {zn:>7d} │ {zi.mean():>6.1f}± {zo.mean():>6.1f}± │ "
              f"{in_excit[zstart:zend].mean():>6.1f}± {in_inhib[zstart:zend].mean():>6.1f}± │ "
              f"{out_excit[zstart:zend].mean():>7.1f}± {out_inhib[zstart:zend].mean():>7.1f}±")

    # ── 4. ZONE-TO-ZONE CONNECTIVITY MATRIX ──
    print(f"\n  ZONE-TO-ZONE EDGE COUNT:")
    hdr = "from \\ to"
    print(f"    {hdr:<10s}", end="")
    for zto in zones:
        print(f"  {zto:>6s}", end="")
    print(f"  {'TOTAL':>6s}")
    print(f"    {'─'*42}")
    for zfrom, (sf, ef) in zones.items():
        print(f"    {zfrom:<10s}", end="")
        row_total = 0
        for zto, (st, et) in zones.items():
            block = mask[sf:ef, st:et]
            cnt = int((block != 0).sum())
            row_total += cnt
            # Also show possible
            possible = (ef - sf) * (et - st)
            if zfrom == zto:
                possible -= min(ef, et) - max(sf, st)  # subtract diagonal
                if possible < 0:
                    possible = 0
            pct = cnt / max(possible, 1) * 100
            print(f"  {cnt:4d} ({pct:3.0f}%)", end="")
        print(f"  {row_total:6d}")

    # ── 5. ZONE-TO-ZONE SIGN BALANCE ──
    print(f"\n  ZONE-TO-ZONE SIGN BALANCE (excit / total):")
    hdr2 = "from \\ to"
    print(f"    {hdr2:<10s}", end="")
    for zto in zones:
        print(f"  {zto:>8s}", end="")
    print()
    print(f"    {'─'*38}")
    for zfrom, (sf, ef) in zones.items():
        print(f"    {zfrom:<10s}", end="")
        for zto, (st, et) in zones.items():
            block = mask[sf:ef, st:et]
            total = int((block != 0).sum())
            excit = int((block > 0).sum())
            if total > 0:
                print(f"  {excit:3d}/{total:<3d} ", end="")
            else:
                print(f"  {'---':>8s}", end="")
        print()

    # ── 6. HUB DETECTION ──
    total_deg = in_deg + out_deg
    top_k = 10
    top_idx = np.argsort(total_deg)[::-1][:top_k]

    print(f"\n  TOP {top_k} HUBOK (legtöbb kapcsolat):")
    print(f"    {'#':>3s} {'Neuron':>6s} {'Zone':>5s} {'Total':>6s} │ "
          f"{'In':>3s} {'Out':>4s} │ {'In+':>4s} {'In-':>4s} {'Out+':>5s} {'Out-':>5s}")
    print(f"    {'─'*58}")
    for rank, idx in enumerate(top_idx):
        # Determine zone
        if idx < V:
            zone = "IN"
        elif idx < out_start:
            zone = "HID"
        else:
            zone = "OUT"
        print(f"    {rank+1:3d} {idx:6d} {zone:>5s} {int(total_deg[idx]):6d} │ "
              f"{int(in_deg[idx]):3d} {int(out_deg[idx]):4d} │ "
              f"{int(in_excit[idx]):4d} {int(in_inhib[idx]):4d} "
              f"{int(out_excit[idx]):5d} {int(out_inhib[idx]):5d}")

    # ── 7. ISOLATED NEURONS ──
    isolated_in = int((in_deg == 0).sum())
    isolated_out = int((out_deg == 0).sum())
    fully_isolated = int(((in_deg == 0) & (out_deg == 0)).sum())
    print(f"\n  IZOLÁLT NEURONOK:")
    print(f"    Nincs bejövő:  {isolated_in} / {N}")
    print(f"    Nincs kimenő:  {isolated_out} / {N}")
    print(f"    Teljesen izolált: {fully_isolated} / {N}")

    # Zone breakdown for isolated
    for zname, (zs, ze) in zones.items():
        fi = int(((in_deg[zs:ze] == 0) & (out_deg[zs:ze] == 0)).sum())
        ni = int((in_deg[zs:ze] == 0).sum())
        no = int((out_deg[zs:ze] == 0).sum())
        zn = ze - zs
        print(f"      {zname}: no_in={ni}/{zn}  no_out={no}/{zn}  full_iso={fi}/{zn}")

    # ── 8. DEGREE HISTOGRAM ──
    print(f"\n  DEGREE ELOSZLÁS (in+out összesített):")
    bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 50]
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        cnt = int(((total_deg >= lo) & (total_deg < hi)).sum())
        bar = "█" * cnt
        if lo == hi - 1:
            print(f"    deg={lo:2d}:     {cnt:3d} {bar}")
        else:
            print(f"    deg={lo:2d}-{hi-1:2d}:  {cnt:3d} {bar}")
    cnt = int((total_deg >= bins[-1]).sum())
    if cnt > 0:
        print(f"    deg≥{bins[-1]:2d}:   {cnt:3d} {'█' * cnt}")

    # ── 9. PATH ANALYSIS: can signal reach output? ──
    print(f"\n  ELÉRHETŐSÉG (BFS az input zónából):")
    # BFS from each input neuron
    reachable_from_input = set()
    adj = {}
    for i in range(N):
        adj[i] = []
    rows, cols = np.where(mask != 0)
    for r, c in zip(rows.tolist(), cols.tolist()):
        adj[r].append(c)

    # BFS from all input neurons
    queue = list(range(V))
    visited = set(queue)
    depth_map = {i: 0 for i in range(V)}
    max_depth = 0
    while queue:
        node = queue.pop(0)
        for nb in adj[node]:
            if nb not in visited:
                visited.add(nb)
                d = depth_map[node] + 1
                depth_map[nb] = d
                max_depth = max(max_depth, d)
                queue.append(nb)

    reached_hid = len([n for n in visited if V <= n < out_start])
    reached_out = len([n for n in visited if n >= out_start])
    total_hid = out_start - V
    total_out = N - out_start

    print(f"    Input-ból elérhető hidden: {reached_hid}/{total_hid}")
    print(f"    Input-ból elérhető output: {reached_out}/{total_out}")
    print(f"    Max távolság (hop): {max_depth}")

    # Depth distribution for output neurons
    out_depths = []
    for n in range(out_start, N):
        if n in depth_map:
            out_depths.append(depth_map[n])
    if out_depths:
        out_depths = np.array(out_depths)
        print(f"    Output neuronok távolsága input-tól:")
        print(f"      min={out_depths.min()} max={out_depths.max()} "
              f"mean={out_depths.mean():.1f} median={np.median(out_depths):.0f}")
        for d in range(1, max_depth + 1):
            cnt = int((out_depths == d).sum())
            if cnt > 0:
                print(f"      {d} hop: {cnt} output neuron")
    else:
        print(f"    FIGYELEM: egyetlen output neuron sem érhető el!")

    return {
        'n_edges': n_edges, 'density': density,
        'in_deg': in_deg, 'out_deg': out_deg,
        'reached_out': reached_out, 'total_out': total_out,
    }


# ══════════════════════════════════════════════════════
#  MAIN: analyze random init + trained networks
# ══════════════════════════════════════════════════════

if __name__ == '__main__':
    V = 27
    N = V * 3  # 81

    # ── A) RANDOM INIT ──
    np.random.seed(42)
    pyrandom.seed(42)
    net_init = SelfWiringGraph(V)
    stats_init = analyze_topology(net_init, "RANDOM INIT (seed=42)")

    # ── B) TRAINED on permutation task ──
    print("\n\n" + "▓" * 75)
    print("  Training network on permutation task (V=27, 8k budget)...")
    print("▓" * 75)

    np.random.seed(42)
    pyrandom.seed(42)
    net_trained = SelfWiringGraph(V)
    targets = np.random.permutation(V)
    train(net_trained, targets, V, max_attempts=8000, ticks=8,
          stale_limit=6000, verbose=True)
    stats_trained = analyze_topology(net_trained, "TRAINED (permutation, 8k steps)")

    # ── C) TRAINED on permutation — different seed ──
    np.random.seed(123)
    pyrandom.seed(123)
    net_t2 = SelfWiringGraph(V)
    targets2 = np.random.permutation(V)
    print("\n\n" + "▓" * 75)
    print("  Training network #2 (seed=123, 8k budget)...")
    print("▓" * 75)
    train(net_t2, targets2, V, max_attempts=8000, ticks=8,
          stale_limit=6000, verbose=True)
    stats_t2 = analyze_topology(net_t2, "TRAINED #2 (seed=123)")

    # ── D) Compare init vs trained ──
    print(f"\n\n{'='*75}")
    print(f"  ÖSSZEHASONLÍTÁS: INIT vs TRAINED")
    print(f"{'='*75}")
    print(f"    {'Metric':<30s} {'Init':>10s} {'Trained#1':>10s} {'Trained#2':>10s}")
    print(f"    {'─'*62}")
    for key, label in [('n_edges', 'Edges'),
                       ('density', 'Density (%)'),
                       ('reached_out', 'Output reached'),
                       ('total_out', 'Output total')]:
        v1 = stats_init[key]
        v2 = stats_trained[key]
        v3 = stats_t2[key]
        if isinstance(v1, float):
            print(f"    {label:<30s} {v1:>10.1f} {v2:>10.1f} {v3:>10.1f}")
        else:
            print(f"    {label:<30s} {v1:>10d} {v2:>10d} {v3:>10d}")

    # Degree comparison
    for key, label in [('in_deg', 'In-degree mean'),
                       ('out_deg', 'Out-degree mean')]:
        v1 = stats_init[key].mean()
        v2 = stats_trained[key].mean()
        v3 = stats_t2[key].mean()
        print(f"    {label:<30s} {v1:>10.1f} {v2:>10.1f} {v3:>10.1f}")
    for key, label in [('in_deg', 'In-degree max'),
                       ('out_deg', 'Out-degree max')]:
        v1 = int(stats_init[key].max())
        v2 = int(stats_trained[key].max())
        v3 = int(stats_t2[key].max())
        print(f"    {label:<30s} {v1:>10d} {v2:>10d} {v3:>10d}")

    print(f"\n{'='*75}")
    print("DONE")
    print(f"{'='*75}")
