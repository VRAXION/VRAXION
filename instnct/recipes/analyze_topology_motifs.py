"""
Topology motif analysis
========================
Compare 6 trained variants: do they share the same MOTIFS
even though they have 0 shared edges?

Analyzes:
  - In/out degree distribution
  - Loop count (2-cycles, 3-cycles)
  - Strongly connected components
  - Path length distribution
  - Hub neurons (high degree)
  - Chain length (longest directed path)
"""
import sys, numpy as np
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'model'))
from graph import SelfWiringGraph

CKPT_DIR = ROOT / 'recipes' / 'checkpoints'
VARIANTS = ['variant_seed42', 'variant_seed777', 'variant_seed123',
            'variant_seed999', 'variant_seed314', 'variant_seed2024']

def analyze_topology(mask):
    """Extract topological motifs from a boolean mask."""
    H = mask.shape[0]
    rows, cols = np.where(mask)
    edges = set(zip(rows.tolist(), cols.tolist()))
    n_edges = len(edges)

    # Degree distributions
    out_degree = np.sum(mask, axis=1)  # row sums
    in_degree = np.sum(mask, axis=0)   # col sums
    total_degree = out_degree + in_degree

    # Active neurons
    active = np.where(total_degree > 0)[0]
    n_active = len(active)

    # Hub neurons (degree >= 3)
    hubs = np.where(total_degree >= 3)[0]

    # 2-cycles (bidirectional edges): A→B and B→A
    n_2cycles = 0
    for r, c in edges:
        if (c, r) in edges:
            n_2cycles += 1
    n_2cycles //= 2  # each counted twice

    # 3-cycles: A→B→C→A
    n_3cycles = 0
    mask_f = mask.astype(np.float32)
    # M^3 diagonal = number of 3-cycles through each node
    m2 = mask_f @ mask_f
    m3 = m2 @ mask_f
    n_3cycles = int(np.trace(m3)) // 3  # each cycle counted 3 times

    # Self-loops (should be 0)
    self_loops = int(np.trace(mask.astype(int)))

    # Strongly connected components (simple DFS)
    def scc_sizes(adj):
        n = adj.shape[0]
        visited = [False] * n
        order = []

        def dfs1(v):
            stack = [v]
            while stack:
                u = stack[-1]
                visited[u] = True
                found = False
                for w in range(n):
                    if adj[u, w] and not visited[w]:
                        stack.append(w)
                        found = True
                        break
                if not found:
                    order.append(stack.pop())

        for i in active:
            if not visited[i]:
                dfs1(i)

        # Transpose
        adj_t = adj.T
        visited2 = [False] * n
        components = []

        def dfs2(v):
            stack = [v]
            comp = []
            while stack:
                u = stack.pop()
                if visited2[u]:
                    continue
                visited2[u] = True
                comp.append(u)
                for w in range(n):
                    if adj_t[u, w] and not visited2[w]:
                        stack.append(w)
            return comp

        for v in reversed(order):
            if not visited2[v]:
                comp = dfs2(v)
                if len(comp) > 0:
                    components.append(len(comp))

        return sorted(components, reverse=True)

    sccs = scc_sizes(mask)

    # Chain: longest path via BFS from each active node
    max_path = 0
    for start in active[:50]:  # sample to keep fast
        visited = {start}
        queue = [(start, 0)]
        while queue:
            node, depth = queue.pop(0)
            if depth > max_path:
                max_path = depth
            for j in range(H):
                if mask[node, j] and j not in visited:
                    visited.add(j)
                    queue.append((j, depth + 1))

    return {
        'edges': n_edges,
        'active_neurons': n_active,
        'out_degree': out_degree[active],
        'in_degree': in_degree[active],
        'total_degree': total_degree[active],
        'hubs': len(hubs),
        'n_2cycles': n_2cycles,
        'n_3cycles': n_3cycles,
        'scc_sizes': sccs,
        'max_path': max_path,
        'degree_mean': float(total_degree[active].mean()) if n_active else 0,
        'degree_std': float(total_degree[active].std()) if n_active else 0,
        'degree_max': int(total_degree[active].max()) if n_active else 0,
        'out_mean': float(out_degree[active].mean()) if n_active else 0,
        'in_mean': float(in_degree[active].mean()) if n_active else 0,
    }

if __name__ == '__main__':
    print('Loading variants...\n')
    all_stats = {}

    for name in VARIANTS:
        path = CKPT_DIR / f'{name}.npz'
        if not path.exists():
            print(f'  SKIP {name} (not found)')
            continue
        net = SelfWiringGraph.load(str(path))
        stats = analyze_topology(net.mask)
        all_stats[name] = stats
        seed = name.split('seed')[1]
        print(f'  seed={seed}: edges={stats["edges"]} active={stats["active_neurons"]} '
              f'hubs={stats["hubs"]} 2cyc={stats["n_2cycles"]} 3cyc={stats["n_3cycles"]} '
              f'max_path={stats["max_path"]} deg={stats["degree_mean"]:.2f}±{stats["degree_std"]:.2f}')

    if len(all_stats) < 2:
        print('\nNot enough variants to compare.')
        sys.exit(0)

    # ── Cross-variant comparison ─────────────────────────────────────────────
    sep = '=' * 60
    print(f'\n{sep}')
    print('  MOTIF COMPARISON ACROSS VARIANTS')
    print(sep)

    # Collect stats
    names = list(all_stats.keys())
    metrics = {
        'edges': [all_stats[n]['edges'] for n in names],
        'active_neurons': [all_stats[n]['active_neurons'] for n in names],
        'hubs': [all_stats[n]['hubs'] for n in names],
        '2-cycles': [all_stats[n]['n_2cycles'] for n in names],
        '3-cycles': [all_stats[n]['n_3cycles'] for n in names],
        'max_path': [all_stats[n]['max_path'] for n in names],
        'degree_mean': [all_stats[n]['degree_mean'] for n in names],
        'degree_max': [all_stats[n]['degree_max'] for n in names],
        'out_degree_mean': [all_stats[n]['out_mean'] for n in names],
        'in_degree_mean': [all_stats[n]['in_mean'] for n in names],
    }

    print(f'\n  {"Metric":<20} {"Mean":>8} {"Std":>8} {"Min":>6} {"Max":>6}  {"CV%":>6}')
    print(f'  {"-"*58}')
    for name, vals in metrics.items():
        arr = np.array(vals, dtype=float)
        mean = arr.mean()
        std = arr.std()
        cv = (std / mean * 100) if mean > 0 else 0
        print(f'  {name:<20} {mean:>8.2f} {std:>8.2f} {arr.min():>6.1f} {arr.max():>6.1f}  {cv:>5.1f}%')

    # SCC analysis
    print(f'\n  Strongly Connected Components:')
    for n in names:
        seed = n.split('seed')[1]
        sccs = all_stats[n]['scc_sizes']
        print(f'    seed={seed}: {sccs[:10]}{"..." if len(sccs)>10 else ""}')

    # Degree distribution similarity (histogram comparison)
    print(f'\n  Degree histogram comparison (bins=[1,2,3,4,5+]):')
    bins = [1, 2, 3, 4, 5]
    for n in names:
        seed = n.split('seed')[1]
        deg = all_stats[n]['total_degree']
        hist = []
        for b in bins[:-1]:
            hist.append(int(np.sum(deg == b)))
        hist.append(int(np.sum(deg >= bins[-1])))
        total = sum(hist)
        pcts = [f'{h/total*100:.0f}%' if total else '0%' for h in hist]
        print(f'    seed={seed}: {hist}  ({", ".join(pcts)})')

    # Interpretation
    print(f'\n{sep}')
    print('  INTERPRETATION')
    print(sep)

    # Check if motifs are consistent despite different edges
    cv_vals = {}
    for name, vals in metrics.items():
        arr = np.array(vals, dtype=float)
        mean = arr.mean()
        cv = (arr.std() / mean * 100) if mean > 0 else 999
        cv_vals[name] = cv

    consistent = [k for k, v in cv_vals.items() if v < 30]
    variable = [k for k, v in cv_vals.items() if v >= 30]

    if consistent:
        print(f'  Consistent motifs (CV<30%): {", ".join(consistent)}')
    if variable:
        print(f'  Variable motifs (CV>=30%): {", ".join(variable)}')

    if len(consistent) > len(variable):
        print(f'\n  -> MOTIF CONVERGENCE: different edges, SAME structural patterns')
        print(f'     The "knowledge" is in the motif type, not the edge positions')
    else:
        print(f'\n  -> NO MOTIF CONVERGENCE: both edges AND structure differ')
        print(f'     The 21.64% plateau may be a trivial solution, not learned structure')
    print(sep)
