"""Quick analysis: do loops/cycles form in the self-wiring graph?
Check both random init and after training."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from model.graph import SelfWiringGraph, train

V = 32
np.random.seed(42)
random.seed(42)
net = SelfWiringGraph(V)
targets = np.arange(V)
np.random.shuffle(targets)

def analyze_loops(net, label):
    N = net.N
    V = net.V
    mask = net.mask
    # Directed graph: edge where mask != 0
    graph = csr_matrix((mask != 0).astype(np.float32))

    # 1. Bidirectional edges (2-cycles): A→B AND B→A
    bidir = 0
    for r, c in net.alive:
        if mask[c, r] != 0:
            bidir += 1
    bidir //= 2  # each pair counted twice

    # 2. Strongly connected components (any SCC size>1 has cycles)
    n_comp, labels = connected_components(graph, directed=True, connection='strong')
    scc_sizes = np.bincount(labels)
    big_sccs = scc_sizes[scc_sizes > 1]

    # 3. Self-loops (should be 0 due to fill_diagonal)
    self_loops = sum(1 for i in range(N) if mask[i, i] != 0)

    # 4. Zone analysis of biggest SCC
    zones = {
        'input': set(range(V)),
        'compute': set(range(V, 2*V)),
        'output': set(range(2*V, 3*V)),
    }

    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  Total connections: {len(net.alive)}")
    print(f"  Bidirectional pairs (2-cycles): {bidir}")
    print(f"  Self-loops: {self_loops}")
    print(f"  Strongly connected components: {n_comp}")
    print(f"  SCCs with >1 node (contain cycles): {len(big_sccs)}")
    if len(big_sccs) > 0:
        print(f"  Largest SCC size: {big_sccs.max()} / {N} neurons ({big_sccs.max()/N*100:.1f}%)")

        # Analyze the largest SCC
        biggest_label = np.argmax(scc_sizes)
        scc_members = set(np.where(labels == biggest_label)[0])
        for zone_name, zone_set in zones.items():
            overlap = len(scc_members & zone_set)
            print(f"    {zone_name}: {overlap}/{len(zone_set)} neurons in largest SCC")

        # Count cross-zone edges within the SCC
        in_to_comp = sum(1 for r, c in net.alive if r in zones['input'] and c in zones['compute']
                         and r in scc_members and c in scc_members)
        comp_to_out = sum(1 for r, c in net.alive if r in zones['compute'] and c in zones['output']
                          and r in scc_members and c in scc_members)
        out_to_comp = sum(1 for r, c in net.alive if r in zones['output'] and c in zones['compute']
                          and r in scc_members and c in scc_members)
        comp_to_in = sum(1 for r, c in net.alive if r in zones['compute'] and c in zones['input']
                         and r in scc_members and c in scc_members)
        out_to_in = sum(1 for r, c in net.alive if r in zones['output'] and c in zones['input']
                        and r in scc_members and c in scc_members)

        print(f"  Cross-zone edges in SCC:")
        print(f"    input→compute: {in_to_comp}")
        print(f"    compute→output: {comp_to_out}")
        print(f"    output→compute: {out_to_comp} (FEEDBACK!)")
        print(f"    compute→input: {comp_to_in} (FEEDBACK!)")
        print(f"    output→input: {out_to_in} (FEEDBACK!)")
    else:
        print(f"  No cycles found — purely feedforward!")

    # 5. Trace a few specific short cycles (3-cycles)
    cycles_3 = 0
    sample_cycles = []
    for r, c in net.alive[:500]:  # sample first 500 edges
        # Check A→B→C→A
        for c2 in range(N):
            if mask[c, c2] != 0 and mask[c2, r] != 0:
                cycles_3 += 1
                if len(sample_cycles) < 3:
                    def zone(n):
                        if n < V: return 'I'
                        elif n < 2*V: return 'C'
                        else: return 'O'
                    sample_cycles.append(f"{zone(r)}{r}→{zone(c)}{c}→{zone(c2)}{c2}→{zone(r)}{r}")

    print(f"  3-cycles (sampled from 500 edges): {cycles_3}")
    for cyc in sample_cycles:
        print(f"    Example: {cyc}")


# Analyze BEFORE training
analyze_loops(net, "BEFORE TRAINING (random init)")

# Train
print("\nTraining V=32 for 8000 steps...")
random.seed(42 * 1000 + 1)
train(net, targets, V, max_attempts=8000, verbose=False)
print(f"Training done. Score reached by train().")

# Analyze AFTER training
analyze_loops(net, "AFTER TRAINING (8000 steps)")
