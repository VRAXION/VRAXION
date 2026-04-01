import sys, os
import numpy as np
import networkx as nx

# Paths
PRUNED_PATH = "instnct/checkpoints/overnight_int4_brain_pruned.npz"

def analyze():
    if not os.path.exists(PRUNED_PATH):
        print(f"Error: {PRUNED_PATH} not found.")
        return

    print(f"Analyzing topology: {PRUNED_PATH}")
    with np.load(PRUNED_PATH) as d:
        H = int(d['H'])
        rows = d['rows']
        cols = d['cols']
    
    # Create NetworkX graph
    G = nx.DiGraph()
    G.add_nodes_from(range(H))
    edges = list(zip(rows, cols))
    G.add_edges_from(edges)

    print(f"Nodes: {H}")
    print(f"Edges: {len(edges)}")

    # 1. Self-loops
    self_loops = list(nx.selfloop_edges(G))
    print(f"Self-loops (A -> A): {len(self_loops)}")

    # 2. Reciprocal connections (A <-> B)
    reciprocal = 0
    for u, v in G.edges():
        if u < v and G.has_edge(v, u):
            reciprocal += 1
    print(f"Reciprocal pairs (A <-> B): {reciprocal}")

    # 3. Strongly Connected Components (SCCs)
    sccs = list(nx.strongly_connected_components(G))
    sccs_larger_than_1 = [s for s in sccs if len(s) > 1]
    print(f"Strongly Connected Components (size > 1): {len(sccs_larger_than_1)}")
    if sccs_larger_than_1:
        sizes = [len(s) for s in sccs_larger_than_1]
        print(f"  SCC sizes: {sizes}")
        print(f"  Largest SCC size: {max(sizes)}")

    # 4. Total Cycles (Warning: can be huge if graph is dense)
    try:
        # We use simple_cycles which is a generator
        cycle_count = 0
        limit = 10000
        for _ in nx.simple_cycles(G):
            cycle_count += 1
            if cycle_count >= limit:
                print(f"Simple cycles: >={limit} (stopping count)")
                break
        if cycle_count < limit:
            print(f"Total Simple Cycles: {cycle_count}")
    except Exception as e:
        print(f"Could not count simple cycles: {e}")

if __name__ == "__main__":
    analyze()
