import sys, os
import numpy as np
import networkx as nx

def analyze(path):
    if not os.path.exists(path):
        return None
    with np.load(path) as d:
        rows = d['rows']
        cols = d['cols']
        H = int(d['H'])
    G = nx.DiGraph()
    G.add_nodes_from(range(H))
    G.add_edges_from(zip(rows, cols))
    sccs = [len(c) for c in nx.strongly_connected_components(G) if len(c) > 1]
    return {"edges": len(rows), "sccs": sccs}

p1 = "instnct/checkpoints/overnight_int4_brain_pruned.npz"
p2 = "instnct/checkpoints/post_prune_brain.npz"

print(f"2100 model: {analyze(p1)}")
print(f"766 model:  {analyze(p2)}")
