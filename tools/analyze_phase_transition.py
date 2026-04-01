import numpy as np
import networkx as nx

def test_scc_transition(N=1024):
    # Teszteljük az élsűrűséget 0.1-től 5.0-ig
    densities = np.linspace(0.1, 5.0, 20) 
    
    print(f"=" * 50)
    print(f"  SCC PHASE TRANSITION (N={N})")
    print(f"=" * 50)
    
    for d in densities:
        num_edges = int(N * d)
        # Random directed graph (A -> B)
        rows = np.random.randint(0, N, num_edges)
        cols = np.random.randint(0, N, num_edges)
        
        G = nx.DiGraph()
        G.add_nodes_from(range(N))
        G.add_edges_from(zip(rows, cols))
        
        sccs = list(nx.strongly_connected_components(G))
        max_scc_size = max(len(c) for c in sccs) if sccs else 0
        pct = max_scc_size / N
        
        # Milyen 'mély' a hálózat? (Hány ugrás)
        # Random éleknél a tipikus átmérő (log N)
        
        print(f"  Density: {d:4.2f} e/n | Edges: {num_edges:5d} | Giant SCC: {pct*100:5.1f}%")
        if pct > 0.9: 
            print(f"  >>> GIANT SCC REACHED (90%+) at {d:.2f} e/n")
            break

if __name__ == "__main__":
    test_scc_transition()
