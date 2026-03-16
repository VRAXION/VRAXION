"""
Pruning + Chain Analytics Test
==============================
1. Train a network to convergence
2. Analyze chain statistics (before prune)
3. Prune iteratively (energy vs random strategy A/B)
4. Analyze chain statistics (after prune)
5. Compare: how much can we compress without accuracy loss?
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
from model.graph_v43 import SelfWiringGraph, train

CONFIGS = [
    # (name,  V, density)
    ("V16",   16, 0.06),
    ("V64",   64, 0.06),
    ("V64_dense", 64, 0.15),
]

SEEDS = [42, 77, 123]
TOLERANCES = [0.005, 0.01, 0.02]  # how much accuracy drop is acceptable


def run_one(name, V, density, seed, tolerance):
    np.random.seed(seed)
    random.seed(seed)

    net = SelfWiringGraph(V, density=density)
    targets = np.random.permutation(V)

    # Train
    score = train(net, targets, V, max_attempts=6000, verbose=False)

    # Chain stats before pruning
    stats_before = net.chain_stats()

    # Prune with energy strategy
    net_energy = SelfWiringGraph(V, density=density)
    net_energy.mask[:net.N, :net.N] = net.mask[:net.N, :net.N].copy()
    net_energy.N = net.N
    net_energy.resync_alive()
    net_energy.loss_pct = net.loss_pct
    # Run forward to populate energy
    net_energy.forward_batch()
    prune_energy = net_energy.prune(targets, tolerance=tolerance,
                                     strategy='energy', verbose=False)

    # Chain stats after energy prune
    stats_after_energy = net_energy.chain_stats()

    # Prune with random strategy (fresh copy)
    net_random = SelfWiringGraph(V, density=density)
    net_random.mask[:net.N, :net.N] = net.mask[:net.N, :net.N].copy()
    net_random.N = net.N
    net_random.resync_alive()
    net_random.loss_pct = net.loss_pct
    net_random.forward_batch()
    prune_random = net_random.prune(targets, tolerance=tolerance,
                                     strategy='random', verbose=False)

    stats_after_random = net_random.chain_stats()

    return {
        'name': name, 'V': V, 'seed': seed, 'tolerance': tolerance,
        'train_score': score,
        'before': stats_before,
        'energy_prune': prune_energy,
        'random_prune': prune_random,
        'after_energy': stats_after_energy,
        'after_random': stats_after_random,
    }


def main():
    print("=" * 90)
    print("PRUNE + CHAIN ANALYTICS TEST")
    print("=" * 90)

    results = []
    for name, V, density in CONFIGS:
        for seed in SEEDS:
            for tol in TOLERANCES:
                print(f"\n--- {name} seed={seed} tol={tol} ---")
                r = run_one(name, V, density, seed, tol)
                results.append(r)

                b = r['before']
                ae = r['after_energy']
                ar = r['after_random']
                pe = r['energy_prune']
                pr = r['random_prune']

                print(f"  Train score: {r['train_score']*100:.1f}%")
                print(f"  BEFORE prune:")
                print(f"    edges={b['total_edges']}  "
                      f"avg_path={b['avg_path']}  "
                      f"diameter={b['diameter']}  "
                      f"clustering={b['clustering']}  "
                      f"dead={b['dead_neurons']}  "
                      f"reachable={b['reachable']}")
                print(f"  ENERGY prune (tol={tol}):")
                print(f"    {pe['init_edges']} → {pe['final_edges']} edges "
                      f"({pe['compression']*100:.1f}% compression)  "
                      f"score: {pe['baseline_score']*100:.1f}% → "
                      f"{pe['final_score']*100:.1f}%")
                print(f"    avg_path={ae['avg_path']}  "
                      f"diameter={ae['diameter']}  "
                      f"clustering={ae['clustering']}  "
                      f"dead={ae['dead_neurons']}")
                print(f"  RANDOM prune (tol={tol}):")
                print(f"    {pr['init_edges']} → {pr['final_edges']} edges "
                      f"({pr['compression']*100:.1f}% compression)  "
                      f"score: {pr['baseline_score']*100:.1f}% → "
                      f"{pr['final_score']*100:.1f}%")
                print(f"    avg_path={ar['avg_path']}  "
                      f"diameter={ar['diameter']}  "
                      f"clustering={ar['clustering']}  "
                      f"dead={ar['dead_neurons']}")

    # Summary table
    print("\n" + "=" * 90)
    print("COMPRESSION SUMMARY")
    print(f"{'Config':12s} {'Tol':>5s} | {'Energy':>10s} {'Rand':>10s} | "
          f"{'E_score':>8s} {'R_score':>8s} | "
          f"{'E_path':>6s} {'R_path':>6s} | {'E_diam':>6s} {'R_diam':>6s}")
    print("-" * 90)

    for name, V, density in CONFIGS:
        for tol in TOLERANCES:
            e_comp = []
            r_comp = []
            e_score = []
            r_score = []
            e_path = []
            r_path = []
            e_diam = []
            r_diam = []
            for r in results:
                if r['name'] == name and r['tolerance'] == tol:
                    e_comp.append(r['energy_prune']['compression'])
                    r_comp.append(r['random_prune']['compression'])
                    e_score.append(r['energy_prune']['final_score'])
                    r_score.append(r['random_prune']['final_score'])
                    e_path.append(r['after_energy']['avg_path'])
                    r_path.append(r['after_random']['avg_path'])
                    e_diam.append(r['after_energy']['diameter'])
                    r_diam.append(r['after_random']['diameter'])
            if e_comp:
                print(f"{name:12s} {tol:5.3f} | "
                      f"{np.mean(e_comp)*100:9.1f}% {np.mean(r_comp)*100:9.1f}% | "
                      f"{np.mean(e_score)*100:7.1f}% {np.mean(r_score)*100:7.1f}% | "
                      f"{np.mean(e_path):6.2f} {np.mean(r_path):6.2f} | "
                      f"{np.mean(e_diam):6.1f} {np.mean(r_diam):6.1f}")

    print("=" * 90)

    # Bottleneck analysis on first V64 result
    v64_results = [r for r in results if r['name'] == 'V64' and r['tolerance'] == 0.01]
    if v64_results:
        r = v64_results[0]
        print("\nBOTTLENECK ANALYSIS (V64, tol=0.01, after energy prune):")
        bns = r['after_energy'].get('bottlenecks', [])
        if bns:
            print(f"  Top bottleneck edges (highest betweenness):")
            for (edge, count) in bns[:5]:
                print(f"    edge {edge[0]:3d} → {edge[1]:3d}  "
                      f"paths through: {count}")
        else:
            print("  No bottlenecks found")


if __name__ == '__main__':
    main()
