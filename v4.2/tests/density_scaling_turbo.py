"""
TURBO Density Scaling — minimal budgets, just enough to see the trend.
Focus: density after training (no pruning), power law fit only.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from model.graph import SelfWiringGraph, train

# Init density is 4% (DENSITY=4 in class). After training, some edges get
# added/removed. We measure the CONVERGED density.
# Key insight: the graph starts at 4% and evolves. We need to track where
# it ends up.

VOCAB_SIZES = [8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
SEEDS = [42, 77, 123]

def budget_for(V):
    """Aggressive budget — just enough to see density trend."""
    if V <= 16:   return 2000
    if V <= 48:   return 3000
    if V <= 96:   return 4000
    if V <= 128:  return 5000
    return 6000

def run_one(V, seed):
    t0 = time.time()
    np.random.seed(seed); random.seed(seed)
    net = SelfWiringGraph(V)
    N = net.N
    targets = np.random.permutation(V)
    max_e = N * (N - 1)

    init_edges = net.count_connections()
    init_density = init_edges / max_e

    score = train(net, targets, V, max_attempts=budget_for(V), ticks=8,
                  verbose=False)

    final_edges = net.count_connections()
    final_density = final_edges / max_e

    # Quick prune: just remove random 5% batches until score drops
    def evaluate():
        logits = net.forward_batch(8)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        acc = (np.argmax(probs, axis=1)[:V] == targets[:V]).mean()
        tp = probs[np.arange(V), targets[:V]].mean()
        return 0.5 * acc + 0.5 * tp

    baseline = evaluate()
    threshold = baseline - 0.005
    pruned_edges = final_edges

    batch = max(1, final_edges // 20)
    while len(net.alive) > 0 and batch >= 1:
        sample_n = min(batch, len(net.alive))
        idxs = random.sample(range(len(net.alive)), sample_n)
        saved = []
        for i in sorted(idxs, reverse=True):
            r, c = net.alive[i]
            saved.append((r, c, net.mask[r, c]))
            net.mask[r, c] = 0
            net.alive_set.discard((r, c))
        net.alive = list(net.alive_set)

        if evaluate() >= threshold:
            pruned_edges = len(net.alive)
        else:
            for r, c, v in saved:
                net.mask[r, c] = v
                net.alive_set.add((r, c))
            net.alive = list(net.alive_set)
            if batch <= 1:
                break
            batch = max(1, batch // 2)

    pruned_density = pruned_edges / max_e
    elapsed = time.time() - t0

    return {
        'V': V, 'N': N, 'seed': seed, 'score': score,
        'init_e': init_edges, 'init_d': init_density,
        'train_e': final_edges, 'train_d': final_density,
        'prune_e': pruned_edges, 'prune_d': pruned_density,
        'elapsed': elapsed,
    }

def fit_pl(Vs, vals):
    ok = [(v, d) for v, d in zip(Vs, vals) if d > 0]
    if len(ok) < 3: return None, None, None
    lv = np.log([v for v,_ in ok]); ld = np.log([d for _,d in ok])
    c = np.polyfit(lv, ld, 1)
    b, a = c[0], np.exp(c[1])
    p = a * np.array([v for v,_ in ok])**b
    ac = np.array([d for _,d in ok])
    r2 = 1 - np.sum((ac-p)**2)/np.sum((ac-ac.mean())**2)
    return a, b, r2

def main():
    ncpu = multiprocessing.cpu_count()
    jobs = [(V, s) for V in VOCAB_SIZES for s in SEEDS]
    print(f"TURBO DENSITY SCALING: {len(jobs)} jobs / {ncpu} cores")
    print("=" * 100)

    results = []; done = 0
    with ProcessPoolExecutor(max_workers=ncpu) as pool:
        futs = {pool.submit(run_one, V, s): (V,s) for V,s in jobs}
        for f in as_completed(futs):
            r = f.result(); results.append(r); done += 1
            print(f"  [{done:2d}/{len(jobs)}] V={r['V']:4d} "
                  f"score={r['score']*100:5.1f}% "
                  f"init={r['init_d']*100:5.2f}% "
                  f"train={r['train_d']*100:5.2f}% "
                  f"prune={r['prune_d']*100:5.2f}% "
                  f"edges:{r['init_e']}→{r['train_e']}→{r['prune_e']} "
                  f"({r['elapsed']:.0f}s)", flush=True)

    print("\n" + "=" * 100)
    print("AGGREGATED (mean of 3 seeds)")
    print(f"{'V':>5} {'N':>5} {'Score':>7} {'InitD%':>8} {'TrainD%':>8} "
          f"{'PruneD%':>8} {'InitE':>7} {'TrainE':>7} {'PruneE':>7}")
    print("-" * 100)

    aV, aTd, aPd, aPe = [], [], [], []
    for V in VOCAB_SIZES:
        rs = [r for r in results if r['V'] == V]
        if not rs: continue
        print(f"{V:5d} {rs[0]['N']:5d} "
              f"{np.mean([r['score'] for r in rs])*100:6.1f}% "
              f"{np.mean([r['init_d'] for r in rs])*100:7.3f}% "
              f"{np.mean([r['train_d'] for r in rs])*100:7.3f}% "
              f"{np.mean([r['prune_d'] for r in rs])*100:7.3f}% "
              f"{np.mean([r['init_e'] for r in rs]):7.0f} "
              f"{np.mean([r['train_e'] for r in rs]):7.0f} "
              f"{np.mean([r['prune_e'] for r in rs]):7.0f}")
        aV.append(V)
        aTd.append(np.mean([r['train_d'] for r in rs]))
        aPd.append(np.mean([r['prune_d'] for r in rs]))
        aPe.append(np.mean([r['prune_e'] for r in rs]))

    print("\n" + "=" * 100)
    print("POWER LAW FITS")
    at, bt, r2t = fit_pl(aV, aTd)
    ap, bp, r2p = fit_pl(aV, aPd)
    ae, be, r2e = fit_pl(aV, aPe)
    if at: print(f"  Trained density = {at:.6f} × V^({bt:.4f})  R²={r2t:.4f}")
    if ap: print(f"  Pruned  density = {ap:.6f} × V^({bp:.4f})  R²={r2p:.4f}")
    if ae: print(f"  Pruned  edges   = {ae:.4f} × V^({be:.4f})  R²={r2e:.4f}")

    print("\n" + "=" * 100)
    print("EXTRAPOLATION TO ENGLISH-SCALE")
    print(f"{'V':>8} {'N':>8} {'PruneDens%':>11} {'PruneEdges':>12} "
          f"{'MB_wt':>8} {'MB_state':>8} {'MB_total':>8}")
    print("-" * 100)
    for V in [256, 512, 1024, 2048, 4096, 8192, 16384, 32000, 50000, 100000]:
        N = V * 3
        if ap and ae:
            dp = ap * V**bp
            ep = ae * V**be
            mb_w = ep * 1.58 / 8 / 1024**2
            mb_s = N * 2 * 4 / 1024**2
            print(f"{V:8d} {N:8d} {dp*100:10.5f}% {ep:11.0f} "
                  f"{mb_w:7.2f} {mb_s:7.2f} {mb_w + mb_s:7.2f}")

    if ae:
        e32 = ae * 32000**be
        mb = e32 * 1.58/8/1024**2 + 32000*3*2*4/1024**2
        print(f"\n  V=32K ENGLISH MODEL: ~{e32:,.0f} edges = {mb:.1f} MB")
        print(f"  GPT-2 Small: ~323 MB  →  SWG/GPT2 = {mb/323:.2f}×")
        print(f"  Edge scaling: V^{be:.2f}")
        if be < 1.5:
            print(f"  VERDICT: SUB-LINEAR edge growth! Extremely efficient.")
        elif be < 2.0:
            print(f"  VERDICT: Sub-quadratic. Good scaling.")
        else:
            print(f"  VERDICT: Quadratic or worse. Density constant.")
    print("=" * 100)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
