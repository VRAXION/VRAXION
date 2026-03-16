"""
Fast Density Scaling Sweep — power law fit for density vs vocab size.
Reduced budgets for quick turnaround. 3 seeds instead of 5.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from model.graph import SelfWiringGraph, train

VOCAB_SIZES = [8, 16, 32, 64, 128, 256, 384, 512]
SEEDS = [42, 77, 123]

def budget_for(V):
    if V <= 32:   return 4000
    if V <= 64:   return 5000
    if V <= 128:  return 6000
    if V <= 256:  return 8000
    return 10000

def prune_fast(net, targets, V, ticks=8, tolerance=0.005):
    """Fast prune: batch removal with binary search on batch size."""
    def evaluate():
        logits = net.forward_batch(ticks)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        acc = (np.argmax(probs, axis=1)[:V] == targets[:V]).mean()
        tp = probs[np.arange(V), targets[:V]].mean()
        return 0.5 * acc + 0.5 * tp

    baseline = evaluate()
    threshold = baseline - tolerance
    removed = 0
    batch = max(1, len(net.alive) // 20)  # 5% at a time

    while net.alive and batch >= 1:
        indices = random.sample(range(len(net.alive)), min(batch, len(net.alive)))
        saved = []
        for idx in sorted(indices, reverse=True):
            r, c = net.alive[idx]
            saved.append((r, c, net.mask[r, c]))
            net.mask[r, c] = 0
            net.alive_set.discard((r, c))
        net.alive = list(net.alive_set)

        if evaluate() >= threshold:
            removed += len(saved)
        else:
            for r, c, val in saved:
                net.mask[r, c] = val
                net.alive_set.add((r, c))
            net.alive = list(net.alive_set)
            if batch == 1:
                break
            batch = max(1, batch // 2)

    return len(net.alive), removed

def run_one(V, seed):
    t0 = time.time()
    np.random.seed(seed); random.seed(seed)
    net = SelfWiringGraph(V)
    N = net.N
    targets = np.random.permutation(V)
    max_N2 = N * (N - 1)

    score = train(net, targets, V, max_attempts=budget_for(V), ticks=8, verbose=False)
    edges_t = net.count_connections()
    density_t = edges_t / max_N2

    edges_p, pruned = prune_fast(net, targets, V)
    density_p = edges_p / max_N2

    return {
        'V': V, 'N': N, 'seed': seed, 'score': score,
        'edges_t': edges_t, 'density_t': density_t,
        'edges_p': edges_p, 'density_p': density_p,
        'compression': pruned / edges_t if edges_t else 0,
        'elapsed': time.time() - t0,
    }

def fit_power_law(Vs, vals):
    valid = [(v, d) for v, d in zip(Vs, vals) if d > 0]
    if len(valid) < 3: return None, None, None
    lv = np.log([v for v, _ in valid])
    ld = np.log([d for _, d in valid])
    c = np.polyfit(lv, ld, 1)
    b, a = c[0], np.exp(c[1])
    pred = a * np.array([v for v, _ in valid]) ** b
    act = np.array([d for _, d in valid])
    ss_res = np.sum((act - pred)**2)
    ss_tot = np.sum((act - act.mean())**2)
    return a, b, 1 - ss_res/ss_tot if ss_tot else 0

def main():
    ncpu = multiprocessing.cpu_count()
    jobs = [(V, s) for V in VOCAB_SIZES for s in SEEDS]
    print(f"FAST DENSITY SCALING: {len(jobs)} jobs on {ncpu} cores")
    print("=" * 95)

    results = []
    done = 0
    with ProcessPoolExecutor(max_workers=ncpu) as pool:
        futs = {pool.submit(run_one, V, s): (V, s) for V, s in jobs}
        for f in as_completed(futs):
            r = f.result(); results.append(r); done += 1
            print(f"  [{done:2d}/{len(jobs)}] V={r['V']:4d} s={r['seed']:3d} "
                  f"score={r['score']*100:5.1f}% "
                  f"edges:{r['edges_t']:5d}→{r['edges_p']:5d} "
                  f"dens:{r['density_t']*100:5.2f}%→{r['density_p']*100:5.2f}% "
                  f"comp={r['compression']*100:4.1f}% ({r['elapsed']:.0f}s)", flush=True)

    # Aggregate
    print("\n" + "=" * 95)
    print(f"{'V':>5} {'N':>5} {'Score%':>8} {'EdgesT':>7} {'EdgesP':>7} "
          f"{'DensT%':>8} {'DensP%':>8} {'Comp%':>7} {'Time':>5}")
    print("-" * 95)

    aV, aDt, aDp, aEp = [], [], [], []
    for V in VOCAB_SIZES:
        rs = [r for r in results if r['V'] == V]
        if not rs: continue
        sc = np.mean([r['score'] for r in rs])
        et = np.mean([r['edges_t'] for r in rs])
        ep = np.mean([r['edges_p'] for r in rs])
        dt = np.mean([r['density_t'] for r in rs])
        dp = np.mean([r['density_p'] for r in rs])
        co = np.mean([r['compression'] for r in rs])
        tm = np.mean([r['elapsed'] for r in rs])
        print(f"{V:5d} {rs[0]['N']:5d} {sc*100:7.1f}% {et:7.0f} {ep:7.0f} "
              f"{dt*100:7.3f}% {dp*100:7.3f}% {co*100:6.1f}% {tm:4.0f}s")
        aV.append(V); aDt.append(dt); aDp.append(dp); aEp.append(ep)

    # Power law
    print("\n" + "=" * 95)
    print("POWER LAW: density = a × V^b")
    at, bt, r2t = fit_power_law(aV, aDt)
    ap, bp, r2p = fit_power_law(aV, aDp)
    ae, be, r2e = fit_power_law(aV, aEp)

    if at: print(f"  Trained density:  {at:.6f} × V^{bt:.4f}  R²={r2t:.4f}")
    if ap: print(f"  Pruned density:   {ap:.6f} × V^{bp:.4f}  R²={r2p:.4f}")
    if ae: print(f"  Pruned edges:     {ae:.4f} × V^{be:.4f}  R²={r2e:.4f}")

    # Extrapolation
    print("\n" + "=" * 95)
    print("EXTRAPOLATION")
    print(f"{'V':>8} {'N':>8} {'DensP%':>10} {'EdgesP':>12} {'Bits':>14} "
          f"{'MB_wt':>8} {'MB_st':>8} {'MB_tot':>8}")
    print("-" * 95)
    for V in [512, 1024, 2048, 4096, 8192, 16384, 32000, 50000]:
        N = V * 3
        if ap and ae:
            dp = ap * V**bp
            ep = ae * V**be
            bits = ep * 1.58
            mb_w = bits / 8 / 1024**2
            mb_s = N * 2 * 4 / 1024**2
            print(f"{V:8d} {N:8d} {dp*100:9.5f}% {ep:11.0f} "
                  f"{bits:13.0f} {mb_w:7.2f} {mb_s:7.2f} {mb_w+mb_s:7.2f}")

    if ae:
        ep32k = ae * 32000**be
        mb32k = ep32k * 1.58 / 8 / 1024**2 + 32000*3*2*4/1024**2
        print(f"\n  === V=32K English SWG: {ep32k:,.0f} edges = {mb32k:.1f} MB total ===")
        print(f"  === vs GPT-2 Small: ~323 MB → ratio: {mb32k/323:.2f}× ===")
        print(f"  === Edge scaling: V^{be:.2f} {'(sub-quadratic!)' if be < 2 else '(quadratic)'} ===")

    print("=" * 95)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
