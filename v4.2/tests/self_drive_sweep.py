"""Self-drive sweep: is 0.015 needed or can we remove it?"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np, random, multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from model.graph import SelfWiringGraph, train

SEEDS = [42, 77, 123, 0, 1]
VALUES = [0, 0.005, 0.015, 0.03, 0.05]

def run_one(sd, seed):
    SelfWiringGraph.SELF_DRIVE = sd
    np.random.seed(seed); random.seed(seed)
    net = SelfWiringGraph(64)
    perm = np.random.permutation(64)
    best = train(net, perm, 64, max_attempts=16000, ticks=8, verbose=False)
    SelfWiringGraph.SELF_DRIVE = 0.015
    return sd, seed, best

def main():
    jobs = [(sd, s) for sd in VALUES for s in SEEDS]
    print(f'SELF_DRIVE SWEEP: {len(jobs)} jobs parallel', flush=True)
    results = {}
    with ProcessPoolExecutor(max_workers=22) as pool:
        for fut in as_completed([pool.submit(run_one, *j) for j in jobs]):
            sd, seed, score = fut.result()
            results.setdefault(sd, []).append(score)
            print(f'  sd={sd:.3f} seed={seed}: {score*100:.1f}%', flush=True)
    print(f'\nSUMMARY:')
    for sd in VALUES:
        scores = results[sd]
        print(f'  {sd:.3f}: {np.mean(scores)*100:.1f}% +/-{np.std(scores)*100:.1f}pp')

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
