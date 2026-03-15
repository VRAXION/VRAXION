"""DRIVE sweep: can we use 1.0 (no multiply needed)?"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np, random, multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from model.graph import SelfWiringGraph, train

SEEDS = [42, 77, 123, 0, 1]
DRIVES = [0.5, 0.6, 0.8, 1.0, 1.5, 2.0]

def run(drive, seed):
    SelfWiringGraph.DRIVE = drive
    np.random.seed(seed); random.seed(seed)
    net = SelfWiringGraph(64)
    perm = np.random.permutation(64)
    best = train(net, perm, 64, max_attempts=16000, ticks=8, verbose=False)
    SelfWiringGraph.DRIVE = 0.6
    return drive, seed, best

def main():
    jobs = [(d, s) for d in DRIVES for s in SEEDS]
    print(f'DRIVE SWEEP: {len(jobs)} jobs parallel', flush=True)
    results = {}
    with ProcessPoolExecutor(max_workers=22) as pool:
        for fut in as_completed([pool.submit(run, *j) for j in jobs]):
            d, s, score = fut.result()
            results.setdefault(d, []).append(score)
            print(f'  drive={d} seed={s}: {score*100:.1f}%', flush=True)
    print(f'\nSUMMARY:')
    for d in DRIVES:
        scores = results.get(d, [])
        print(f'  DRIVE={d}: {np.mean(scores)*100:.1f}% +/-{np.std(scores)*100:.1f}pp')

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
