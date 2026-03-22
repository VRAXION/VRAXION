"""Copy vs undo scaling: where does undo beat memcpy?
Tests at N=64,128,192,384,768,1536 — find the crossover."""
import numpy as np
import random
import time

Ns = [64, 128, 192, 384, 768, 1536]
DENSITY = 0.06
REPEATS = 5000


def bench_copy(mask):
    """Full mask.copy() + restore."""
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter_ns()
        saved = mask.copy()
        mask[:] = saved
        times.append(time.perf_counter_ns() - t0)
    return np.median(times)


def bench_resync(mask):
    """np.where rebuild of alive list."""
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter_ns()
        rows, cols = np.where(mask != 0)
        alive = list(zip(rows.tolist(), cols.tolist()))
        times.append(time.perf_counter_ns() - t0)
    return np.median(times)


def bench_replay_resync(mask, n_changes=15):
    """Undo 15 flips + resync alive."""
    alive_count = int((mask != 0).sum())
    positions = []
    rows, cols = np.where(mask != 0)
    if len(rows) >= n_changes:
        idxs = np.random.choice(len(rows), n_changes, replace=False)
        positions = [(rows[i], cols[i]) for i in idxs]

    times = []
    for _ in range(REPEATS):
        # simulate: flip n_changes cells, then undo
        for r, c in positions:
            mask[r, c] *= -1
        t0 = time.perf_counter_ns()
        # replay: flip back
        for r, c in positions:
            mask[r, c] *= -1
        # resync
        rs, cs = np.where(mask != 0)
        alive = list(zip(rs.tolist(), cs.tolist()))
        times.append(time.perf_counter_ns() - t0)
    return np.median(times)


def bench_replay_set(mask, n_changes=15):
    """Undo 15 flips + set-based alive (no resync needed)."""
    rows, cols = np.where(mask != 0)
    alive_set = set(zip(rows.tolist(), cols.tolist()))
    alive_list = list(alive_set)

    positions = []
    if len(alive_list) >= n_changes:
        positions = random.sample(alive_list, n_changes)

    times = []
    for _ in range(REPEATS):
        # simulate flips (set doesn't change for flip)
        for r, c in positions:
            mask[r, c] *= -1
        t0 = time.perf_counter_ns()
        # replay: flip back (set doesn't need update for flips)
        for r, c in positions:
            mask[r, c] *= -1
        times.append(time.perf_counter_ns() - t0)
    return np.median(times)


def bench_replay_set_structural(mask, n_changes=15):
    """Undo 15 add/removes via set — the realistic case."""
    rows, cols = np.where(mask != 0)
    alive_set = set(zip(rows.tolist(), cols.tolist()))

    times = []
    N = mask.shape[0]
    for _ in range(REPEATS):
        # simulate: add some, remove some
        added = []
        removed = []
        for i in range(n_changes):
            if i % 2 == 0:
                r, c = random.randint(0, N-1), random.randint(0, N-1)
                if r != c and mask[r, c] == 0:
                    mask[r, c] = 1
                    alive_set.add((r, c))
                    added.append((r, c))
            else:
                if alive_set:
                    item = random.choice(list(alive_set))
                    old = mask[item[0], item[1]]
                    mask[item[0], item[1]] = 0
                    alive_set.discard(item)
                    removed.append((item[0], item[1], old))

        t0 = time.perf_counter_ns()
        # undo via set
        for r, c in reversed(added):
            mask[r, c] = 0
            alive_set.discard((r, c))
        for r, c, old in reversed(removed):
            mask[r, c] = old
            alive_set.add((r, c))
        times.append(time.perf_counter_ns() - t0)
    return np.median(times)


def main():
    print(f"COPY vs UNDO SCALING SWEEP", flush=True)
    print(f"Density={DENSITY}, repeats={REPEATS}", flush=True)
    print(f"{'N':>6s} {'mask_KB':>8s} {'copy':>10s} {'resync':>10s} {'undo+rsync':>10s} "
          f"{'undo_set':>10s} {'set_struct':>10s} {'winner':>10s}", flush=True)
    print("-" * 80, flush=True)

    for N in Ns:
        mask = np.zeros((N, N), dtype=np.int8)
        r = np.random.rand(N, N)
        mask[r < DENSITY / 2] = -1
        mask[r > 1 - DENSITY / 2] = 1
        np.fill_diagonal(mask, 0)

        mask_kb = N * N / 1024
        t_copy = bench_copy(mask)
        t_resync = bench_resync(mask)
        t_replay_resync = bench_replay_resync(mask)
        t_set_flip = bench_replay_set(mask)
        t_set_struct = bench_replay_set_structural(mask)

        winner = "COPY" if t_copy < min(t_replay_resync, t_set_struct) else \
                 "SET" if t_set_struct < t_replay_resync else "RESYNC"

        print(f"{N:6d} {mask_kb:7.0f}KB {t_copy:9.0f}ns {t_resync:9.0f}ns "
              f"{t_replay_resync:9.0f}ns {t_set_flip:9.0f}ns {t_set_struct:9.0f}ns "
              f"{winner:>10s}", flush=True)

    print("-" * 80, flush=True)


if __name__ == '__main__':
    main()
