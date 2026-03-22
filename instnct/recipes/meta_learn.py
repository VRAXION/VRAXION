"""
Meta-Learning Sweep v2: SWG learns a learning rule for a tiny MLP.
===================================================================
Fix: continuous fitness (1-MSE), higher lr, multi-seed averaging.

Logs to: meta_learn_log.jsonl / meta_learn_live.txt
"""

import sys, os, time, json
import numpy as np

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
from graph import SelfWiringGraph

# ─── XOR dataset ────────────────────────────────────────────────────

XOR_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
XOR_Y = np.array([[0], [1], [1], [0]], dtype=np.float32)
AND_Y = np.array([[0], [0], [0], [1]], dtype=np.float32)
OR_Y  = np.array([[0], [1], [1], [1]], dtype=np.float32)

TASKS = [("XOR", XOR_X, XOR_Y), ("AND", XOR_X, AND_Y), ("OR", XOR_X, OR_Y)]


# ─── Tiny MLP ────────────────────────────────────────────────────────

class TinyMLP:
    def __init__(self, h=8, seed=42):
        rng = np.random.RandomState(seed)
        self.w1 = rng.randn(2, h).astype(np.float32) * 0.3
        self.w2 = rng.randn(h, 1).astype(np.float32) * 0.3
        self.h_size = h
        self.n_edges = 2 * h + h  # w1 + w2

    def forward(self, x):
        self._x = x
        self._h = np.maximum(x @ self.w1, 0)  # ReLU
        self._out = self._h @ self.w2
        return self._out

    def sigmoid_out(self, x):
        """Forward + sigmoid for proper 0-1 output."""
        raw = self.forward(x)
        return 1.0 / (1.0 + np.exp(-np.clip(raw, -10, 10)))

    def get_edge_features(self, error, phase):
        """(n_edges, 5): [pre, post, error, weight, phase]"""
        feats = np.empty((self.n_edges, 5), dtype=np.float32)
        pre_m = self._x.mean(axis=0)
        post_m = self._h.mean(axis=0)
        out_m = float(np.mean(np.abs(self._out)))
        idx = 0
        for i in range(2):
            for j in range(self.h_size):
                feats[idx] = [pre_m[i], post_m[j], error, self.w1[i, j], phase]
                idx += 1
        for i in range(self.h_size):
            feats[idx] = [post_m[i], out_m, error, self.w2[i, 0], phase]
            idx += 1
        return feats

    def apply_deltas(self, deltas):
        idx = 0
        for i in range(2):
            for j in range(self.h_size):
                self.w1[i, j] += deltas[idx]
                idx += 1
        for i in range(self.h_size):
            self.w2[i, 0] += deltas[idx]
            idx += 1


# ─── Batched SWG rule ───────────────────────────────────────────────

def swg_rule_batch(swg, edge_features, lr=0.1, ticks=4):
    """All edges through SWG at once. Returns (n_edges,) delta_w."""
    n = edge_features.shape[0]
    V, H = swg.V, swg.H
    inp = np.zeros((n, V), dtype=np.float32)
    inp[:, :5] = edge_features
    projected = inp @ swg.input_projection
    charges = np.zeros((n, H), dtype=np.float32)
    acts = np.zeros((n, H), dtype=np.float32)
    retain = float(swg.retention_mean)
    for t in range(ticks):
        if t == 0:
            acts = acts + projected
        raw = acts @ swg.mask
        np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        charges += raw
        charges *= retain
        acts = np.maximum(charges - swg.theta_mean, 0.0)
        charges = np.clip(charges, -1.0, 1.0)
    out = charges @ swg.output_projection
    # Use tanh to bound output, then scale by lr
    return np.tanh(out[:, 0]) * lr


# ─── Fitness ─────────────────────────────────────────────────────────

def fitness(swg, tasks=TASKS, inner_steps=50, mlp_h=8, lr=0.1, seeds=(42, 123, 777)):
    """Train fresh MLPs on tasks using SWG rule. Continuous fitness = 1 - mean_MSE.
    Average over multiple seeds to reduce noise."""
    total_quality = 0.0
    n_evals = 0

    for seed in seeds:
        for _, tx, ty in tasks:
            mlp = TinyMLP(h=mlp_h, seed=seed)
            for step in range(inner_steps):
                pred = mlp.sigmoid_out(tx)
                mse = float(np.mean((pred - ty) ** 2))
                phase = step / inner_steps
                feats = mlp.get_edge_features(mse, phase)
                deltas = swg_rule_batch(swg, feats, lr=lr, ticks=4)
                mlp.apply_deltas(deltas)

            # Final eval: continuous quality
            pred = mlp.sigmoid_out(tx)
            final_mse = float(np.mean((pred - ty) ** 2))
            acc = float(((pred > 0.5).astype(np.float32) == ty).mean())
            # Blend: mostly continuous MSE + some accuracy signal
            quality = 0.7 * (1.0 - final_mse) + 0.3 * acc
            total_quality += quality
            n_evals += 1

    return total_quality / n_evals


# ─── CPU throttle ───────────────────────────────────────────────────

def cpu_throttle(max_cpu=75):
    try:
        import psutil
        usage = psutil.cpu_percent(interval=0.05)
        while usage > max_cpu:
            time.sleep(0.5)
            usage = psutil.cpu_percent(interval=0.1)
    except ImportError:
        pass


# ─── Main ────────────────────────────────────────────────────────────

def main():
    base = os.path.dirname(os.path.abspath(__file__))
    LOG_JSONL = os.path.join(base, "meta_learn_log.jsonl")
    LOG_LIVE  = os.path.join(base, "meta_learn_live.txt")

    V = 16
    MAX_ATT = 50000
    INNER_STEPS = 30        # Reduced for speed (3 seeds × 3 tasks = 9 MLPs)
    MLP_H = 8
    LR = 0.1               # Much bigger lr
    CPU_MAX = 75
    SEEDS = (42, 123, 777)  # 3 seeds for noise reduction

    print(f"Meta-Learning Sweep v2")
    print(f"  SWG V={V} | inner={INNER_STEPS} | h={MLP_H} | lr={LR} | seeds={len(SEEDS)}")

    # Benchmark
    swg = SelfWiringGraph(V)
    t_bench = time.time()
    score = fitness(swg, inner_steps=INNER_STEPS, mlp_h=MLP_H, lr=LR, seeds=SEEDS)
    dt = time.time() - t_bench
    print(f"  1 eval: {dt*1000:.0f}ms | ETA {MAX_ATT}: {MAX_ATT*dt/60:.0f} min")
    print(f"  Initial fitness: {score:.4f}")
    print("=" * 60)

    best = score
    stale = 0
    accepts = 0
    best_acc_seen = 0.0
    t0 = time.time()

    with open(LOG_LIVE, "w") as f:
        f.write(f"START v2 | V={V} score={score:.4f} conns={swg.count_connections()}\n")
    with open(LOG_JSONL, "w") as f:
        f.write(json.dumps({"att": 0, "score": round(score, 4), "best": round(best, 4),
                            "conns": swg.count_connections(), "event": "init"}) + "\n")

    for att in range(1, MAX_ATT + 1):
        if att % 20 == 0:
            cpu_throttle(CPU_MAX)

        old_loss = int(swg.loss_pct)
        old_drive = int(swg.mutation_drive)
        undo = swg.mutate()
        new_score = fitness(swg, inner_steps=INNER_STEPS, mlp_h=MLP_H, lr=LR, seeds=SEEDS)

        if new_score > score:
            score = new_score
            stale = 0
            accepts += 1
            if score > best:
                best = score
                # Quick accuracy check on XOR
                mlp = TinyMLP(h=MLP_H, seed=42)
                for step in range(INNER_STEPS):
                    pred = mlp.sigmoid_out(XOR_X)
                    mse = float(np.mean((pred - XOR_Y)**2))
                    feats = mlp.get_edge_features(mse, step/INNER_STEPS)
                    deltas = swg_rule_batch(swg, feats, lr=LR, ticks=4)
                    mlp.apply_deltas(deltas)
                xor_pred = mlp.sigmoid_out(XOR_X)
                xor_acc = float(((xor_pred > 0.5).astype(np.float32) == XOR_Y).mean())
                best_acc_seen = max(best_acc_seen, xor_acc)

                with open(LOG_JSONL, "a") as f:
                    f.write(json.dumps({
                        "att": att, "score": round(score, 4), "best": round(best, 4),
                        "xor_acc": round(xor_acc, 3),
                        "conns": swg.count_connections(), "drive": int(swg.mutation_drive),
                        "loss_pct": int(swg.loss_pct), "accepts": accepts,
                        "elapsed": round(time.time() - t0, 1), "event": "new_best"
                    }) + "\n")
        else:
            swg.replay(undo)
            swg.loss_pct = np.int8(old_loss)
            swg.mutation_drive = np.int8(old_drive)
            stale += 1

        if att % 100 == 0:
            elapsed = time.time() - t0
            rate = att / elapsed if elapsed > 0 else 0
            line = (f"[{att:6d}/{MAX_ATT}] best={best:.4f} score={score:.4f} "
                    f"xor_best={best_acc_seen*100:.0f}% stale={stale} "
                    f"conns={swg.count_connections()} accepts={accepts} "
                    f"rate={rate:.1f}/s {elapsed:.0f}s")
            print(line, flush=True)
            with open(LOG_LIVE, "a") as f:
                f.write(line + "\n")
            with open(LOG_JSONL, "a") as f:
                f.write(json.dumps({
                    "att": att, "score": round(score, 4), "best": round(best, 4),
                    "xor_best": round(best_acc_seen, 3),
                    "conns": swg.count_connections(), "stale": stale,
                    "accepts": accepts, "rate": round(rate, 2),
                    "elapsed": round(elapsed, 1), "event": "checkpoint"
                }) + "\n")

        if best_acc_seen >= 1.0 and best >= 0.95:
            print(f"\n=== XOR SOLVED at attempt {att}! ===")
            break

        if stale >= 5000:
            print(f"  [STALE RESET at {att}]")
            swg = SelfWiringGraph(V)
            score = fitness(swg, inner_steps=INNER_STEPS, mlp_h=MLP_H, lr=LR, seeds=SEEDS)
            stale = 0
            with open(LOG_JSONL, "a") as f:
                f.write(json.dumps({"att": att, "score": round(score, 4),
                                    "best": round(best, 4), "event": "stale_reset"}) + "\n")

    elapsed = time.time() - t0
    final = f"DONE | Best: {best:.4f} | XOR acc: {best_acc_seen*100:.0f}% | Att: {att} | {elapsed:.0f}s | Accepts: {accepts}"
    print(f"\n{final}")
    with open(LOG_JSONL, "a") as f:
        f.write(json.dumps({"best": round(best, 4), "xor_best": round(best_acc_seen, 3),
                            "att": att, "elapsed": round(elapsed, 1),
                            "accepts": accepts, "event": "done"}) + "\n")
    with open(LOG_LIVE, "a") as f:
        f.write(f"\n{final}\n")


if __name__ == "__main__":
    main()

