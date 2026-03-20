"""
Meta-Learning v3: Two parallel approaches
==========================================
A) Linear rule: delta_w = tanh(a0*pre + a1*post + a2*err + a3*w + a4*phase + bias) * lr
   Only 6 params — fast to search, proof of concept.

B) SWG rule with bulk mutations (5 edges at a time) for stronger signal.

Both run sequentially, results compared.
Logs to: meta_learn_v3_log.jsonl / meta_learn_v3_live.txt
"""

import sys, os, time, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))
from graph import SelfWiringGraph

# ─── Data ────────────────────────────────────────────────────────────

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
        self.n_w1 = 2 * h
        self.n_w2 = h
        self.n_edges = self.n_w1 + self.n_w2

    def forward(self, x):
        self._x = x
        self._h = np.maximum(x @ self.w1, 0)
        self._out = self._h @ self.w2
        return 1.0 / (1.0 + np.exp(-np.clip(self._out, -10, 10)))

    def get_features_and_apply(self, error, phase, rule_fn):
        """Get features for all edges, apply rule, update weights in-place."""
        pre_m = self._x.mean(axis=0)    # (2,)
        post_m = self._h.mean(axis=0)   # (h,)
        out_m = float(np.mean(np.abs(self._out)))

        # Build feature matrix (n_edges, 5)
        feats = np.empty((self.n_edges, 5), dtype=np.float32)
        idx = 0
        for i in range(2):
            for j in range(self.h_size):
                feats[idx] = [pre_m[i], post_m[j], error, self.w1[i, j], phase]
                idx += 1
        for i in range(self.h_size):
            feats[idx] = [post_m[i], out_m, error, self.w2[i, 0], phase]
            idx += 1

        # Get deltas from rule
        deltas = rule_fn(feats)

        # Apply
        idx = 0
        for i in range(2):
            for j in range(self.h_size):
                self.w1[i, j] += deltas[idx]
                idx += 1
        for i in range(self.h_size):
            self.w2[i, 0] += deltas[idx]
            idx += 1


# ─── Fitness (shared) ───────────────────────────────────────────────

def compute_fitness(rule_fn, tasks=TASKS, inner_steps=30, mlp_h=8,
                    seeds=(42, 123, 777)):
    total = 0.0
    n = 0
    for seed in seeds:
        for _, tx, ty in tasks:
            mlp = TinyMLP(h=mlp_h, seed=seed)
            for step in range(inner_steps):
                pred = mlp.forward(tx)
                mse = float(np.mean((pred - ty) ** 2))
                phase = step / inner_steps
                mlp.get_features_and_apply(mse, phase, rule_fn)
            pred = mlp.forward(tx)
            final_mse = float(np.mean((pred - ty) ** 2))
            acc = float(((pred > 0.5).astype(np.float32) == ty).mean())
            total += 0.7 * (1.0 - final_mse) + 0.3 * acc
            n += 1
    return total / n


def xor_accuracy(rule_fn, mlp_h=8, inner_steps=30):
    """Quick XOR accuracy check."""
    mlp = TinyMLP(h=mlp_h, seed=42)
    for step in range(inner_steps):
        pred = mlp.forward(XOR_X)
        mse = float(np.mean((pred - XOR_Y) ** 2))
        mlp.get_features_and_apply(mse, step / inner_steps, rule_fn)
    pred = mlp.forward(XOR_X)
    return float(((pred > 0.5).astype(np.float32) == XOR_Y).mean())


# ═══ APPROACH A: Linear Rule ════════════════════════════════════════

class LinearRule:
    """delta_w = tanh(w @ features + bias) * lr"""
    def __init__(self):
        self.weights = np.random.randn(5).astype(np.float32) * 0.1
        self.bias = np.float32(0.0)
        self.lr = np.float32(0.05)

    def __call__(self, feats):
        """feats: (n_edges, 5) -> (n_edges,) delta_w"""
        raw = feats @ self.weights + self.bias
        return np.tanh(raw) * self.lr

    def save_state(self):
        return (self.weights.copy(), float(self.bias), float(self.lr))

    def restore_state(self, s):
        self.weights[:] = s[0]
        self.bias = np.float32(s[1])
        self.lr = np.float32(s[2])

    def mutate(self, sigma=0.05):
        """Gaussian mutation on all 7 params."""
        self.weights += np.random.randn(5).astype(np.float32) * sigma
        self.bias += np.float32(np.random.randn() * sigma)
        self.lr = np.float32(np.clip(self.lr + np.random.randn() * 0.005, 0.001, 0.5))

    def describe(self):
        names = ['pre', 'post', 'err', 'w', 'phase']
        parts = [f"{self.weights[i]:+.3f}*{names[i]}" for i in range(5)]
        return f"tanh({' '.join(parts)} {self.bias:+.3f}) * {self.lr:.4f}"


# ═══ APPROACH B: SWG Rule with bulk mutations ═══════════════════════

class SWGRule:
    """SWG as rule function, with configurable mutation strength."""
    def __init__(self, V=16):
        self.swg = SelfWiringGraph(V)
        self.lr = 0.1

    def __call__(self, feats):
        n = feats.shape[0]
        V, H = self.swg.V, self.swg.H
        inp = np.zeros((n, V), dtype=np.float32)
        inp[:, :5] = feats
        projected = inp @ self.swg.W_in
        charges = np.zeros((n, H), dtype=np.float32)
        acts = np.zeros((n, H), dtype=np.float32)
        retain = float(self.swg.retention)
        for t in range(4):
            if t == 0:
                acts = acts + projected
            raw = acts @ self.swg.mask
            np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            charges += raw
            charges *= retain
            acts = np.maximum(charges - self.swg.THRESHOLD, 0.0)
            charges = np.clip(charges, -1.0, 1.0)
        out = charges @ self.swg.W_out
        return np.tanh(out[:, 0]) * self.lr

    def save_state(self):
        return self.swg.save_state()

    def restore_state(self, s):
        self.swg.restore_state(s)

    def mutate(self, n_changes=5):
        """Bulk mutation: n_changes edges at once."""
        undo = self.swg.mutate()
        # Extra mutations for stronger signal
        for _ in range(n_changes - 1):
            extra = self.swg.mutate(forced_op='rewire')
        return undo


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


# ─── Sweep one approach ─────────────────────────────────────────────

def sweep(name, rule, max_att=20000, log_jsonl=None, log_live=None):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    # Benchmark
    t0 = time.time()
    score = compute_fitness(rule)
    dt = time.time() - t0
    print(f"  1 eval: {dt*1000:.0f}ms | ETA: {max_att*dt/60:.0f} min")
    print(f"  Initial fitness: {score:.4f}")

    best = score
    best_state = rule.save_state()
    stale = 0
    accepts = 0
    best_xor = xor_accuracy(rule)

    log_entry = lambda **kw: json.dumps({**kw, "approach": name})

    if log_jsonl:
        with open(log_jsonl, "a") as f:
            f.write(log_entry(att=0, score=round(score, 4), best=round(best, 4),
                              event="init") + "\n")

    for att in range(1, max_att + 1):
        if att % 20 == 0:
            cpu_throttle(75)

        state = rule.save_state()
        rule.mutate()
        new_score = compute_fitness(rule)

        if new_score > score:
            score = new_score
            stale = 0
            accepts += 1
            if score > best:
                best = score
                best_state = rule.save_state()
                xor_acc = xor_accuracy(rule)
                best_xor = max(best_xor, xor_acc)

                if log_jsonl:
                    with open(log_jsonl, "a") as f:
                        f.write(log_entry(att=att, score=round(score, 4),
                                          best=round(best, 4), xor_acc=round(xor_acc, 3),
                                          accepts=accepts, elapsed=round(time.time()-t0, 1),
                                          event="new_best") + "\n")

                if hasattr(rule, 'describe'):
                    print(f"    NEW BEST [{att}] {score:.4f} XOR={xor_acc*100:.0f}% | {rule.describe()}")
                else:
                    print(f"    NEW BEST [{att}] {score:.4f} XOR={xor_acc*100:.0f}%")
        else:
            rule.restore_state(state)
            stale += 1

        if att % 500 == 0:
            elapsed = time.time() - t0
            rate = att / elapsed
            line = (f"  [{att:6d}/{max_att}] best={best:.4f} xor={best_xor*100:.0f}% "
                    f"stale={stale} accepts={accepts} rate={rate:.0f}/s {elapsed:.0f}s")
            print(line, flush=True)
            if log_live:
                with open(log_live, "a") as f:
                    f.write(f"[{name}] {line}\n")
            if log_jsonl:
                with open(log_jsonl, "a") as f:
                    f.write(log_entry(att=att, score=round(score, 4),
                                      best=round(best, 4), xor_best=round(best_xor, 3),
                                      stale=stale, accepts=accepts, rate=round(rate, 1),
                                      elapsed=round(elapsed, 1), event="checkpoint") + "\n")

        if best_xor >= 1.0 and best >= 0.95:
            print(f"\n  === {name}: SOLVED at att {att}! ===")
            break

        if stale >= 3000:
            print(f"    [STALE RESET at {att}]")
            # Restore best and re-mutate from there
            rule.restore_state(best_state)
            score = best
            stale = 0

    elapsed = time.time() - t0
    print(f"  {name} DONE | best={best:.4f} xor={best_xor*100:.0f}% | {att} att | {elapsed:.0f}s | {accepts} accepts")
    return best, best_xor, best_state


# ─── Main ────────────────────────────────────────────────────────────

def main():
    base = os.path.dirname(os.path.abspath(__file__))
    LOG_JSONL = os.path.join(base, "meta_learn_v3_log.jsonl")
    LOG_LIVE  = os.path.join(base, "meta_learn_v3_live.txt")

    with open(LOG_LIVE, "w") as f:
        f.write(f"Meta-Learning v3 | {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    with open(LOG_JSONL, "w") as f:
        pass  # truncate

    print("Meta-Learning v3: Linear vs SWG rule")
    print(f"  Log: {LOG_JSONL}")

    # A) Linear rule — fast, proof of concept
    linear = LinearRule()
    best_lin, xor_lin, state_lin = sweep(
        "LINEAR", linear, max_att=20000, log_jsonl=LOG_JSONL, log_live=LOG_LIVE)

    # B) SWG rule — bulk mutations
    swg = SWGRule(V=16)
    best_swg, xor_swg, state_swg = sweep(
        "SWG-BULK", swg, max_att=20000, log_jsonl=LOG_JSONL, log_live=LOG_LIVE)

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  LINEAR:   best={best_lin:.4f}  XOR={xor_lin*100:.0f}%")
    print(f"  SWG-BULK: best={best_swg:.4f}  XOR={xor_swg*100:.0f}%")
    winner = "LINEAR" if best_lin > best_swg else "SWG-BULK"
    print(f"  WINNER: {winner}")

    # If linear worked, show the learned rule
    if xor_lin >= 0.75:
        linear.restore_state(state_lin)
        print(f"\n  Learned linear rule: {linear.describe()}")

    with open(LOG_LIVE, "a") as f:
        f.write(f"\nSUMMARY: LINEAR={best_lin:.4f}/{xor_lin*100:.0f}% vs SWG={best_swg:.4f}/{xor_swg*100:.0f}%\n")
        f.write(f"WINNER: {winner}\n")

    with open(LOG_JSONL, "a") as f:
        f.write(json.dumps({"linear_best": round(best_lin, 4), "linear_xor": round(xor_lin, 3),
                            "swg_best": round(best_swg, 4), "swg_xor": round(xor_swg, 3),
                            "winner": winner, "event": "summary"}) + "\n")


if __name__ == "__main__":
    main()
