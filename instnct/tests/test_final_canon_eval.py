
import sys, os, time, random
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.graph import SelfWiringGraph

# Config
VOCAB = 32
HIDDEN_RATIO = 3
TICKS = 8
STEPS = 1000  # More steps to see if the new canon can pull ahead
N_TRIALS = 3

def make_perm_targets(V, seed):
    np.random.seed(seed)
    return np.random.permutation(V)

class LegacySoftGraph(SelfWiringGraph):
    """Replicates the old Soft-Spike, Percent Leak dynamics."""
    def rollout_token(self, injected, **kwargs):
        # Force old-style dynamics
        H = self.H
        state = np.zeros(H, dtype=np.float32)
        charge = np.zeros(H, dtype=np.float32)
        ret = 0.85 # Old default decay
        theta = 0.1 # Old default theta
        for t in range(kwargs.get('ticks', 6)):
            if t < 1: charge += injected
            raw = self._sparse_mul_1d(state)
            charge += raw
            charge *= ret
            state = np.maximum(charge - theta, 0.0)
            charge = np.maximum(charge, 0.0)
        return state, charge

def evaluate_perm(net, targets, V):
    correct = 0
    total_fired = 0
    # Capture telemetry if possible
    for i in range(V):
        w = np.zeros(V); w[i] = 1.0
        logits = net.forward(w, ticks=TICKS)
        if np.argmax(logits) == targets[i]:
            correct += 1
    return correct / V

def run_evo(net_cls, seed, targets, steps=STEPS):
    random.seed(seed); np.random.seed(seed)
    net = net_cls(VOCAB, hidden_ratio=HIDDEN_RATIO, seed=seed)
    best_acc = evaluate_perm(net, targets, VOCAB)
    for _ in range(steps):
        undo = net.mutate()
        net.reset()
        acc = evaluate_perm(net, targets, VOCAB)
        if acc >= best_acc: best_acc = acc
        else: net.replay(undo)
    return best_acc, net

def test_final_canon():
    print("=" * 80)
    print("  FINAL CANON EVALUATION: Int4 Brain vs Legacy Soft-Spike")
    print(f"  V={VOCAB}, H={VOCAB*HIDDEN_RATIO}, steps={STEPS}, trials={N_TRIALS}")
    print("=" * 80)

    results = {'Legacy': [], 'Int4Canon': []}
    
    for trial in range(N_TRIALS):
        seed = 42 + trial * 555
        targets = make_perm_targets(VOCAB, seed)
        
        print(f"  Trial {trial+1}...")
        acc_legacy, _ = run_evo(LegacySoftGraph, seed, targets)
        results['Legacy'].append(acc_legacy)
        
        acc_canon, best_net = run_evo(SelfWiringGraph, seed, targets)
        results['Int4Canon'].append(acc_canon)
        
        print(f"    Legacy: {acc_legacy*100:5.1f}% | Int4 Canon: {acc_canon*100:5.1f}%")

    print("\n" + "─" * 80)
    print(f"  FINAL MEAN Legacy:     {np.mean(results['Legacy'])*100:5.1f}%")
    print(f"  FINAL MEAN Int4 Canon: {np.mean(results['Int4Canon'])*100:5.1f}%")
    delta = (np.mean(results['Int4Canon']) - np.mean(results['Legacy'])) * 100
    print(f"  Overall Delta:         {delta:+.1f}%")
    print("─" * 80)

    # Telemetry for the best Canon network
    print("\n  [TELEMETRY] Int4 Canon Internal Dynamics (Best Net):")
    print(f"  {'Tick':>4} | {'Firing Rate %':>12} | {'Avg Charge (0-15)':>15}")
    print("  " + "-" * 45)
    
    # Run one pass with telemetry
    w = np.zeros(VOCAB); w[5] = 1.0
    injected = w @ best_net.input_projection
    
    # Manual rollout to capture stats
    h = best_net.H
    s = np.zeros(h); c = np.zeros(h); r = np.zeros(h)
    for t in range(TICKS):
        c = np.maximum(c - 1.0, 0.0) # Leak
        if t < 1: c += injected * 8.0 # Sensory scaling
        c += best_net._sparse_mul_1d(s)
        np.clip(c, 0, 15, out=c)
        fired = (c >= best_net.theta) & (r == 0)
        r[r > 0] -= 1; r[fired] = 1
        s = fired.astype(np.float32) * best_net._polarity_f32
        c[fired] = 0.0
        
        print(f"  {t+1:4d} | {np.mean(fired)*100:12.1f}% | {np.mean(c):15.2f}")

if __name__ == '__main__':
    test_final_canon()
