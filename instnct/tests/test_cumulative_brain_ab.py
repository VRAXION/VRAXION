
import sys, os, time, random
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.graph import SelfWiringGraph

# Config
VOCAB = 32
HIDDEN_RATIO = 3
TICKS = 6
STEPS = 800  # Give them some room to learn
N_TRIALS = 5

def make_perm_targets(V, seed):
    np.random.seed(seed)
    return np.random.permutation(V)

class BaselineGraph(SelfWiringGraph):
    """Vanilla SWG: No Polarity, No Refractory, No Reset."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.polarity.fill(1)
        self._polarity_f32.fill(1.0)

    def rollout_baseline(self, injected, ticks):
        H = self.H
        state = np.zeros(H, dtype=np.float32)
        charge = np.zeros(H, dtype=np.float32)
        ret = 1.0 - self.decay
        for t in range(ticks):
            if t < 1: charge += injected
            raw = self._sparse_mul_1d(state)
            charge += raw
            charge *= ret
            state = np.maximum(charge - self.theta, 0.0)
            charge = np.maximum(charge, 0.0)
        return state, charge

    def forward(self, world, ticks=6):
        world_vec = np.asarray(world, dtype=np.float32)
        injected = world_vec @ self.input_projection
        self.state, self.charge = self.rollout_baseline(injected, ticks)
        return self.readout(self.charge)

class DaleOnlyGraph(SelfWiringGraph):
    """SWG + Dale Polarity, but NO Refractory and NO Reset."""
    def rollout_dale(self, injected, ticks):
        H = self.H
        state = np.zeros(H, dtype=np.float32)
        charge = np.zeros(H, dtype=np.float32)
        ret = 1.0 - self.decay
        for t in range(ticks):
            if t < 1: charge += injected
            raw = self._sparse_mul_1d(state)
            charge += raw
            charge *= ret
            state = np.maximum(charge - self.theta, 0.0) * self._polarity_f32
            charge = np.maximum(charge, 0.0)
        return state, charge

    def forward(self, world, ticks=6):
        world_vec = np.asarray(world, dtype=np.float32)
        injected = world_vec @ self.input_projection
        self.state, self.charge = self.rollout_dale(injected, ticks)
        return self.readout(self.charge)

# The current mainline 'SelfWiringGraph' already includes Dale + Refractory + Reset.

def evaluate_perm(net, targets, V):
    correct = 0
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
    return best_acc

def test_cumulative_ab():
    print("=" * 80)
    print("  Cumulative Brain Improvement A/B Test")
    print("  A = Baseline (No Dale, No Reset, No Refractory)")
    print("  B = Dale Law Only (+6% candidate)")
    print("  C = Mainline Peak (Dale + Refractory + Partial Reset) (+13% candidate)")
    print(f"  V={VOCAB}, H={VOCAB*HIDDEN_RATIO}, steps={STEPS}, trials={N_TRIALS}")
    print("=" * 80)

    results = {'A': [], 'B': [], 'C': []}
    for trial in range(N_TRIALS):
        seed = 42 + trial * 111
        targets = make_perm_targets(VOCAB, seed)
        
        acc_a = run_evo(BaselineGraph, seed + 1000, targets)
        results['A'].append(acc_a)
        
        acc_b = run_evo(DaleOnlyGraph, seed + 1000, targets)
        results['B'].append(acc_b)
        
        acc_c = run_evo(SelfWiringGraph, seed + 1000, targets)
        results['C'].append(acc_c)
        
        print(f"  Trial {trial+1}: A={acc_a*100:5.1f}%  B={acc_b*100:5.1f}%  C={acc_c*100:5.1f}%")

    print("\n" + "─" * 80)
    print(f"  FINAL MEAN A (Baseline):  {np.mean(results['A'])*100:5.1f}%")
    print(f"  FINAL MEAN B (Dale Only): {np.mean(results['B'])*100:5.1f}% (Delta: {(np.mean(results['B'])-np.mean(results['A']))*100:+.1f}%)")
    print(f"  FINAL MEAN C (Peak Brain):{np.mean(results['C'])*100:5.1f}% (Delta: {(np.mean(results['C'])-np.mean(results['A']))*100:+.1f}%)")
    print("─" * 80)

if __name__ == '__main__':
    test_cumulative_ab()
