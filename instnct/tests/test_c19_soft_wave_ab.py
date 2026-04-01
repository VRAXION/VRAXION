
import sys, os, time, random
import numpy as np

sys.path.insert(0, os.getcwd())
from instnct.model.graph import SelfWiringGraph

# Config
VOCAB = 32
HIDDEN_RATIO = 4
TICKS = 12
STEPS = 1000
N_TRIALS = 3
RHO = 0.5 # Fixed Rho (modulation depth)

class C19RevisedGraph(SelfWiringGraph):
    """
    Variant B: C19 style with Learnable C (period) and Fixed Rho.
    The threshold itself oscillates instead of a hard gate.
    """
    def rollout_c19(self, injected, ticks):
        H, state, charge, ref = self.H, np.zeros(self.H), np.zeros(self.H), np.zeros(self.H, dtype=np.int8)
        for t in range(ticks):
            # 1. Leak
            charge = np.maximum(charge - 1.0, 0.0)
            # 2. Input
            if t < 1: charge += injected * 8.0
            # 3. Propagate
            charge += self._sparse_mul_1d(state)
            np.clip(charge, 0, 15, out=charge)
            
            # 4. C19 WAVE: Threshold modulation (Soft)
            # freq is basically 2*pi/C. We keep the freq name for code compatibility.
            wave = np.sin(t * self.freq + self.phase)
            effective_theta = self.theta * (1.0 + RHO * wave)
            
            # 5. Decision (No hard rhythm gate, just the waving threshold)
            fired = (charge >= effective_theta) & (ref == 0)
            
            state = fired.astype(np.float32) * self._polarity_f32
            charge[fired] = 0.0
            ref[ref > 0] -= 1; ref[fired] = 1
        return state, charge

    def forward(self, world, ticks=TICKS):
        world_vec = np.asarray(world, dtype=np.float32)
        injected = world_vec @ self.input_projection
        self.state, self.charge = self.rollout_c19(injected, ticks)
        return self.readout(self.state) # Spike Readout

def evaluate_net(net, data):
    correct = 0
    for i in range(len(data)-1):
        w = np.zeros(VOCAB); w[i % VOCAB] = 1.0 # Permutation-like evaluation
        logits = net.forward(w)
        if np.argmax(logits) == data[i+1]: correct += 1
    return correct / (len(data)-1)

def run_trial(net_cls, seed, data):
    random.seed(seed); np.random.seed(seed)
    net = net_cls(VOCAB, hidden_ratio=HIDDEN_RATIO, seed=seed)
    best_acc = evaluate_net(net, data)
    for _ in range(STEPS):
        undo = net.mutate()
        acc = evaluate_net(net, data)
        if acc >= best_acc: best_acc = acc
        else: net.replay(undo)
    return best_acc

def test_c19_vs_musical():
    print("=" * 75)
    print(f"  A/B Test: Musical Gating (Hard) vs C19 Wave (Soft)")
    print(f"  Fixed Rho={RHO}, Learnable C/Freq, Spike Readout")
    print("=" * 75)
    
    TEXT = "the rhythm of the brain is like a symphony. soft waves vs hard gates." * 20
    data = [ord(c) % VOCAB for c in TEXT[:500]]
    
    results = {'A': [], 'B': []}
    for i in range(N_TRIALS):
        seed = 555 + i * 88
        print(f"Trial {i+1}...")
        a = run_trial(SelfWiringGraph, seed, data)
        b = run_trial(C19RevisedGraph, seed, data)
        results['A'].append(a); results['B'].append(b)
        print(f"  A (Musical Gate): {a*100:5.2f}% | B (C19 Soft Wave): {b*100:5.2f}%")
        
    print("-" * 75)
    print(f"MEAN A: {np.mean(results['A'])*100:.2f}%")
    print(f"MEAN B: {np.mean(results['B'])*100:.2f}%")
    delta = (np.mean(results['B']) - np.mean(results['A'])) * 100
    print(f"Delta:  {delta:+.2f}%")

if __name__ == "__main__":
    test_c19_vs_musical()
