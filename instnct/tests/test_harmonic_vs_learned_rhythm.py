
import sys, os, time, random
import numpy as np

sys.path.insert(0, os.getcwd())
from instnct.model.graph import SelfWiringGraph

# Config
VOCAB = 32
HIDDEN_RATIO = 4
TICKS = 12
STEPS = 800
N_TRIALS = 3

class HarmonicGraph(SelfWiringGraph):
    """Option A: Global harmonic frequencies (1, 2, 3, 4)."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.period = np.random.randint(1, 5, size=self.H).astype(np.int8)
        self.phase = np.random.randint(0, 4, size=self.H).astype(np.int8)

    def rollout(self, injected, ticks):
        H, state, charge, ref = self.H, np.zeros(self.H), np.zeros(self.H), np.zeros(self.H, dtype=np.int8)
        for t in range(ticks):
            charge = np.maximum(charge - 1.0, 0.0)
            if t < 1: charge += injected * 8.0
            charge += self._sparse_mul_1d(state)
            np.clip(charge, 0, 15, out=charge)
            is_turn = ((t + self.phase) % self.period == 0)
            fired = (charge >= self.theta) & (ref == 0) & is_turn
            state = fired.astype(np.float32) * self._polarity_f32
            charge[fired] = 0.0
            ref[ref > 0] -= 1; ref[fired] = 1
        return state, charge

    def forward(self, world, ticks=TICKS):
        world_vec = np.asarray(world, dtype=np.float32)
        injected = world_vec @ self.input_projection
        self.state, self.charge = self.rollout(injected, ticks)
        return self.readout(self.state) # Spike Readout

    def mutate(self):
        undo = super().mutate()
        if random.random() < 0.1:
            idx = random.randint(0, self.H - 1)
            undo.append(('PERIOD', idx, self.period[idx]))
            self.period[idx] = random.randint(1, 4)
        return undo

    def replay(self, undo):
        for entry in reversed(undo):
            if entry[0] == 'PERIOD': self.period[entry[1]] = entry[2]
        super().replay([op for op in undo if op[0] != 'PERIOD'])

class LearnedFreqGraph(SelfWiringGraph):
    """Option B: Per-neuron learned continuous frequency."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # We use a sin-wave threshold modulation similar to C19 but as a gate
        self.freq = np.random.uniform(0.5, 2.0, size=self.H).astype(np.float32)
        self.phase = np.random.uniform(0, 2*np.pi, size=self.H).astype(np.float32)

    def rollout(self, injected, ticks):
        H, state, charge, ref = self.H, np.zeros(self.H), np.zeros(self.H), np.zeros(self.H, dtype=np.int8)
        for t in range(ticks):
            charge = np.maximum(charge - 1.0, 0.0)
            if t < 1: charge += injected * 8.0
            charge += self._sparse_mul_1d(state)
            np.clip(charge, 0, 15, out=charge)
            # Freq-based gating: neuron fires only at peaks of its internal wave
            gate = np.sin(t * self.freq + self.phase)
            is_turn = (gate > 0.7) # Only top 30% of the wave is "open"
            fired = (charge >= self.theta) & (ref == 0) & is_turn
            state = fired.astype(np.float32) * self._polarity_f32
            charge[fired] = 0.0
            ref[ref > 0] -= 1; ref[fired] = 1
        return state, charge

    def forward(self, world, ticks=TICKS):
        world_vec = np.asarray(world, dtype=np.float32)
        injected = world_vec @ self.input_projection
        self.state, self.charge = self.rollout(injected, ticks)
        return self.readout(self.state) # Spike Readout

    def mutate(self):
        undo = super().mutate()
        if random.random() < 0.1:
            idx = random.randint(0, self.H - 1)
            undo.append(('FREQ', idx, self.freq[idx]))
            self.freq[idx] = np.random.uniform(0.5, 2.0)
        return undo

    def replay(self, undo):
        for entry in reversed(undo):
            if entry[0] == 'FREQ': self.freq[entry[1]] = entry[2]
        super().replay([op for op in undo if op[0] != 'FREQ'])

def evaluate_net(net, data):
    correct = 0
    for i in range(len(data)-1):
        w = np.zeros(VOCAB); w[data[i]] = 1.0
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

def test_battle():
    print("=" * 70)
    print("  BATTLE OF RHYTHMS: Harmonic vs Learned Continuous")
    print("=" * 70)
    TEXT = "music is the shorthand of emotion. brains love patterns and pulses." * 20
    data = [ord(c) % VOCAB for c in TEXT[:500]]
    results = {'A': [], 'B': []}
    for i in range(N_TRIALS):
        seed = 444 + i * 123
        print(f"Trial {i+1}...")
        a = run_trial(HarmonicGraph, seed, data)
        b = run_trial(LearnedFreqGraph, seed, data)
        results['A'].append(a); results['B'].append(b)
        print(f"  A (Harmonic): {a*100:5.2f}% | B (Learned): {b*100:5.2f}%")
    print("-" * 70)
    print(f"MEAN A: {np.mean(results['A'])*100:.2f}%")
    print(f"MEAN B: {np.mean(results['B'])*100:.2f}%")
    delta = (np.mean(results['B']) - np.mean(results['A'])) * 100
    print(f"Delta:  {delta:+.2f}%")

if __name__ == "__main__":
    test_battle()
