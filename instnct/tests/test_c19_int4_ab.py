
import sys, os, time, random
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.graph import SelfWiringGraph

# Config
VOCAB = 32
HIDDEN_RATIO = 4
TICKS = 8
STEPS = 1000
N_TRIALS = 3

class C19Int4Graph(SelfWiringGraph):
    """
    C19-style Int4 Brain:
    - Learnable per-neuron decay (0.5 to 3.0)
    - Periodic Wave Modulation: the firing threshold oscillates over ticks.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Per-neuron wave parameters
        # Frequency (how fast it oscillates) and Phase (where it starts)
        self.freq = np.random.uniform(0.5, 1.5, size=self.H).astype(np.float32)
        self.phase = np.random.uniform(0, 2*np.pi, size=self.H).astype(np.float32)
        
    def rollout_c19(self, injected, ticks):
        H = self.H
        state = np.zeros(H, dtype=np.float32)
        charge = np.zeros(H, dtype=np.float32)
        refractory = np.zeros(H, dtype=np.int8)
        
        for t in range(ticks):
            # 1. Learnable Decay
            charge = np.maximum(charge - self.decay, 0.0)
            
            # 2. Input
            if t < 1: charge += injected * 8.0
            
            # 3. Propagate
            raw = self._sparse_mul_1d(state)
            charge += raw
            
            # 4. C19 WAVE MODULATION
            # Threshold oscillates: sometimes it's easy to fire, sometimes hard.
            # wave = 1.0 + 0.5 * sin(...) -> range [0.5, 1.5]
            modulation = 1.0 + 0.5 * np.sin(t * self.freq + self.phase)
            effective_theta = self.theta * modulation
            
            # 5. Spike Decision
            can_fire = (refractory == 0)
            fired = (charge >= effective_theta) & can_fire
            
            state = fired.astype(np.float32) * self._polarity_f32
            
            # 6. Reset & Refractory
            charge[fired] = 0.0
            refractory[refractory > 0] -= 1
            refractory[fired] = 1
            
            # Clamp
            np.clip(charge, 0, 15, out=charge)
            
        return state, charge

    def forward(self, world, ticks=TICKS):
        world_vec = np.asarray(world, dtype=np.float32)
        injected = world_vec @ self.input_projection
        self.state, self.charge = self.rollout_c19(injected, ticks)
        return self.readout(self.charge)

    def mutate(self):
        undo = super().mutate()
        # Mutate Wave params
        if random.random() < 0.1:
            idx = random.randint(0, self.H - 1)
            undo.append(('FREQ', idx, self.freq[idx]))
            self.freq[idx] = np.random.uniform(0.5, 1.5)
        if random.random() < 0.1:
            idx = random.randint(0, self.H - 1)
            undo.append(('PHASE', idx, self.phase[idx]))
            self.phase[idx] = np.random.uniform(0, 2*np.pi)
        return undo

    def replay(self, undo):
        for entry in reversed(undo):
            if entry[0] == 'FREQ': self.freq[entry[1]] = entry[2]
            if entry[0] == 'PHASE': self.phase[entry[1]] = entry[2]
        super().replay([op for op in undo if op[0] not in ('FREQ', 'PHASE')])

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

def test_c19_int4():
    print("=" * 70)
    print("  A/B Test: Int4 Brain vs C19-Wave Int4")
    print(f"  V={VOCAB}, steps={STEPS}, ticks={TICKS}")
    print("=" * 70)
    
    from lib.data import TEXT
    data = [b % VOCAB for b in TEXT.encode('ascii')[:600]]
    
    results = {'A': [], 'B': []}
    for i in range(N_TRIALS):
        seed = 42 + i * 100
        print(f"Trial {i+1}...")
        a = run_trial(SelfWiringGraph, seed, data)
        b = run_trial(C19Int4Graph, seed, data)
        results['A'].append(a); results['B'].append(b)
        print(f"  A (Canon): {a*100:5.2f}% | B (C19): {b*100:5.2f}%")
        
    print("-" * 70)
    print(f"MEAN A: {np.mean(results['A'])*100:.2f}%")
    print(f"MEAN B: {np.mean(results['B'])*100:.2f}%")
    print(f"Delta:  {(np.mean(results['B']) - np.mean(results['A']))*100:+.2f}%")

if __name__ == "__main__":
    test_c19_int4()
