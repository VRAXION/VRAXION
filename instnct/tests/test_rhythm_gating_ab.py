
import sys, os, time, random
import numpy as np

sys.path.insert(0, os.getcwd())
from instnct.model.graph import SelfWiringGraph

# Config
VOCAB = 32
HIDDEN_RATIO = 4
TICKS = 12 # Több tick, hogy a ritmusok kifuthassanak
STEPS = 600
N_TRIALS = 3

class RhythmGraph(SelfWiringGraph):
    """
    Rhythm Gating: Every neuron has a firing frequency (Period) and a Phase (Offset).
    It can only fire when (tick + phase) % period == 0.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Period: 1 (every tick), 2 (every 2nd), 3, 4
        self.period = np.random.randint(1, 5, size=self.H).astype(np.int8)
        self.phase = np.random.randint(0, 4, size=self.H).astype(np.int8)
        
    def rollout_rhythm(self, injected, ticks):
        H = self.H
        state = np.zeros(H, dtype=np.float32)
        charge = np.zeros(H, dtype=np.float32)
        refractory = np.zeros(H, dtype=np.int8)
        
        firing_log = []
        
        for t in range(ticks):
            # 1. Leak (-1.0)
            charge = np.maximum(charge - 1.0, 0.0)
            
            # 2. Input
            if t < 1: charge += injected * 8.0
            
            # 3. Propagate
            raw = self._sparse_mul_1d(state)
            charge += raw
            
            # 4. RHYTHM GATING
            # A neuron can only fire if it's its "turn" in the rhythm
            is_rhythm_turn = ((t + self.phase) % self.period == 0)
            
            # 5. Decision
            fired = (charge >= self.theta) & (refractory == 0) & is_rhythm_turn
            
            state = fired.astype(np.float32) * self._polarity_f32
            
            # 6. Reset & Refractory
            charge[fired] = 0.0
            refractory[refractory > 0] -= 1
            refractory[fired] = 1
            
            # Clamp
            np.clip(charge, 0, 15, out=charge)
            firing_log.append(np.sum(fired))
            
        return state, charge, np.mean(firing_log)

    def forward(self, world, ticks=TICKS):
        world_vec = np.asarray(world, dtype=np.float32)
        injected = world_vec @ self.input_projection
        self.state, self.charge, _ = self.rollout_rhythm(injected, ticks)
        return self.readout(self.charge)

    def mutate(self):
        undo = super().mutate()
        # Mutate Rhythm
        if random.random() < 0.1:
            idx = random.randint(0, self.H - 1)
            undo.append(('PERIOD', idx, self.period[idx]))
            self.period[idx] = random.randint(1, 4)
        if random.random() < 0.1:
            idx = random.randint(0, self.H - 1)
            undo.append(('PHASE', idx, self.phase[idx]))
            self.phase[idx] = random.randint(0, 3)
        return undo

    def replay(self, undo):
        for entry in reversed(undo):
            if entry[0] == 'PERIOD': self.period[entry[1]] = entry[2]
            if entry[0] == 'PHASE': self.phase[entry[1]] = entry[2]
        super().replay([op for op in undo if op[0] not in ('PERIOD', 'PHASE')])

def evaluate_net(net, data):
    correct = 0
    total_fired = 0
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

def test_rhythm_ab():
    print("=" * 70)
    print("  A/B Test: Int4 Canon vs Rhythm Gating (Musical Brain)")
    print(f"  V={VOCAB}, steps={STEPS}, ticks={TICKS}")
    print("=" * 70)
    
    # Simple data
    TEXT = "the quick brown fox jumps over the lazy dog. a brilliant idea for a brain." * 20
    data = [ord(c) % VOCAB for c in TEXT[:600]]
    
    results = {'A': [], 'B': []}
    for i in range(N_TRIALS):
        seed = 42 + i * 777
        print(f"Trial {i+1}...")
        a = run_trial(SelfWiringGraph, seed, data)
        b = run_trial(RhythmGraph, seed, data)
        results['A'].append(a); results['B'].append(b)
        print(f"  A (Canon): {a*100:5.2f}% | B (Rhythm): {b*100:5.2f}%")
        
    print("-" * 70)
    print(f"MEAN A: {np.mean(results['A'])*100:.2f}%")
    print(f"MEAN B: {np.mean(results['B'])*100:.2f}%")
    delta = (np.mean(results['B']) - np.mean(results['A'])) * 100
    print(f"Delta:  {delta:+.2f}%")

if __name__ == "__main__":
    test_rhythm_ab()
