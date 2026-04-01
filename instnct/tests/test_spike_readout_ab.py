
import sys, os, time, random
import numpy as np

sys.path.insert(0, os.getcwd())
from instnct.model.graph import SelfWiringGraph

# Config
VOCAB = 32
HIDDEN_RATIO = 4
TICKS = 8
STEPS = 1000 # More steps because learning to spike is harder
N_TRIALS = 2

class SpikeReadoutGraph(SelfWiringGraph):
    """Variant that reads out from state (spikes) instead of charge."""
    def forward(self, world, ticks=TICKS):
        world_vec = np.asarray(world, dtype=np.float32)
        injected = world_vec @ self.input_projection
        # We manually call rollout to get the final state
        self.state, self.charge = self.rollout_token(
            injected, mask=self.mask, theta=self.theta, decay=self.decay, 
            ticks=ticks, state=self.state, charge=self.charge,
            sparse_cache=self._sp_cache, edge_magnitude=self.edge_magnitude,
            polarity=self._polarity_f32, refractory=self.refractory
        )
        # READOUT FROM STATE (SPIKES) INSTEAD OF CHARGE
        return self.readout(self.state)

class RhythmSpikeGraph(SelfWiringGraph):
    """Variant with Rhythm Gating AND Spike Readout."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.period = np.random.randint(1, 5, size=self.H).astype(np.int8)
        self.phase = np.random.randint(0, 4, size=self.H).astype(np.int8)

    def rollout_rhythm(self, injected, ticks):
        H = self.H
        state = np.zeros(H, dtype=np.float32)
        charge = np.zeros(H, dtype=np.float32)
        ref = np.zeros(H, dtype=np.int8)
        for t in range(ticks):
            charge = np.maximum(charge - 1.0, 0.0)
            if t < 1: charge += injected * 8.0
            charge += self._sparse_mul_1d(state)
            np.clip(charge, 0, 15, out=charge)
            
            is_rhythm_turn = ((t + self.phase) % self.period == 0)
            fired = (charge >= self.theta) & (ref == 0) & is_rhythm_turn
            
            state = fired.astype(np.float32) * self._polarity_f32
            charge[fired] = 0.0
            ref[ref > 0] -= 1; ref[fired] = 1
        return state, charge

    def forward(self, world, ticks=TICKS):
        world_vec = np.asarray(world, dtype=np.float32)
        injected = world_vec @ self.input_projection
        self.state, self.charge = self.rollout_rhythm(injected, ticks)
        return self.readout(self.state) # Readout from spikes

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

def evaluate_net(net, data):
    correct = 0
    firing_samples = []
    for i in range(len(data)-1):
        w = np.zeros(VOCAB); w[data[i]] = 1.0
        logits = net.forward(w)
        if np.argmax(logits) == data[i+1]: correct += 1
        firing_samples.append(np.mean(net.state != 0))
    return correct / (len(data)-1), np.mean(firing_samples)

def run_trial(net_cls, seed, data):
    random.seed(seed); np.random.seed(seed)
    net = net_cls(VOCAB, hidden_ratio=HIDDEN_RATIO, seed=seed)
    best_acc, _ = evaluate_net(net, data)
    for _ in range(STEPS):
        undo = net.mutate()
        acc, _ = evaluate_net(net, data)
        if acc >= best_acc: best_acc = acc
        else: net.replay(undo)
    return best_acc, net

def test_spike_readout():
    print("=" * 80)
    print("  A/B/C Test: The Readout Revolution")
    print("  A = Charge Readout (Current Canon - 'The Leak')")
    print("  B = Spike Readout (Pure Spiking - 'The Truth')")
    print("  C = Rhythm + Spike Readout (Musical Brain)")
    print(f"  V={VOCAB}, steps={STEPS}, ticks={TICKS}")
    print("=" * 80)
    
    TEXT = "the quick brown fox jumps over the lazy dog. can we learn to fire patterns?" * 20
    data = [ord(c) % VOCAB for c in TEXT[:500]]
    
    for i in range(N_TRIALS):
        seed = 123 + i * 99
        print(f"Trial {i+1}...")
        
        acc_a, _ = run_trial(SelfWiringGraph, seed, data)
        acc_b, _ = run_trial(SpikeReadoutGraph, seed, data)
        acc_c, net_c = run_trial(RhythmSpikeGraph, seed, data)
        
        _, f_rate_c = evaluate_net(net_c, data)
        
        print(f"    A (Charge): {acc_a*100:5.2f}%")
        print(f"    B (Spike):  {acc_b*100:5.2f}%")
        print(f"    C (Rhythm): {acc_c*100:5.2f}% (Firing: {f_rate_c*100:4.1f}%)")

if __name__ == "__main__":
    test_spike_readout()
