
import sys, os, time, random
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.graph import SelfWiringGraph

# Config
VOCAB = 32
HIDDEN_RATIO = 4
TICKS = 12
STEPS = 400
REFRACTORY_TICKS = 1 # Agyszerű rövid pihenő

class DigitalBrainGraph(SelfWiringGraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.refractory_counter = np.zeros(self.H, dtype=np.int8)
        
    def rollout_with_telemetry(self, injected, ticks):
        H = self.H
        state = np.zeros(H, dtype=np.float32)
        charge = np.zeros(H, dtype=np.float32)
        self.refractory_counter.fill(0)
        ret = 1.0 - self.decay
        
        stats = {'firing_rate': [], 'avg_charge': [], 'refractory_rate': []}
        
        for t in range(ticks):
            # 1. Decay
            charge *= ret
            
            # 2. Input (csak az első körben)
            if t < 1: charge += injected
            
            # 3. Graph Propagation (Binary + Polarity)
            raw = self._sparse_mul_1d(state)
            charge += raw
            
            # 4. DIGITAL SPIKE DECISION (Brain-like)
            can_fire = (self.refractory_counter == 0)
            fired = (charge > self.theta) & can_fire
            
            # Binary Spike: Mindent vagy semmit (1.0 vagy 0.0)
            state = fired.astype(np.float32)
            state *= self._polarity_f32 # Dale Law
            
            # 5. HARD RESET & REFRACTORY
            charge[fired] = 0.0
            self.refractory_counter[fired] = REFRACTORY_TICKS
            
            # 6. Countdown
            self.refractory_counter[self.refractory_counter > 0] -= 1
            
            # 7. Clamp (ne legyen negatív kondenzátor)
            charge = np.maximum(charge, 0.0)
            
            stats['firing_rate'].append(np.mean(fired))
            stats['avg_charge'].append(np.mean(charge))
            stats['refractory_rate'].append(np.mean(self.refractory_counter > 0))
            
        return state, charge, stats

def run_brain_telemetry():
    print("=" * 70)
    print("  BRAIN MODEL: Binary Spike + Hard Reset + Refractory")
    print("=" * 70)
    
    net = DigitalBrainGraph(VOCAB, hidden_ratio=HIDDEN_RATIO, seed=42)
    world = np.zeros(VOCAB); world[5] = 1.0
    injected = world @ net.input_projection
    
    _, _, stats = net.rollout_with_telemetry(injected, TICKS)
    
    print("\n  Tick | Firing % | Resting % | Avg Charge")
    print("  " + "-" * 40)
    for i in range(TICKS):
        print(f"  {i+1:4d} | {stats['firing_rate'][i]*100:8.1f}% | {stats['refractory_rate'][i]*100:9.1f}% | {stats['avg_charge'][i]:10.4f}")

    # Accuracy A/B
    from lib.data import TEXT
    data = [b % VOCAB for b in TEXT.encode('ascii')[:1000]]
    
    def eval_acc(net, data):
        correct = 0
        for i in range(len(data)-1):
            w = np.zeros(VOCAB); w[data[i]] = 1.0
            if isinstance(net, DigitalBrainGraph):
                injected = w @ net.input_projection
                _, c, _ = net.rollout_with_telemetry(injected, 8)
                logits = net.readout(c)
            else:
                logits = net.forward(w, ticks=8)
            if np.argmax(logits) == data[i+1]: correct += 1
        return correct / (len(data)-1)

    net_a = SelfWiringGraph(VOCAB, hidden_ratio=HIDDEN_RATIO, seed=42)
    net_b = DigitalBrainGraph(VOCAB, hidden_ratio=HIDDEN_RATIO, seed=42)
    
    print("\n" + "=" * 70)
    print(f"  A (Mainline Soft-Spike): {eval_acc(net_a, data)*100:.2f}%")
    print(f"  B (Brain Digital-Spike): {eval_acc(net_b, data)*100:.2f}%")
    print("=" * 70)

if __name__ == "__main__":
    run_brain_telemetry()
