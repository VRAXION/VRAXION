
import sys, os, time, random
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.graph import SelfWiringGraph

# Config
VOCAB = 32
HIDDEN_RATIO = 4
TICKS = 12 # Több tick, hogy lássuk a dinamikát
STEPS = 400
REFRACTORY_TICKS = 2

class RecoveryGraph(SelfWiringGraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.refractory_counter = np.zeros(self.H, dtype=np.int8)
        
    def rollout_with_telemetry(self, injected, ticks):
        H = self.H
        state = np.zeros(H, dtype=np.float32)
        charge = np.zeros(H, dtype=np.float32)
        self.refractory_counter.fill(0)
        ret = 1.0 - self.decay
        
        # Telemetry storage
        stats = {
            'firing_rate': [],
            'refractory_rate': [],
            'avg_charge': []
        }
        
        for t in range(ticks):
            # 1. Decay
            charge *= ret
            
            # 2. Input
            if t < 1: charge += injected
            
            # 3. Graph Propagation (Binary + Polarity)
            raw = self._sparse_mul_1d(state)
            charge += raw
            
            # 4. Spike Decision with Recovery Logic
            # Can only fire if NOT in refractory period
            can_fire = (self.refractory_counter == 0)
            fired = (charge > self.theta) & can_fire
            
            # Spike strength: soft-spike
            state = np.where(fired, charge - self.theta, 0.0)
            state *= self._polarity_f32
            
            # 5. Partial Reset & Refractory Start
            charge[fired] -= self.theta[fired]
            self.refractory_counter[fired] = REFRACTORY_TICKS
            
            # 6. Counter Countdown
            self.refractory_counter[self.refractory_counter > 0] -= 1
            
            # 7. Clamp
            charge = np.maximum(charge, 0.0)
            
            # Collect Telemetry
            stats['firing_rate'].append(np.mean(fired))
            stats['refractory_rate'].append(np.mean(self.refractory_counter > 0))
            stats['avg_charge'].append(np.mean(charge))
            
        return state, charge, stats

def run_telemetry_demo():
    print("=" * 70)
    print("  Neuron Recovery & Telemetry Analysis")
    print(f"  Refractory: {REFRACTORY_TICKS} ticks, Reset: Partial (charge -= theta)")
    print("=" * 70)
    
    net = RecoveryGraph(VOCAB, hidden_ratio=HIDDEN_RATIO, seed=42)
    
    # Simulate a single input (token 5)
    world = np.zeros(VOCAB, dtype=np.float32)
    world[5] = 1.0
    injected = world @ net.input_projection
    
    _, _, stats = net.rollout_with_telemetry(injected, TICKS)
    
    print("\n  Tick-by-tick Brain Activity:")
    print(f"  {'Tick':>4} | {'Firing %':>10} | {'Resting %':>12} | {'Avg Charge':>10}")
    print("  " + "-" * 45)
    for i in range(TICKS):
        print(f"  {i+1:4d} | {stats['firing_rate'][i]*100:9.1f}% | {stats['refractory_rate'][i]*100:11.1f}% | {stats['avg_charge'][i]:10.4f}")

    # A/B Test for Accuracy
    print("\n" + "=" * 70)
    print("  Running Quick Accuracy A/B Test...")
    
    def eval_acc(net, data):
        correct = 0
        for i in range(len(data)-1):
            w = np.zeros(VOCAB); w[data[i]] = 1.0
            # Use standard forward but we need to know if it uses recovery
            if isinstance(net, RecoveryGraph):
                injected = w @ net.input_projection
                s, c, _ = net.rollout_with_telemetry(injected, 8)
                logits = net.readout(c)
            else:
                logits = net.forward(w, ticks=8)
            if np.argmax(logits) == data[i+1]: correct += 1
        return correct / (len(data)-1)

    from lib.data import TEXT
    data = [b % VOCAB for b in TEXT.encode('ascii')[:1000]]
    
    # A: Baseline (Mainline with Dale)
    net_a = SelfWiringGraph(VOCAB, hidden_ratio=HIDDEN_RATIO, seed=42)
    acc_a = eval_acc(net_a, data)
    print(f"  A (Mainline Dale): {acc_a*100:.2f}%")
    
    # B: Recovery (Reset + Refractory)
    net_b = RecoveryGraph(VOCAB, hidden_ratio=HIDDEN_RATIO, seed=42)
    acc_b = eval_acc(net_b, data)
    print(f"  B (Recovery):      {acc_b*100:.2f}%")
    
    delta = (acc_b - acc_a) * 100
    print(f"\n  Delta: {delta:+.2f}%")

if __name__ == "__main__":
    run_telemetry_demo()
