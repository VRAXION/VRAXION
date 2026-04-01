
import sys, os, time, random
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.graph import SelfWiringGraph

# Config
VOCAB = 32
HIDDEN_RATIO = 4
TICKS = 12
STEPS = 400
MAX_CHARGE = 15 # Int4 limit

class Int4CapacitorGraph(SelfWiringGraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # We'll use float for simulation, but values will be integers 0-15
        self.charge = np.zeros(self.H, dtype=np.float32)
        
    def rollout_with_telemetry(self, injected, ticks):
        H = self.H
        state = np.zeros(H, dtype=np.float32)
        charge = np.zeros(H, dtype=np.float32)
        
        stats = {'firing_rate': [], 'avg_charge': [], 'max_charge_hit': []}
        
        # Scaling injection to fit Int4: let's say a strong input is +5 charge
        scaled_injected = np.clip(injected * 5.0, -10, 10)
        
        for t in range(ticks):
            # 1. FIXED SUBTRACTIVE LEAK (Int4 style)
            charge = np.maximum(charge - 1.0, 0.0)
            
            # 2. INPUT
            if t < 1: charge += scaled_injected
            
            # 3. GRAPH PROPAGATION (+1 / -1 pulses)
            raw = self._sparse_mul_1d(state) # state is 0 or 1, edges are 1
            # Here Dale polarity is already handled by state * polarity in the decision
            charge += raw
            
            # 4. INT4 CLAMPING
            charge = np.clip(charge, 0, MAX_CHARGE)
            
            # 5. SPIKE DECISION (Fire at 15)
            fired = (charge >= MAX_CHARGE)
            
            # Binary Spike with Polarity
            state = fired.astype(np.float32)
            state *= self._polarity_f32 # Dale Law
            
            # 6. RESET FIRED NEURONS
            charge[fired] = 0.0
            
            stats['firing_rate'].append(np.mean(fired))
            stats['avg_charge'].append(np.mean(charge))
            stats['max_charge_hit'].append(np.sum(fired))
            
        return state, charge, stats

def run_int4_telemetry():
    print("=" * 70)
    print(f"  INT4 CAPACITOR MODEL (Max Charge={MAX_CHARGE}, Leak=-1/tick)")
    print("=" * 70)
    
    net = Int4CapacitorGraph(VOCAB, hidden_ratio=HIDDEN_RATIO, seed=42)
    world = np.zeros(VOCAB); world[5] = 1.0
    injected = world @ net.input_projection
    
    _, _, stats = net.rollout_with_telemetry(injected, TICKS)
    
    print("\n  Tick | Firing Rate | Neurons Fired | Avg Charge (0-15)")
    print("  " + "-" * 55)
    for i in range(TICKS):
        print(f"  {i+1:4d} | {stats['firing_rate'][i]*100:10.1f}% | {int(stats['max_charge_hit'][i]):13d} | {stats['avg_charge'][i]:10.2f}")

    # Accuracy check
    from lib.data import TEXT
    data = [b % VOCAB for b in TEXT.encode('ascii')[:1000]]
    
    def eval_acc(net, data):
        correct = 0
        for i in range(len(data)-1):
            w = np.zeros(VOCAB); w[data[i]] = 1.0
            if isinstance(net, Int4CapacitorGraph):
                injected = w @ net.input_projection
                _, c, _ = net.rollout_with_telemetry(injected, 8)
                logits = net.readout(c)
            else:
                logits = net.forward(w, ticks=8)
            if np.argmax(logits) == data[i+1]: correct += 1
        return correct / (len(data)-1)

    net_main = SelfWiringGraph(VOCAB, hidden_ratio=HIDDEN_RATIO, seed=42)
    net_int4 = Int4CapacitorGraph(VOCAB, hidden_ratio=HIDDEN_RATIO, seed=42)
    
    print("\n" + "=" * 70)
    print(f"  Mainline (Soft-Spike Float): {eval_acc(net_main, data)*100:.2f}%")
    print(f"  Int4 Brain (Integer 0-15):  {eval_acc(net_int4, data)*100:.2f}%")
    print("=" * 70)

if __name__ == "__main__":
    run_int4_telemetry()
