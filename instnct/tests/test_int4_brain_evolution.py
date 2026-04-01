
import sys, os, time, random
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.graph import SelfWiringGraph

# Config
VOCAB = 32
HIDDEN_RATIO = 4
TICKS = 8
STEPS = 300 # Rövid tanítás
MAX_CHARGE = 15

class Int4EvolvingBrain(SelfWiringGraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Integer thresholds between 1 and 15
        self.theta = np.random.randint(1, MAX_CHARGE + 1, size=self.H).astype(np.float32)
        
    def rollout_int4(self, injected, ticks):
        H = self.H
        state = np.zeros(H, dtype=np.float32)
        charge = np.zeros(H, dtype=np.float32)
        # Scaled injection to act as "sensory input"
        scaled_injected = np.clip(injected * 8.0, -10, 10)
        
        firing_counts = []
        
        for t in range(ticks):
            # 1. Leak
            charge = np.maximum(charge - 1.0, 0.0)
            # 2. Input
            if t < 1: charge += scaled_injected
            # 3. Propagate (Binary edges + Dale Polarity)
            raw = self._sparse_mul_1d(state)
            charge += raw
            # 4. Clamp
            charge = np.clip(charge, 0, MAX_CHARGE)
            # 5. Spike
            fired = (charge >= self.theta)
            state = fired.astype(np.float32) * self._polarity_f32
            # 6. Reset
            charge[fired] = 0.0
            
            firing_counts.append(np.sum(fired))
            
        return state, charge, np.mean(firing_counts)

    def mutate(self):
        undo = super().mutate()
        # Custom integer theta mutation
        if random.random() < 0.2:
            idx = random.randint(0, self.H - 1)
            old_theta = self.theta[idx]
            self.theta[idx] = np.clip(self.theta[idx] + random.choice([-1, 1]), 1, MAX_CHARGE)
            undo.append(('T_INT', idx, old_theta))
        return undo

    def replay(self, undo):
        t_ops = [op for op in undo if op[0] == 'T_INT']
        for _, idx, old in t_ops:
            self.theta[idx] = old
        super().replay([op for op in undo if op[0] != 'T_INT'])

def eval_net(net, data):
    correct = 0
    total_fired = 0
    for i in range(len(data)-1):
        w = np.zeros(VOCAB); w[data[i]] = 1.0
        injected = w @ net.input_projection
        s, c, avg_fired = net.rollout_int4(injected, TICKS)
        logits = net.readout(c)
        if np.argmax(logits) == data[i+1]: correct += 1
        total_fired += avg_fired
    return correct / (len(data)-1), total_fired / (len(data)-1)

def run_evolution_test():
    print("=" * 70)
    print(f"  INT4 BRAIN EVOLUTION (Max Charge={MAX_CHARGE})")
    print("=" * 70)
    
    from lib.data import TEXT
    data = [b % VOCAB for b in TEXT.encode('ascii')[:500]]
    
    net = Int4EvolvingBrain(VOCAB, hidden_ratio=HIDDEN_RATIO, seed=42)
    
    best_acc, _ = eval_net(net, data)
    print(f"  Initial Accuracy: {best_acc*100:.2f}%")
    
    print("\n  Training...")
    for step in range(STEPS):
        undo = net.mutate()
        acc, fired = eval_net(net, data)
        if acc >= best_acc:
            best_acc = acc
        else:
            net.replay(undo)
            
        if step % 50 == 0:
            print(f"    Step {step:3d} | Best Acc: {best_acc*100:5.2f}% | Avg Fired: {fired:5.1f} neurons")

    final_acc, final_fired = eval_net(net, data)
    print("-" * 70)
    print(f"  FINAL ACCURACY:  {final_acc*100:.2f}%")
    print(f"  AVG FIRING RATE: {final_fired / net.H * 100:.1f}% of neurons")
    print("=" * 70)

if __name__ == "__main__":
    run_evolution_test()
