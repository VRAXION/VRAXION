import sys, os, time, random
import numpy as np

sys.path.insert(0, os.getcwd())
from instnct.model.graph import SelfWiringGraph

VOCAB = 32
HIDDEN_RATIO = 4
TICKS = 8
STEPS = 400

class C19Int4Graph(SelfWiringGraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.freq = np.random.uniform(0.5, 1.5, size=self.H).astype(np.float32)
        self.phase = np.random.uniform(0, 2*np.pi, size=self.H).astype(np.float32)
        
    def rollout_c19(self, injected, ticks):
        H = self.H
        state = np.zeros(H, dtype=np.float32)
        charge = np.zeros(H, dtype=np.float32)
        refractory = np.zeros(H, dtype=np.int8)
        for t in range(ticks):
            charge = np.maximum(charge - self.decay, 0.0)
            if t < 1: charge += injected * 8.0
            raw = self._sparse_mul_1d(state)
            charge += raw
            modulation = 1.0 + 0.5 * np.sin(t * self.freq + self.phase)
            eff_theta = self.theta * modulation
            fired = (charge >= eff_theta) & (refractory == 0)
            state = fired.astype(np.float32) * self._polarity_f32
            charge[fired] = 0.0
            refractory[refractory > 0] -= 1
            refractory[fired] = 1
            np.clip(charge, 0, 15, out=charge)
        return state, charge

    def forward(self, world, ticks=TICKS):
        world_vec = np.asarray(world, dtype=np.float32)
        injected = world_vec @ self.input_projection
        self.state, self.charge = self.rollout_c19(injected, ticks)
        return self.readout(self.charge)

def eval_net(net, data):
    correct = 0
    for i in range(len(data)-1):
        w = np.zeros(VOCAB); w[data[i]] = 1.0
        logits = net.forward(w)
        if np.argmax(logits) == data[i+1]: correct += 1
    return correct / (len(data)-1)

def test():
    # Use simple dummy data to avoid import issues
    TEXT = "the quick brown fox jumps over the lazy dog" * 50
    data = [ord(c) % VOCAB for c in TEXT]
    data = data[:400]
    
    seed = 42
    print("Trial 1 (Quick)...")
    random.seed(seed); np.random.seed(seed)
    net_a = SelfWiringGraph(VOCAB, hidden_ratio=HIDDEN_RATIO, seed=seed)
    best_a = eval_net(net_a, data)
    for _ in range(STEPS):
        undo = net_a.mutate()
        acc = eval_net(net_a, data)
        if acc >= best_a: best_a = acc
        else: net_a.replay(undo)
    
    random.seed(seed); np.random.seed(seed)
    net_b = C19Int4Graph(VOCAB, hidden_ratio=HIDDEN_RATIO, seed=seed)
    best_b = eval_net(net_b, data)
    for _ in range(STEPS):
        undo = net_b.mutate()
        acc = eval_net(net_b, data)
        if acc >= best_b: best_b = acc
        else: net_b.replay(undo)
    
    print(f"A (Canon): {best_a*100:.2f}%")
    print(f"B (C19):   {best_b*100:.2f}%")
    print(f"Delta: {(best_b - best_a)*100:+.2f}%")

if __name__ == "__main__":
    test()
