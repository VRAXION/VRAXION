
import sys, os, time, random
import numpy as np

sys.path.insert(0, os.getcwd())
from instnct.model.graph import SelfWiringGraph

# Config
VOCAB = 32
HIDDEN_RATIO = 4
TICKS = 12
STEPS = 1500 # More steps to see convergence
N_TRIALS = 2

class LearnableRhoGraph(SelfWiringGraph):
    """
    Variant: Rho (modulation depth) is learnable per-neuron.
    We initialize it at 0.5 and let evolution move it between 0.0 and 1.0.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Per-neuron Rho
        self.rho = np.full(self.H, 0.5, dtype=np.float32)
        
    def rollout_rho(self, injected, ticks):
        H, state, charge, ref = self.H, np.zeros(self.H), np.zeros(self.H), np.zeros(self.H, dtype=np.int8)
        for t in range(ticks):
            charge = np.maximum(charge - 1.0, 0.0)
            if t < 1: charge += injected * 8.0
            charge += self._sparse_mul_1d(state)
            np.clip(charge, 0, 15, out=charge)
            
            wave = np.sin(t * self.freq + self.phase)
            # Use per-neuron learnable rho
            effective_theta = self.theta * (1.0 + self.rho * wave)
            
            fired = (charge >= effective_theta) & (ref == 0)
            state = fired.astype(np.float32) * self._polarity_f32
            charge[fired] = 0.0
            ref[ref > 0] -= 1; ref[fired] = 1
        return state, charge

    def forward(self, world, ticks=TICKS):
        world_vec = np.asarray(world, dtype=np.float32)
        injected = world_vec @ self.input_projection
        self.state, self.charge = self.rollout_rho(injected, ticks)
        return self.readout(self.state)

    def mutate(self):
        undo = super().mutate()
        # Mutate Rho
        if random.random() < 0.2:
            idx = random.randint(0, self.H - 1)
            undo.append(('RHO', idx, self.rho[idx]))
            self.rho[idx] = np.clip(self.rho[idx] + random.uniform(-0.1, 0.1), 0.0, 1.0)
        return undo

    def replay(self, undo):
        for entry in reversed(undo):
            if entry[0] == 'RHO': self.rho[entry[1]] = entry[2]
        super().replay([op for op in undo if op[0] != 'RHO'])

def evaluate_net(net, data):
    correct = 0
    for i in range(len(data)-1):
        w = np.zeros(VOCAB); w[data[i]] = 1.0
        logits = net.forward(w)
        if np.argmax(logits) == data[i+1]: correct += 1
    return correct / (len(data)-1)

def run_trial(seed, data):
    random.seed(seed); np.random.seed(seed)
    # A: Fixed Rho 0.5 (Canon)
    net_a = SelfWiringGraph(VOCAB, hidden_ratio=HIDDEN_RATIO, seed=seed)
    # B: Learnable Rho
    net_b = LearnableRhoGraph(VOCAB, hidden_ratio=HIDDEN_RATIO, seed=seed)
    
    best_a = evaluate_net(net_a, data)
    best_b = evaluate_net(net_b, data)
    
    for s in range(STEPS):
        u_a = net_a.mutate(); acc_a = evaluate_net(net_a, data)
        if acc_a >= best_a: best_a = acc_a
        else: net_a.replay(u_a)
        
        u_b = net_b.mutate(); acc_b = evaluate_net(net_b, data)
        if acc_b >= best_b: best_b = acc_b
        else: net_b.replay(u_b)
            
    return best_a, best_b, net_b.rho

def test_learnable_rho():
    print("=" * 70)
    print("  A/B Test: Fixed Rho (0.5) vs Learnable Rho [0, 1]")
    print("=" * 70)
    TEXT = "the patterns of the mind are flexible. some neurons wave hard, some stay still." * 20
    data = [ord(c) % VOCAB for c in TEXT[:500]]
    
    for i in range(N_TRIALS):
        print(f"Trial {i+1}...")
        a, b, rhos = run_trial(42+i, data)
        print(f"  A (Fixed): {a*100:5.2f}% | B (Learnable): {b*100:5.2f}%")
        print(f"  Rho Statistics: Mean={rhos.mean():.3f}, Std={rhos.std():.3f}, Min={rhos.min():.3f}, Max={rhos.max():.3f}")

if __name__ == "__main__":
    test_learnable_rho()
