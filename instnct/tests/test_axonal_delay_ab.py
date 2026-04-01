
import sys, os, time, random
import numpy as np

sys.path.insert(0, os.getcwd())
from instnct.model.graph import SelfWiringGraph

# Config
VOCAB = 32
HIDDEN_RATIO = 4
TICKS = 12 # Több tick kell a késleltetések kifutásához
MAX_DELAY = 4
STEPS = 1000
N_TRIALS = 2

class AxonalDelayGraph(SelfWiringGraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Delay mask: 0=nincs él, 1-4=él ennyi tick késleltetéssel
        # Inicializáljuk véletlenszerű késleltetésekkel a meglévő éleket
        self.delays = np.zeros((self.H, self.H), dtype=np.int8)
        rows, cols = np.where(self.mask != 0)
        for r, c in zip(rows, cols):
            self.delays[r, c] = random.randint(1, MAX_DELAY)
        self._sync_delay_caches()

    def _sync_delay_caches(self):
        """Precompute sparse caches for EACH delay level."""
        self._delay_caches = {}
        for d in range(1, MAX_DELAY + 1):
            r, c = np.where(self.delays == d)
            if len(r) > 0:
                self._delay_caches[d] = (r.astype(np.intp), c.astype(np.intp))
            else:
                self._delay_caches[d] = (None, None)

    def rollout_delay(self, injected, ticks):
        H = self.H
        # Ring buffer for state history
        history = [np.zeros(H, dtype=np.float32) for _ in range(MAX_DELAY)]
        charge = np.zeros(H, dtype=np.float32)
        ref = np.zeros(H, dtype=np.int8)
        
        for t in range(ticks):
            # 1. Leak
            charge = np.maximum(charge - 1.0, 0.0)
            
            # 2. Input
            if t < 1: charge += injected * 8.0
            
            # 3. Propagate with DELAYS
            # Sum signals from different past time slices
            for d, (rows, cols) in self._delay_caches.items():
                if rows is not None:
                    # Get state from 'd' ticks ago
                    # history[0] is t-1, history[1] is t-2, etc.
                    past_state = history[d-1]
                    np.add.at(charge, cols, past_state[rows])
            
            # 4. Dynamics (C19 Soft-Wave)
            wave = np.sin(t * self.freq + self.phase)
            eff_theta = self.theta * (1.0 + self.rho * wave)
            
            # 5. Decision
            fired = (charge >= eff_theta) & (ref == 0)
            curr_state = fired.astype(np.float32) * self._polarity_f32
            
            # 6. Reset & History Update
            charge[fired] = 0.0
            ref[ref > 0] -= 1; ref[fired] = 1
            
            # Push new state into ring buffer
            history.insert(0, curr_state)
            history.pop() # Keep buffer size constant
            
            np.clip(charge, 0, 15, out=charge)
            
        return curr_state, charge

    def forward(self, world, ticks=TICKS):
        world_vec = np.asarray(world, dtype=np.float32)
        injected = world_vec @ self.input_projection
        self.state, self.charge = self.rollout_delay(injected, ticks)
        return self.readout(self.state)

    def mutate(self):
        # We need a custom mutate to handle edge delays
        undo = super().mutate()
        
        # Chance to mutate an existing edge's delay
        if random.random() < 0.2 and len(self.alive) > 0:
            idx = random.randint(0, len(self.alive)-1)
            r, c = self.alive[idx]
            old_d = self.delays[r, c]
            self.delays[r, c] = random.randint(1, MAX_DELAY)
            undo.append(('DELAY', r, c, old_d))
            
        # Ensure new edges from super().mutate get a delay
        rows, cols = np.where((self.mask != 0) & (self.delays == 0))
        for r, c in zip(rows, cols):
            self.delays[r, c] = random.randint(1, MAX_DELAY)
            # Add to undo so we clean up if rejected
            undo.append(('DELAY_NEW', r, c))

        self._sync_delay_caches()
        return undo

    def replay(self, undo):
        for entry in reversed(undo):
            if entry[0] == 'DELAY':
                self.delays[entry[1], entry[2]] = entry[3]
            elif entry[0] == 'DELAY_NEW':
                self.delays[entry[1], entry[2]] = 0
        super().replay([op for op in undo if op[0] not in ('DELAY', 'DELAY_NEW')])
        self._sync_delay_caches()

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
    for s in range(STEPS):
        undo = net.mutate()
        acc = evaluate_net(net, data)
        if acc >= best_acc: best_acc = acc
        else: net.replay(undo)
        if (s + 1) % 250 == 0:
            print(f"    - Step {s+1}/{STEPS} | best_acc: {best_acc*100:.2f}%")
    return best_acc

def test_axonal_delay():
    print("=" * 70)
    print(f"  AXONAL DELAY BATTLE (1-{MAX_DELAY} Ticks)")
    print(f"  V={VOCAB}, steps={STEPS}, ticks={TICKS}")
    print("=" * 70)
    
    # Text data that needs temporal memory (e.g. word pairs)
    TEXT = "the quick brown fox jumps over the lazy dog. axonal delay is the future." * 20
    data = [ord(c) % VOCAB for c in TEXT[:500]]
    
    results = {'A': [], 'B': []}
    for i in range(N_TRIALS):
        seed = 888 + i * 55
        print(f"Trial {i+1}...")
        a = run_trial(SelfWiringGraph, seed, data)
        b = run_trial(AxonalDelayGraph, seed, data)
        results['A'].append(a); results['B'].append(b)
        print(f"  A (Canon): {a*100:5.2f}% | B (Axonal): {b*100:5.2f}%")
        
    print("-" * 70)
    print(f"MEAN A: {np.mean(results['A'])*100:.2f}%")
    print(f"MEAN B: {np.mean(results['B'])*100:.2f}%")
    delta = (np.mean(results['B']) - np.mean(results['A'])) * 100
    print(f"Delta:  {delta:+.2f}%")

if __name__ == "__main__":
    test_axonal_delay()
