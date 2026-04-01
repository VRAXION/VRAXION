
import sys, os, time, random
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.graph import SelfWiringGraph
from lib.data import TEXT

# Config - matching the previous high-accuracy run
VOCAB = 32
H_RATIO = 4
TICKS = 8
STEPS = 500
SEEDS = [42, 777, 123]

def make_data(vocab):
    raw = [b for b in TEXT.encode('ascii') if b < vocab + 32]
    return np.array(raw, dtype=np.uint8) % vocab

class SoftDaleGraph(SelfWiringGraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Claude's secret sauce: ~20% inhibitory
        self.neuron_sign = np.ones(self.H, dtype=np.float32)
        rng = np.random.default_rng(kwargs.get('seed', 42))
        self.neuron_sign[rng.random(self.H) < 0.20] = -1.0
        self._sync_dale_cache()

    def _sync_dale_cache(self):
        rows, cols = np.where(self.mask != 0)
        if len(rows) == 0:
            self._dale_cache = (np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp),
                                np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp))
            return
            
        # Dale's law: sign is determined by the SOURCE neuron
        signs = self.neuron_sign[rows]
        pos = signs > 0
        neg = signs < 0
        self._dale_cache = (
            rows[pos].astype(np.intp), cols[pos].astype(np.intp),
            rows[neg].astype(np.intp), cols[neg].astype(np.intp)
        )

    def rollout_dale(self, injected, ticks):
        H = self.H
        state = np.zeros(H, dtype=np.float32)
        charge = np.zeros(H, dtype=np.float32)
        ret = 1.0 - self.decay
        pr, pc, nr, nc = self._dale_cache
        
        for t in range(ticks):
            if t < 1:
                state = state + injected
            
            raw = np.zeros(H, dtype=np.float32)
            # Excitatory neurons ADD to charge
            if len(pr): np.add.at(raw, pc, state[pr])
            # Inhibitory neurons SUBTRACT from charge
            if len(nr): np.subtract.at(raw, nc, state[nr])
            
            charge += raw
            charge *= ret
            
            # Continuous activation (ReLU-style), NO hard reset
            state = np.maximum(charge - self.theta, 0.0)
            charge = np.maximum(charge, 0.0)
            
        return state, charge

    def forward(self, world, ticks=TICKS):
        world_vec = np.asarray(world, dtype=np.float32)
        injected = world_vec @ self.input_projection
        self.state, self.charge = self.rollout_dale(injected, ticks)
        return self.readout(self.charge)

    def mutate(self):
        undo = super().mutate()
        # Occasional sign flip
        if random.random() < 0.05:
            idx = random.randint(0, self.H - 1)
            old_sign = self.neuron_sign[idx]
            self.neuron_sign[idx] *= -1.0
            undo.append(('SIGN', idx, old_sign))
        
        self._sync_dale_cache()
        return undo

    def replay(self, undo):
        sign_ops = [op for op in undo if op[0] == 'SIGN']
        for _, idx, old in sign_ops:
            self.neuron_sign[idx] = old
        super().replay([op for op in undo if op[0] != 'SIGN'])
        self._sync_dale_cache()

def eval_net(net, data):
    correct = 0; total = 0
    net.state.fill(0); net.charge.fill(0)
    for i in range(len(data) - 1):
        world = np.zeros(VOCAB, dtype=np.float32)
        world[data[i]] = 1.0
        logits = net.forward(world)
        if np.argmax(logits) == data[i+1]:
            correct += 1
        total += 1
    return correct / total

def run_experiment(mode, seed):
    random.seed(seed); np.random.seed(seed)
    data = make_data(VOCAB)
    if mode == 'A':
        net = SelfWiringGraph(VOCAB, hidden_ratio=H_RATIO, seed=seed)
    else:
        net = SoftDaleGraph(VOCAB, hidden_ratio=H_RATIO, seed=seed)
    best_acc = eval_net(net, data)
    for _ in range(STEPS):
        undo = net.mutate()
        acc = eval_net(net, data)
        if acc >= best_acc: best_acc = acc
        else: net.replay(undo)
    return best_acc

if __name__ == "__main__":
    print(f"Soft-Dale E/I A/B Test (H_ratio={H_RATIO}, steps={STEPS})")
    print("-" * 65)
    results = {'A': [], 'B': []}
    for seed in SEEDS:
        print(f"Seed {seed}:")
        for mode in ['A', 'B']:
            t0 = time.time()
            acc = run_experiment(mode, seed)
            elapsed = time.time() - t0
            results[mode].append(acc)
            label = "Mainline (ReLU)" if mode == 'A' else "Soft-Dale (E/I)"
            print(f"  {label:18s}: {acc*100:5.2f}% ({elapsed:4.1f}s)")
            
    print("-" * 65)
    print(f"FINAL MEAN A (Baseline): {np.mean(results['A'])*100:.2f}%")
    print(f"FINAL MEAN B (Soft-Dale): {np.mean(results['B'])*100:.2f}%")
    delta = (np.mean(results['B']) - np.mean(results['A'])) * 100
    print(f"Delta: {delta:+.2f}%")
