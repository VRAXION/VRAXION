
import sys, os, time, random
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.graph import SelfWiringGraph

# Config
VOCAB = 32
HIDDEN_RATIO = 3
TICKS = 6
STEPS = 600
N_TRIALS = 5

def make_perm_targets(V, seed):
    np.random.seed(seed)
    return np.random.permutation(V)

class DecayFirstGraph(SelfWiringGraph):
    """Override rollout logic to apply decay BEFORE adding new input."""
    
    @staticmethod
    def rollout_token(injected, *, mask, theta, decay, ticks, input_duration=1, 
                      state=None, charge=None, sparse_cache=None, edge_magnitude=1.0, polarity=None):
        mask = np.asarray(mask)
        mask_f32 = mask.astype(np.float32) * np.float32(edge_magnitude) if mask.dtype != np.float32 else mask
        theta = np.asarray(theta, dtype=np.float32)
        decay = np.asarray(decay, dtype=np.float32)
        H = mask.shape[0]
        injected = np.asarray(injected, dtype=np.float32)

        act = np.zeros(H, dtype=np.float32) if state is None else np.asarray(state, dtype=np.float32).copy()
        cur_charge = np.zeros(H, dtype=np.float32) if charge is None else np.asarray(charge, dtype=np.float32).copy()
        ret = 1.0 - decay
        sparse_cache = sparse_cache or SelfWiringGraph.build_sparse_cache(mask)
        use_sparse = len(sparse_cache[0]) < H * H * 0.1

        for tick in range(int(ticks)):
            # 1. DECAY FIRST (Apply leak to the past memory)
            cur_charge *= ret
            
            # 2. COLLECT NEW INPUTS
            if tick < int(input_duration):
                act = act + injected
            raw = (
                SelfWiringGraph._sparse_mul_1d_from_cache(H, act, sparse_cache)
                if use_sparse else act @ mask_f32
            )
            np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 3. ADD NEW INPUTS TO CHARGE
            cur_charge += raw
            
            # 4. THRESHOLD & SPIKE (with Dale polarity)
            act = np.maximum(cur_charge - theta, 0.0)
            if polarity is not None:
                act = act * polarity
            
            # 5. CLAMP
            cur_charge = np.maximum(cur_charge, 0.0)
            
        return act, cur_charge

    @staticmethod
    def rollout_token_batch(injected_batch, *, mask, theta, decay, ticks, input_duration=1, 
                            acts=None, charges=None, sparse_cache=None, edge_magnitude=1.0, polarity=None):
        mask = np.asarray(mask)
        mask_f32 = mask.astype(np.float32) * np.float32(edge_magnitude) if mask.dtype != np.float32 else mask
        theta = np.asarray(theta, dtype=np.float32)
        decay = np.asarray(decay, dtype=np.float32)
        H = mask.shape[0]
        injected_batch = np.asarray(injected_batch, dtype=np.float32)
        batch = injected_batch.shape[0]
        
        cur_acts = np.zeros((batch, H), dtype=np.float32) if acts is None else np.asarray(acts, dtype=np.float32).copy()
        cur_charges = np.zeros((batch, H), dtype=np.float32) if charges is None else np.asarray(charges, dtype=np.float32).copy()
        ret = 1.0 - decay
        sparse_cache = sparse_cache or SelfWiringGraph.build_sparse_cache(mask)
        use_sparse = len(sparse_cache[0]) < H * H * 0.1

        for tick in range(int(ticks)):
            # 1. DECAY FIRST
            cur_charges *= ret
            
            # 2. COLLECT NEW INPUTS
            if tick < int(input_duration):
                cur_acts = cur_acts + injected_batch
            raw = (
                SelfWiringGraph._sparse_mul_2d_from_cache(H, cur_acts, sparse_cache)
                if use_sparse else cur_acts @ mask_f32
            )
            np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 3. ADD NEW INPUTS TO CHARGE
            cur_charges += raw
            
            # 4. THRESHOLD & SPIKE
            cur_acts = np.maximum(cur_charges - theta, 0.0)
            if polarity is not None:
                cur_acts = cur_acts * polarity
            
            # 5. CLAMP
            cur_charges = np.maximum(cur_charges, 0.0)
            
        return cur_acts, cur_charges

    def forward(self, world, ticks=6):
        world_vec = np.asarray(world, dtype=np.float32).copy()
        np.nan_to_num(world_vec, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        injected = world_vec @ self.input_projection
        self.state, self.charge = self.rollout_token(
            injected, mask=self.mask, theta=self.theta, decay=self.decay, ticks=ticks,
            state=self.state, charge=self.charge, sparse_cache=self._sp_cache,
            edge_magnitude=self.edge_magnitude, polarity=self._polarity_f32,
        )
        return self.readout(self.charge)

    def forward_batch(self, ticks=6):
        V = self.V
        projected = np.eye(V, dtype=np.float32) @ self.input_projection
        _, charges = self.rollout_token_batch(
            projected, mask=self.mask, theta=self.theta, decay=self.decay, ticks=ticks,
            sparse_cache=self._sp_cache, edge_magnitude=self.edge_magnitude, polarity=self._polarity_f32,
        )
        return self.readout_batch(charges)

def evaluate_perm(net, targets, V, ticks=TICKS):
    logits = net.forward_batch(ticks=ticks)
    preds = np.argmax(logits, axis=1)
    acc = (preds[:V] == targets[:V]).mean()
    e = np.exp(logits[:V] - logits[:V].max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    tp = probs[np.arange(V), targets[:V]].mean()
    return 0.5 * acc + 0.5 * tp, acc

def run_evo(net_cls, seed, targets, steps=STEPS):
    random.seed(seed); np.random.seed(seed)
    net = net_cls(VOCAB, hidden_ratio=HIDDEN_RATIO, seed=seed)
    best_combined, best_acc = evaluate_perm(net, targets, VOCAB)
    for _ in range(steps):
        undo = net.mutate()
        net.state.fill(0); net.charge.fill(0)
        combined, acc = evaluate_perm(net, targets, VOCAB)
        if combined >= best_combined:
            best_combined = combined
            best_acc = max(best_acc, acc)
        else:
            net.replay(undo)
    return best_acc

def test_decay_order():
    print("=" * 70)
    print("  A/B Test: Decay Order (LIF Dynamics)")
    print("  A = Decay AFTER adding current inputs (Current)")
    print("  B = Decay BEFORE adding current inputs (LIF-style)")
    print(f"  V={VOCAB}, H={VOCAB*HIDDEN_RATIO}, steps={STEPS}, trials={N_TRIALS}")
    print("=" * 70)

    results_a, results_b = [], []
    for trial in range(N_TRIALS):
        seed = 42 + trial * 111
        targets = make_perm_targets(VOCAB, seed)
        
        acc_a = run_evo(SelfWiringGraph, seed + 1000, targets)
        results_a.append(acc_a)
        
        acc_b = run_evo(DecayFirstGraph, seed + 1000, targets)
        results_b.append(acc_b)
        
        print(f"  trial {trial+1}/{N_TRIALS}:  A={acc_a*100:5.1f}%  B={acc_b*100:5.1f}%")

    mean_a, mean_b = np.mean(results_a), np.mean(results_b)
    print(f"\n{'─'*70}")
    print(f"  FINAL MEAN A: {mean_a*100:5.1f}%")
    print(f"  FINAL MEAN B: {mean_b*100:5.1f}%")
    print(f"  Delta:        {(mean_b - mean_a)*100:+5.1f}%")
    print(f"{'─'*70}")

if __name__ == '__main__':
    test_decay_order()
