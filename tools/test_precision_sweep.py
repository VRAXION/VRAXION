
import sys, os, time, random
import numpy as np

sys.path.insert(0, os.getcwd())
from instnct.model.graph import SelfWiringGraph

# Config
VOCAB = 8
H = 128
TICKS = 8
# A nehéz "Légy-agy" setup
INHIB_RATE = 0.40
RECIP_RATE = 0.50

def build_fly_mask(N, inhib_rate, recip_rate, precision_mode='binary'):
    mask = np.zeros((N, N), dtype=np.float32)
    # 1. Fill random edges
    density = 0.05
    for r in range(N):
        for c in range(N):
            if r != c and random.random() < density:
                # Initial weight based on mode
                if precision_mode == 'binary': val = 1.0
                elif precision_mode == 'low_int': val = float(random.randint(1, 3))
                elif precision_mode == 'mid_int': val = float(random.randint(1, 15))
                else: val = random.uniform(0.1, 2.0)
                mask[r, c] = val

    # 2. Force Reciprocal pairs
    rows, cols = np.where(mask > 0)
    for r, c in zip(rows, cols):
        if random.random() < recip_rate:
            if precision_mode == 'binary': val = 1.0
            elif precision_mode == 'low_int': val = float(random.randint(1, 3))
            elif precision_mode == 'mid_int': val = float(random.randint(1, 15))
            else: val = random.uniform(0.1, 2.0)
            mask[c, r] = val
            
    # 3. Assign Polarity
    polarity = np.ones(N, dtype=np.float32)
    inhib_indices = random.sample(range(N), int(N * inhib_rate))
    for idx in inhib_indices:
        polarity[idx] = -1.0
        
    return mask, polarity

def test_config(precision_mode):
    # Distinct separation test: 8 inputs mapped to 8 target neurons
    targets = np.arange(VOCAB)
    successes = 0
    
    # 3 trials per precision mode
    for _ in range(3):
        mask, polarity = build_fly_mask(H, INHIB_RATE, RECIP_RATE, precision_mode)
        # Simulation
        # We'll use a simplified forward pass to test 'separation'
        active_counts = []
        correct_targets = 0
        
        for i in range(VOCAB):
            # Input projection (sensory)
            charge = np.zeros(H)
            charge[i] = 10.0 # sensory pulse
            state = np.zeros(H)
            
            for t in range(TICKS):
                charge = np.maximum(charge - 1.0, 0.0) # Leak
                # Propagate: signal = (state * polarity) @ mask
                charge += (state * polarity) @ mask
                np.clip(charge, 0, 15, out=charge)
                fired = (charge >= 10.0) # threshold
                state = fired.astype(np.float32)
                charge[fired] = 0.0
            
            # If the i-th neuron is active at the end, it's a win
            if state[i] > 0:
                correct_targets += 1
            active_counts.append(np.sum(state))
            
        if correct_targets >= VOCAB * 0.7: # 70%+ success
            successes += 1
            
    return successes / 3.0, np.mean(active_counts)

def run_sweep():
    print("=" * 75)
    print(f"  PRECISION SWEEP: Can we survive Fly-Brain config? (40% Inhib, 50% Recip)")
    print("=" * 75)
    print(f"  Mode         | Success Rate | Avg Activity | Verdict")
    print("-" * 75)
    
    modes = ['binary', 'low_int', 'mid_int', 'float']
    for mode in modes:
        rate, activity = test_config(mode)
        verdict = "DEAD" if rate == 0 else "ALIVE" if rate < 0.8 else "STABLE"
        print(f"  {mode:12s} | {rate*100:11.1f}% | {activity:12.2f} | {verdict}")

if __name__ == "__main__":
    run_sweep()
