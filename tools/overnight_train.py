import sys, os, time, json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__) + '/..')
from instnct.model.graph import SelfWiringGraph

# Config for Overnight Run
VOCAB = 64 # Nagyobb szótár, több feladat
H_RATIO = 8 # Brutálisan nagy hálózat (H=512) az éjszakára
TICKS = 8
STEPS = 200000  # Massive run (kb. 6-8 óra a Decken)
BATCH_SIZE = 128
SAVE_PATH = "instnct/checkpoints/overnight_int4_brain.npz"

def load_data(path, vocab):
    with open(path, 'rb') as f:
        data = f.read()
    # Filter to printable ASCII and map to vocab
    valid = [b for b in data if 32 <= b < 127]
    return np.array(valid, dtype=np.uint8) % vocab

def batch_generator(data, batch_size, seq_len=1):
    while True:
        idx = np.random.randint(0, len(data) - seq_len - 1, size=batch_size)
        yield data[idx], data[idx + 1]

def eval_batch(net, x, y):
    # One-hot encode inputs
    B = x.shape[0]
    w = np.zeros((B, net.V), dtype=np.float32)
    w[np.arange(B), x] = 1.0
    
    injected = w @ net.input_projection
    
    # Forward pass (Int4 Canon)
    acts, charges = net.rollout_token_batch(injected, 
                                            mask=net.mask,
                                            theta=net.theta,
                                            decay=net.decay,
                                            ticks=TICKS,
                                            polarity=net._polarity_f32,
                                            refractory=None) # Batch mode doesn't track refractory cleanly yet
                                            
    logits = net.readout_batch(charges)
    preds = np.argmax(logits, axis=1)
    acc = np.mean(preds == y)
    
    # Collect basic telemetry
    avg_firing_rate = np.mean(acts != 0)
    avg_charge = np.mean(charges)
    
    return acc, avg_firing_rate, avg_charge

def run():
    print("="*60)
    print("  OVERNIGHT TRAINING RUN (Int4 Brain)")
    print(f"  Steps: {STEPS}, Batch: {BATCH_SIZE}, V: {VOCAB}, H: {VOCAB*H_RATIO}")
    print("="*60)
    
    # Load dataset
    data_path = os.path.join(os.path.dirname(__file__), '../instnct/data/alpaca_chat.txt')
    data = load_data(data_path, VOCAB)
    print(f"Dataset loaded: {len(data)} valid tokens.")
    
    gen = batch_generator(data, BATCH_SIZE)
    
    # Init Network
    net = SelfWiringGraph(VOCAB, hidden_ratio=H_RATIO, seed=42)
    save_dir = os.path.join(os.path.dirname(__file__), '../instnct/checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    full_save_path = os.path.join(save_dir, 'overnight_int4_brain.npz')
    
    # Init Telemetry
    telemetry = []
    
    # Baseline eval
    x, y = next(gen)
    best_acc, _, _ = eval_batch(net, x, y)
    
    t0 = time.time()
    for step in range(STEPS):
        undo = net.mutate()
        
        # Evaluate on a new batch
        x, y = next(gen)
        acc, f_rate, c_avg = eval_batch(net, x, y)
        
        # Exponential moving average for stability over batches
        if acc >= best_acc - 0.01: # Small tolerance to prevent getting stuck
            best_acc = 0.9 * best_acc + 0.1 * acc
        else:
            net.replay(undo)
            
        # Log every 1000 steps
        if step % 1000 == 0:
            elapsed = time.time() - t0
            edges = len(net.alive)
            entry = {
                'step': step,
                'time': round(elapsed, 1),
                'best_acc': round(float(best_acc), 4),
                'edges': edges,
                'firing_rate': round(float(f_rate), 4),
                'avg_charge': round(float(c_avg), 4),
                'theta_mean': round(float(net.theta.mean()), 2)
            }
            telemetry.append(entry)
            print(f"Step {step:6d} | Acc: {best_acc*100:5.2f}% | Edges: {edges:5d} | Firing: {f_rate*100:5.1f}% | Theta: {entry['theta_mean']:.1f}")
            
            # Save checkpoint and telemetry
            net.save(full_save_path)
            with open(full_save_path + ".json", 'w') as f:
                json.dump(telemetry, f, indent=2)
                
    print("Training Complete! Saved to", full_save_path)

if __name__ == "__main__":
    run()