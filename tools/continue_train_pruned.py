import sys, os, time, json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__) + '/..')
from instnct.model.graph import SelfWiringGraph

# Config
VOCAB = 64
TICKS = 8
STEPS = 30000 
BATCH_SIZE = 128
LOAD_PATH = "instnct/checkpoints/overnight_int4_brain_pruned.npz"
SAVE_PATH = "instnct/checkpoints/post_prune_brain.npz"

def load_data(path, vocab):
    with open(path, 'rb') as f:
        data = f.read()
    valid = [b for b in data if 32 <= b < 127]
    return np.array(valid, dtype=np.uint8) % vocab

def batch_generator(data, batch_size):
    while True:
        idx = np.random.randint(0, len(data) - 2, size=batch_size)
        yield data[idx], data[idx + 1]

def eval_batch(net, x, y):
    B = x.shape[0]
    w = np.zeros((B, net.V), dtype=np.float32)
    w[np.arange(B), x] = 1.0
    injected = w @ net.input_projection
    acts, charges = net.rollout_token_batch(injected, mask=net.mask, theta=net.theta, decay=net.decay, 
                                            ticks=TICKS, polarity=net._polarity_f32)
    logits = net.readout_batch(charges)
    preds = np.argmax(logits, axis=1)
    return np.mean(preds == y), np.mean(acts != 0)

def run():
    print("="*60)
    print("  CONTINUING FROM PRUNED STATE")
    print(f"  Target: {STEPS} steps, Loading: {LOAD_PATH}")
    print("="*60)
    
    data_path = "instnct/data/alpaca_chat.txt"
    data = load_data(data_path, VOCAB)
    gen = batch_generator(data, BATCH_SIZE)
    
    net = SelfWiringGraph.load(LOAD_PATH)
    
    # Baseline
    x, y = next(gen)
    best_acc, _ = eval_batch(net, x, y)
    initial_acc = best_acc
    
    t0 = time.time()
    for step in range(STEPS):
        undo = net.mutate()
        x, y = next(gen)
        acc, f_rate = eval_batch(net, x, y)
        
        if acc >= best_acc - 0.005:
            best_acc = 0.95 * best_acc + 0.05 * acc
        else:
            net.replay(undo)
            
        if step % 500 == 0:
            elapsed = time.time() - t0
            print(f"Step {step:5d} | Acc: {best_acc*100:5.2f}% | Edges: {len(net.alive):5d} | Firing: {f_rate*100:4.1f}%")
            
    print(f"\nFinal Accuracy change: {initial_acc*100:.2f}% -> {best_acc*100:.2f}%")
    net.save(SAVE_PATH)

if __name__ == "__main__":
    run()
