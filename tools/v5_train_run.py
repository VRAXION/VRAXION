import sys, os, time, json
import numpy as np

sys.path.insert(0, os.getcwd())
from instnct.model.graph import SelfWiringGraph
from instnct.lib.utils import score_batch

# Config for v5.0 Run
VOCAB = 64
H_RATIO = 8 # H=512
TICKS = 12
STEPS = 50000
BATCH_SIZE = 128
SAVE_PATH = "instnct/checkpoints/v5_axonal_brain.npz"

def load_data(path, vocab):
    with open(path, 'rb') as f:
        data = f.read()
    valid = [b for b in data if 32 <= b < 127]
    return np.array(valid, dtype=np.uint8) % vocab

def batch_generator(data, batch_size):
    while True:
        idx = np.random.randint(0, len(data) - 2, size=batch_size)
        yield data[idx], data[idx + 1]

def run():
    print("="*60)
    print("  V5.0 MUSICAL AXONAL BRAIN RUN")
    print(f"  H={VOCAB*H_RATIO}, Ticks={TICKS}, Max Delay=4")
    print("="*60)
    
    data = load_data("instnct/data/alpaca_chat.txt", VOCAB)
    gen = batch_generator(data, BATCH_SIZE)
    
    net = SelfWiringGraph(VOCAB, hidden_ratio=H_RATIO, seed=42)
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    
    # Baseline
    x, y = next(gen)
    best_score, best_acc = score_batch(net, y, VOCAB, ticks=TICKS)
    
    t0 = time.time()
    for step in range(1, STEPS + 1):
        undo = net.mutate()
        
        x, y = next(gen)
        # Note: score_batch in utils.py takes (net, targets, V, ticks)
        # But wait, our batch generator gives x, y. y is the target.
        # Let's use a local small eval for speed and stability
        score, acc = score_batch(net, y, VOCAB, ticks=TICKS)
        
        if score >= best_score:
            best_score = score
            best_acc = acc
        else:
            net.replay(undo)
            
        if step % 100 == 0:
            elapsed = time.time() - t0
            edges = len(net.alive)
            # Basic telemetry
            firing_rate = np.mean(net.state != 0)
            
            status = {
                'step': step,
                'acc': round(float(best_acc), 4),
                'score': round(float(best_score), 4),
                'edges': edges,
                'firing': round(float(firing_rate), 4),
                'time': round(elapsed, 1)
            }
            # Print for the user to see live
            print(f"STEP {step:5d} | ACC: {status['acc']*100:5.2f}% | SCORE: {status['score']:.4f} | EDGES: {edges:5d} | FIRING: {status['firing']*100:4.1f}%")
            
            # Save checkpoint
            net.save(SAVE_PATH)
            with open(SAVE_PATH + ".log", 'a') as f:
                f.write(json.dumps(status) + "\n")

if __name__ == "__main__":
    run()
