
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__) + '/..')
from instnct.model.graph import SelfWiringGraph

# Config
VOCAB = 64
H_RATIO = 8
TICKS = 8
CHECKPOINT_PATH = "instnct/checkpoints/overnight_int4_brain.npz"
DATA_PATH = "instnct/data/alpaca_chat.txt"

def load_data(path, vocab):
    with open(path, 'rb') as f:
        data = f.read()
    valid = [b for b in data if 32 <= b < 127]
    return np.array(valid, dtype=np.uint8) % vocab

def main():
    print("="*60)
    print("  CRYSTALLIZE: Pruning the Overnight Brain")
    print("="*60)

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint not found at {CHECKPOINT_PATH}")
        return

    # Load data
    data = load_data(DATA_PATH, VOCAB)
    # Use a fixed validation slice for crystallization to be consistent
    val_data = data[:2000] 
    
    # Load Network
    net = SelfWiringGraph.load(CHECKPOINT_PATH)
    initial_edges = len(net.alive)
    
    def evaluate():
        correct = 0
        net.reset()
        for i in range(len(val_data) - 1):
            w = np.zeros(VOCAB, dtype=np.float32)
            w[val_data[i]] = 1.0
            # Use the Int4 Canon logic manually since we are outside the overnight script
            # Actually, SelfWiringGraph.forward now uses the Int4 Canon logic!
            logits = net.forward(w, ticks=TICKS)
            if np.argmax(logits) == val_data[i+1]:
                correct += 1
        return correct / (len(val_data) - 1)

    print(f"Initial edges: {initial_edges}")
    t0 = time.time()
    initial_score = evaluate()
    print(f"Initial accuracy on validation slice: {initial_score*100:.2f}%")

    print("\nStarting crystallization pass...")
    # The crystallize method in graph.py is systematic and slow.
    # It shuffles edges and tries to remove each one.
    removed = net.crystallize(evaluate, verbose=True)
    
    elapsed = time.time() - t0
    final_edges = len(net.alive)
    final_score = evaluate()

    print("\n" + "="*60)
    print(f"Crystallization Complete in {elapsed:.1f}s")
    print(f"Edges: {initial_edges} -> {final_edges} ({removed} removed, -{removed/initial_edges*100:.1f}%)")
    print(f"Accuracy: {initial_score*100:.2f}% -> {final_score*100:.2f}%")
    
    # Save the lean version
    save_path = CHECKPOINT_PATH.replace(".npz", "_crystallized.npz")
    net.save(save_path)
    print(f"Crystallized model saved to: {save_path}")

if __name__ == "__main__":
    main()
