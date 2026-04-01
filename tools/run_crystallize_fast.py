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
    print("  FAST CRYSTALLIZE: Pruning the Overnight Brain")
    print("="*60)

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint not found at {CHECKPOINT_PATH}")
        return

    # Load data
    data = load_data(DATA_PATH, VOCAB)
    val_data = data[:1000] # Smaller slice for faster eval
    
    # Load Network
    net = SelfWiringGraph.load(CHECKPOINT_PATH)
    initial_edges = len(net.alive)
    
    def evaluate():
        correct = 0
        net.reset()
        for i in range(len(val_data) - 1):
            w = np.zeros(VOCAB, dtype=np.float32)
            w[val_data[i]] = 1.0
            logits = net.forward(w, ticks=TICKS)
            if np.argmax(logits) == val_data[i+1]:
                correct += 1
        return correct / (len(val_data) - 1)

    initial_score = evaluate()
    print(f"Initial accuracy: {initial_score*100:.2f}% | Edges: {initial_edges}")

    # Faster Pruning: Try removing blocks of edges
    alive_snapshot = list(net.alive)
    np.random.shuffle(alive_snapshot)
    
    removed_total = 0
    block_size = 50
    
    print(f"Starting block-pruning (block_size={block_size})...")
    
    for i in range(0, len(alive_snapshot), block_size):
        block = alive_snapshot[i:i+block_size]
        saved_block = []
        for r, c in block:
            saved_block.append((r, c, net.mask[r, c]))
            net.mask[r, c] = 0
        
        net.resync_alive()
        new_score = evaluate()
        
        if new_score >= initial_score - 0.001:
            removed_total += len(block)
            print(f"  Removed {removed_total} edges... (current edges: {len(net.alive)})")
        else:
            # Revert block
            for r, c, val in saved_block:
                net.mask[r, c] = val
            net.resync_alive()

    print("\n" + "="*60)
    print(f"Pruning Complete. Edges: {initial_edges} -> {len(net.alive)} (-{removed_total})")
    print(f"Final Accuracy: {evaluate()*100:.2f}%")
    
    save_path = CHECKPOINT_PATH.replace(".npz", "_pruned.npz")
    net.save(save_path)
    print(f"Pruned model saved to: {save_path}")

if __name__ == "__main__":
    main()
