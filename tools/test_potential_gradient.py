import sys, os, time, random
import numpy as np

sys.path.insert(0, os.getcwd())
from instnct.model.graph import SelfWiringGraph

VOCAB = 32
HIDDEN_RATIO = 4
TICKS = 8
STEPS = 1200 
N_TRIALS = 2

def evaluate_potential(net, data):
    correct = 0
    total_potential = 0
    for i in range(len(data)-1):
        w = np.zeros(VOCAB); w[data[i]] = 1.0
        logits = net.forward(w)
        if np.argmax(logits) == data[i+1]: correct += 1
        total_potential += logits[data[i+1]]
    acc = correct / (len(data)-1)
    return acc, acc + 0.05 * (total_potential / (len(data)-1))

def run_trial(seed, data, use_potential=False):
    random.seed(seed); np.random.seed(seed)
    net = SelfWiringGraph(VOCAB, hidden_ratio=HIDDEN_RATIO, seed=seed)
    acc, score = evaluate_potential(net, data)
    best_score = score if use_potential else acc
    best_acc = acc
    for _ in range(STEPS):
        undo = net.mutate()
        acc, score = evaluate_potential(net, data)
        curr = score if use_potential else acc
        if curr >= best_score:
            best_score = curr
            best_acc = acc
        else:
            net.replay(undo)
    return best_acc

def test():
    TEXT = "knowledge is a gradient. small steps lead to big loops." * 20
    data = [ord(c) % VOCAB for c in TEXT[:500]]
    for i in range(N_TRIALS):
        seed = 777 + i * 111
        print(f"Trial {i+1}...")
        a = run_trial(seed, data, use_potential=False)
        b = run_trial(seed, data, use_potential=True)
        print(f"  A (Pure Acc): {a*100:5.2f}% | B (Potential): {b*100:5.2f}%")
        print(f"  Delta: {(b-a)*100:+.2f}%")

if __name__ == "__main__":
    test()
