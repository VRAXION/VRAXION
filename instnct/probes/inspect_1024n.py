"""Inspect inner structure of the 1024n network."""
import sys, os, numpy as np, glob

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
from graph import SelfWiringGraph

IO = 256; H = IO * 4
SelfWiringGraph.NV_RATIO = 4
np.random.seed(42)
net = SelfWiringGraph(IO)
input_projection = net.input_projection; output_projection = net.output_projection

CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "english_1024n_step*.npz")),
               key=lambda x: int(x.split("step")[1].split(".")[0]))
ckpt = ckpts[-1]
print(f"Loading: {ckpt}")
d = np.load(ckpt)
rows = d['rows']; cols = d['cols']; vals = d['vals']
theta = d['theta']; decay = d['decay']
mask = np.zeros((H, H), dtype=np.float32)
mask[rows, cols] = vals
n_edges = len(rows)

print(f"\n{'='*70}")
print(f"NETWORK STRUCTURE: {H} neurons, {n_edges} edges")
print(f"{'='*70}")

# Edge polarity
pos = np.sum(vals > 0)
neg = np.sum(vals < 0)
print(f"\nEdge polarity: +{pos} excitatory, -{neg} inhibitory ({pos/(pos+neg)*100:.0f}%/{ neg/(pos+neg)*100:.0f}%)")

# Connectivity per neuron
in_degree = np.zeros(H, dtype=int)
out_degree = np.zeros(H, dtype=int)
for r, c in zip(rows, cols):
    out_degree[r] += 1
    in_degree[c] += 1

print(f"\n--- IN-DEGREE (how many inputs a neuron receives) ---")
print(f"  Mean: {in_degree.mean():.1f}, Max: {in_degree.max()}, Min: {in_degree.min()}")
print(f"  Neurons with 0 inputs: {np.sum(in_degree == 0)} ({np.sum(in_degree == 0)/H*100:.0f}%)")
print(f"  Top 10 most connected (in): {np.argsort(in_degree)[-10:][::-1]} = {np.sort(in_degree)[-10:][::-1]}")

print(f"\n--- OUT-DEGREE (how many outputs a neuron sends) ---")
print(f"  Mean: {out_degree.mean():.1f}, Max: {out_degree.max()}, Min: {out_degree.min()}")
print(f"  Neurons with 0 outputs: {np.sum(out_degree == 0)} ({np.sum(out_degree == 0)/H*100:.0f}%)")
print(f"  Top 10 most connected (out): {np.argsort(out_degree)[-10:][::-1]} = {np.sort(out_degree)[-10:][::-1]}")

# Hub neurons (high in AND out)
total_degree = in_degree + out_degree
hubs = np.argsort(total_degree)[-20:][::-1]
print(f"\n--- TOP 20 HUB NEURONS (in+out) ---")
print(f"  {'Neuron':>8} {'In':>5} {'Out':>5} {'Total':>6} {'Theta':>8} {'Decay':>8}")
for n in hubs:
    print(f"  {n:>8} {in_degree[n]:>5} {out_degree[n]:>5} {total_degree[n]:>6} {theta[n]:>8.3f} {decay[n]:>8.3f}")

# Isolated neurons
isolated = np.sum(total_degree == 0)
print(f"\n  Isolated neurons (0 connections): {isolated}/{H} ({isolated/H*100:.0f}%)")

# Theta distribution
print(f"\n--- THETA DISTRIBUTION ---")
bins = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.0]
for i in range(len(bins)-1):
    count = np.sum((theta >= bins[i]) & (theta < bins[i+1]))
    bar = '#' * (count // 5)
    print(f"  [{bins[i]:.2f}-{bins[i+1]:.2f}): {count:>4} {bar}")
print(f"  Mean={theta.mean():.3f}, Std={theta.std():.3f}, Min={theta.min():.3f}, Max={theta.max():.3f}")

# Theta vs connectivity
active = total_degree > 0
print(f"\n  Connected neurons theta: mean={theta[active].mean():.3f} std={theta[active].std():.3f}")
print(f"  Isolated neurons theta:  mean={theta[~active].mean():.3f} std={theta[~active].std():.3f}")

# Decay distribution
print(f"\n--- DECAY DISTRIBUTION ---")
bins_d = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
for i in range(len(bins_d)-1):
    count = np.sum((decay >= bins_d[i]) & (decay < bins_d[i+1]))
    bar = '#' * (count // 5)
    print(f"  [{bins_d[i]:.2f}-{bins_d[i+1]:.2f}): {count:>4} {bar}")
print(f"  Mean={decay.mean():.3f}, Std={decay.std():.3f}, Min={decay.min():.3f}, Max={decay.max():.3f}")

# Decay vs connectivity
print(f"\n  Connected neurons decay: mean={decay[active].mean():.3f} std={decay[active].std():.3f}")
print(f"  Isolated neurons decay:  mean={decay[~active].mean():.3f} std={decay[~active].std():.3f}")

# Cluster detection: which neurons connect to each other?
print(f"\n--- SELF-LOOPS & RECIPROCAL CONNECTIONS ---")
reciprocal = 0
for r, c in zip(rows, cols):
    if mask[c, r] != 0:
        reciprocal += 1
print(f"  Reciprocal pairs (A->B AND B->A): {reciprocal//2}")

# input_projection sensitivity: which neurons get the strongest input signal?
input_projection_norm = np.linalg.norm(input_projection, axis=0)  # per neuron input strength
print(f"\n--- INPUT SENSITIVITY (input_projection norm per neuron) ---")
print(f"  Mean={input_projection_norm.mean():.3f}, Std={input_projection_norm.std():.3f}")
top_input = np.argsort(input_projection_norm)[-10:][::-1]
print(f"  Top 10 input-sensitive neurons: {top_input}")
print(f"  Are they hubs? Overlap with top20 hubs: {len(set(top_input) & set(hubs))}/10")

# Edge weight distribution around hubs
print(f"\n--- HUB EDGE POLARITY (top 5 hubs) ---")
for n in hubs[:5]:
    out_edges = vals[rows == n]
    in_edges = vals[cols == n]
    out_p = np.sum(out_edges > 0); out_n = np.sum(out_edges < 0)
    in_p = np.sum(in_edges > 0); in_n = np.sum(in_edges < 0)
    print(f"  Neuron {n}: OUT +{out_p}/-{out_n} | IN +{in_p}/-{in_n} | theta={theta[n]:.3f} decay={decay[n]:.3f}")

# Sparsity map (16x16 blocks)
print(f"\n--- BLOCK DENSITY MAP (64x64 neuron blocks) ---")
BS = 64
blocks = H // BS
print(f"  {blocks}x{blocks} blocks, each {BS}x{BS} neurons")
density_map = np.zeros((blocks, blocks))
for bi in range(blocks):
    for bj in range(blocks):
        block = mask[bi*BS:(bi+1)*BS, bj*BS:(bj+1)*BS]
        density_map[bi, bj] = np.count_nonzero(block)

# Print as heatmap
print("     " + "".join([f"{j:>4}" for j in range(blocks)]))
for i in range(blocks):
    row = "".join([f"{int(density_map[i,j]):>4}" for j in range(blocks)])
    print(f"  {i:>2}: {row}")
total_per_block = density_map.sum() / (blocks * blocks)
print(f"  Avg edges per block: {total_per_block:.1f}")
print(f"  Empty blocks: {np.sum(density_map == 0)}/{blocks*blocks}")
