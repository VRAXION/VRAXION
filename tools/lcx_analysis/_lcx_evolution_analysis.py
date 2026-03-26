"""LCX Space Evolution Analysis — correlate structure with per-task accuracy."""
import sys, torch, random, json, numpy as np
sys.path.insert(0, r"S:\AI\work\VRAXION_DEV\Diamond Code")
from swarm_model import SwarmByteRingModel as SwarmMoE
from test_swarm_config import generate_multitask_batch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_DIR = r"S:\AI\work\VRAXION_DEV\Diamond Code\checkpoints\swarm"
N_SAMPLES = 500  # per task per checkpoint
NUM_BITS = 8
SEED = 777

# Sample checkpoints across training trajectory (batch=1 run only: 500+)
STEPS = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500]

def load_model_and_lcx(step):
    """Load checkpoint, return (model, lcx_tensor, config)."""
    path = f"{CKPT_DIR}\\checkpoint_step_{step}.pt"
    ckpt = torch.load(path, weights_only=False, map_location=DEVICE)
    cfg = ckpt['config']
    model = SwarmMoE(
        num_bits=cfg['num_bits'], embedding_dim=cfg['embedding_dim'],
        depth=cfg['depth'], num_beings=cfg['num_beings'],
        combiner_mode=cfg['combiner'], bits_per_being=cfg['bits_per_being'],
        min_coverage=cfg['min_coverage'], mask_seed=cfg['mask_seed'],
        fibonacci=cfg['fibonacci'], combinatorial=cfg['combinatorial'],
        temporal_fibonacci=cfg['temporal_fibonacci'],
        capacity_fibonacci=cfg['capacity_fibonacci'],
        full_view=cfg.get('full_view', False), use_lcx=cfg.get('use_lcx', True),
        max_hidden=cfg.get('max_hidden', 256), min_hidden=cfg.get('min_hidden', 32),
    )
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model = model.to(DEVICE)
    model.eval()
    lcx = model.lcx.detach().cpu().clone() if hasattr(model, 'lcx') and model.lcx is not None else torch.zeros(64)
    return model, lcx, cfg

def eval_task(model, task, cfg):
    """Evaluate model on specific task, return (byte_match, bit_acc)."""
    random.seed(SEED); torch.manual_seed(SEED)
    x, y, _ = generate_multitask_batch(
        N_SAMPLES, seq_len=cfg.get('seq_len', 16),
        num_bits=NUM_BITS, task=task, seed=SEED
    )
    x, y = x.to(DEVICE), y.to(DEVICE)
    with torch.no_grad():
        output = model(x)
    pred = (output[:, 2, :] > 0.0).float()
    tgt = y[:, 2, :].float()
    byte_match = (pred == tgt).all(dim=-1).float().mean().item()
    bit_acc = (pred == tgt).float().mean().item()
    return byte_match, bit_acc

def analyze_lcx_structure(lcx, num_bits=8):
    """Analyze the 8x8 LCX as a spatial structure."""
    grid = lcx.reshape(num_bits, num_bits).numpy()

    # Basic stats
    norm = np.linalg.norm(grid)
    mean = grid.mean()
    std = grid.std()

    # Spatial structure: row means and col means (are rows/cols specializing?)
    row_means = grid.mean(axis=1)
    col_means = grid.mean(axis=0)
    row_std = row_means.std()  # high = rows are different from each other
    col_std = col_means.std()  # high = cols are different from each other

    # Entropy-like measure: how uniform vs structured?
    # Normalize to [0,1] range and compute entropy
    g_abs = np.abs(grid)
    g_norm = g_abs / (g_abs.sum() + 1e-8)
    entropy = -np.sum(g_norm * np.log(g_norm + 1e-8))
    max_entropy = np.log(64)  # uniform distribution
    rel_entropy = entropy / max_entropy  # 1.0 = uniform, 0.0 = concentrated

    # Sparsity: fraction of cells near zero (< 0.05 * max)
    threshold = 0.05 * np.abs(grid).max()
    sparsity = (np.abs(grid) < threshold).mean()

    # Diagonal vs off-diagonal (is there positional self-attention?)
    diag_mean = np.abs(np.diag(grid)).mean()
    off_diag = grid.copy()
    np.fill_diagonal(off_diag, 0)
    off_diag_mean = np.abs(off_diag).sum() / (64 - 8)
    diag_ratio = diag_mean / (off_diag_mean + 1e-8)

    # Quadrant analysis (top-left, top-right, bottom-left, bottom-right)
    half = num_bits // 2
    q_tl = np.abs(grid[:half, :half]).mean()
    q_tr = np.abs(grid[:half, half:]).mean()
    q_bl = np.abs(grid[half:, :half]).mean()
    q_br = np.abs(grid[half:, half:]).mean()

    # Symmetry: how symmetric is the grid?
    sym_horiz = np.corrcoef(grid.flatten(), np.fliplr(grid).flatten())[0, 1]
    sym_vert = np.corrcoef(grid.flatten(), np.flipud(grid).flatten())[0, 1]

    # Gradient: is there a consistent trend across rows/cols?
    row_gradient = np.polyfit(range(num_bits), row_means, 1)[0]
    col_gradient = np.polyfit(range(num_bits), col_means, 1)[0]

    return {
        'norm': float(norm), 'mean': float(mean), 'std': float(std),
        'row_std': float(row_std), 'col_std': float(col_std),
        'rel_entropy': float(rel_entropy), 'sparsity': float(sparsity),
        'diag_ratio': float(diag_ratio),
        'q_tl': float(q_tl), 'q_tr': float(q_tr),
        'q_bl': float(q_bl), 'q_br': float(q_br),
        'sym_horiz': float(sym_horiz), 'sym_vert': float(sym_vert),
        'row_gradient': float(row_gradient), 'col_gradient': float(col_gradient),
        'row_means': row_means.tolist(), 'col_means': col_means.tolist(),
        'grid': grid.tolist(),
    }

def lcx_similarity(lcx1, lcx2):
    """Cosine similarity between two LCX states."""
    v1 = lcx1.flatten()
    v2 = lcx2.flatten()
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return float(cos)

# ============================================================
print(f"LCX EVOLUTION ANALYSIS — {len(STEPS)} checkpoints, {N_SAMPLES} samples/task")
print(f"Device: {DEVICE}")
print("=" * 80)

all_data = []
prev_lcx = None
tasks = ['or', 'and', 'xor', 'add']

for step in STEPS:
    print(f"\n--- Step {step} ---")
    model, lcx, cfg = load_model_and_lcx(step)

    # Per-task accuracy
    task_results = {}
    for task in tasks:
        bm, ba = eval_task(model, task, cfg)
        task_results[task] = {'byte_match': bm, 'bit_acc': ba}
        print(f"  {task.upper():>4s}: byte={bm*100:5.1f}%  bit={ba*100:5.1f}%")

    # LCX structure analysis
    lcx_info = analyze_lcx_structure(lcx)

    # Similarity to previous checkpoint
    sim_to_prev = None
    sim_to_first = None
    if prev_lcx is not None:
        sim_to_prev = lcx_similarity(lcx.numpy(), prev_lcx)

    prev_lcx = lcx.numpy().copy()

    print(f"  LCX: norm={lcx_info['norm']:.3f}  std={lcx_info['std']:.4f}  "
          f"entropy={lcx_info['rel_entropy']:.3f}  sparsity={lcx_info['sparsity']:.2f}")
    print(f"       row_spread={lcx_info['row_std']:.4f}  col_spread={lcx_info['col_std']:.4f}  "
          f"diag_ratio={lcx_info['diag_ratio']:.2f}")
    print(f"       quadrants: TL={lcx_info['q_tl']:.3f} TR={lcx_info['q_tr']:.3f} "
          f"BL={lcx_info['q_bl']:.3f} BR={lcx_info['q_br']:.3f}")
    if sim_to_prev is not None:
        print(f"       similarity to prev: {sim_to_prev:.4f}")

    entry = {
        'step': step, 'tasks': task_results, 'lcx': lcx_info,
        'sim_to_prev': sim_to_prev,
    }
    all_data.append(entry)

    del model
    torch.cuda.empty_cache() if DEVICE == "cuda" else None

# ============================================================
# CORRELATION ANALYSIS
print("\n" + "=" * 80)
print("EVOLUTION SUMMARY")
print("=" * 80)

# Extract time series
steps_arr = [d['step'] for d in all_data]
or_bit = [d['tasks']['or']['bit_acc'] for d in all_data]
and_bit = [d['tasks']['and']['bit_acc'] for d in all_data]
xor_bit = [d['tasks']['xor']['bit_acc'] for d in all_data]
add_bit = [d['tasks']['add']['bit_acc'] for d in all_data]

lcx_norm = [d['lcx']['norm'] for d in all_data]
lcx_std = [d['lcx']['std'] for d in all_data]
lcx_entropy = [d['lcx']['rel_entropy'] for d in all_data]
lcx_sparsity = [d['lcx']['sparsity'] for d in all_data]
lcx_diag = [d['lcx']['diag_ratio'] for d in all_data]
lcx_row_sp = [d['lcx']['row_std'] for d in all_data]
lcx_col_sp = [d['lcx']['col_std'] for d in all_data]

print(f"\n{'Step':>6s} | {'OR':>5s} {'AND':>5s} {'XOR':>5s} {'ADD':>5s} | "
      f"{'Norm':>5s} {'Std':>6s} {'Entr':>5s} {'Sparse':>6s} {'Diag':>5s} {'RowSp':>5s} {'ColSp':>5s} | {'SimPr':>6s}")
print("-" * 100)
for d in all_data:
    sp = f"{d['sim_to_prev']:.3f}" if d['sim_to_prev'] is not None else "  ---"
    print(f"{d['step']:>6d} | "
          f"{d['tasks']['or']['bit_acc']*100:5.1f} {d['tasks']['and']['bit_acc']*100:5.1f} "
          f"{d['tasks']['xor']['bit_acc']*100:5.1f} {d['tasks']['add']['bit_acc']*100:5.1f} | "
          f"{d['lcx']['norm']:5.2f} {d['lcx']['std']:6.4f} {d['lcx']['rel_entropy']:5.3f} "
          f"{d['lcx']['sparsity']:6.3f} {d['lcx']['diag_ratio']:5.2f} "
          f"{d['lcx']['row_std']:5.4f} {d['lcx']['col_std']:5.4f} | {sp}")

# Correlations between LCX structure and task accuracy
print("\n" + "=" * 80)
print("CORRELATIONS: LCX Structure vs Task Accuracy")
print("=" * 80)

def corr(a, b):
    a, b = np.array(a), np.array(b)
    if a.std() < 1e-10 or b.std() < 1e-10:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

lcx_metrics = {
    'norm': lcx_norm, 'std': lcx_std, 'entropy': lcx_entropy,
    'sparsity': lcx_sparsity, 'diag_ratio': lcx_diag,
    'row_spread': lcx_row_sp, 'col_spread': lcx_col_sp,
}

print(f"\n{'Metric':<14s} | {'OR':>6s} {'AND':>6s} {'XOR':>6s} {'ADD':>6s}")
print("-" * 50)
for name, vals in lcx_metrics.items():
    r_or = corr(vals, or_bit)
    r_and = corr(vals, and_bit)
    r_xor = corr(vals, xor_bit)
    r_add = corr(vals, add_bit)
    print(f"{name:<14s} | {r_or:+6.3f} {r_and:+6.3f} {r_xor:+6.3f} {r_add:+6.3f}")

# Phase detection: when does LCX structure stabilize?
print("\n" + "=" * 80)
print("LCX STABILITY (similarity between consecutive checkpoints)")
print("=" * 80)
for d in all_data:
    if d['sim_to_prev'] is not None:
        bar = '#' * int(d['sim_to_prev'] * 40)
        print(f"  step {d['step']:>5d}: {d['sim_to_prev']:.4f}  [{bar:<40s}]")

# Grid snapshots at key moments
print("\n" + "=" * 80)
print("LCX GRID SNAPSHOTS (8x8, key checkpoints)")
print("=" * 80)

snapshot_steps = [500, 2000, 3500, 5000, 5500]
for d in all_data:
    if d['step'] in snapshot_steps:
        grid = np.array(d['lcx']['grid'])
        print(f"\n  Step {d['step']} (norm={d['lcx']['norm']:.3f}):")
        for r in range(8):
            row_str = ' '.join(f"{grid[r,c]:+.2f}" for c in range(8))
            print(f"    [{row_str}]")

# Final diagnosis
print("\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)

# Is LCX organizing toward harder tasks?
last = all_data[-1]
first = all_data[0]
or_delta = last['tasks']['or']['bit_acc'] - first['tasks']['or']['bit_acc']
and_delta = last['tasks']['and']['bit_acc'] - first['tasks']['and']['bit_acc']
xor_delta = last['tasks']['xor']['bit_acc'] - first['tasks']['xor']['bit_acc']
add_delta = last['tasks']['add']['bit_acc'] - first['tasks']['add']['bit_acc']

print(f"\n  Accuracy deltas (step {first['step']} -> {last['step']}):")
print(f"    OR:  {or_delta:+.1%}   (easy, bitwise)")
print(f"    AND: {and_delta:+.1%}   (easy, bitwise)")
print(f"    XOR: {xor_delta:+.1%}   (medium, non-linear)")
print(f"    ADD: {add_delta:+.1%}   (hard, carry chain)")

# Check if structural complexity increases with accuracy
struct_complexity = [d['lcx']['std'] * (1 - d['lcx']['sparsity']) for d in all_data]
total_acc = [(d['tasks']['or']['bit_acc'] + d['tasks']['and']['bit_acc'] +
              d['tasks']['xor']['bit_acc'] + d['tasks']['add']['bit_acc']) / 4 for d in all_data]
r_struct_acc = corr(struct_complexity, total_acc)

print(f"\n  Structure-Accuracy correlation: {r_struct_acc:+.3f}")
print(f"  (positive = LCX getting more structured as accuracy improves)")

# Check inter-checkpoint distance trend
sims = [d['sim_to_prev'] for d in all_data if d['sim_to_prev'] is not None]
if len(sims) >= 4:
    early_drift = 1 - np.mean(sims[:len(sims)//2])
    late_drift = 1 - np.mean(sims[len(sims)//2:])
    print(f"\n  Early avg drift: {early_drift:.4f}  (first half)")
    print(f"  Late avg drift:  {late_drift:.4f}  (second half)")
    if late_drift < early_drift * 0.5:
        print(f"  -> LCX is CONVERGING (settling into a fixed structure)")
    elif late_drift > early_drift:
        print(f"  -> LCX is DIVERGING (still searching)")
    else:
        print(f"  -> LCX drift is MODERATE (slow refinement)")

# Quadrant evolution
print(f"\n  Quadrant evolution (absolute mean):")
print(f"  {'Step':>6s}  {'TL':>6s} {'TR':>6s} {'BL':>6s} {'BR':>6s}  {'Pattern':>20s}")
for d in all_data:
    q = [d['lcx']['q_tl'], d['lcx']['q_tr'], d['lcx']['q_bl'], d['lcx']['q_br']]
    dominant = ['TL', 'TR', 'BL', 'BR'][np.argmax(q)]
    weakest = ['TL', 'TR', 'BL', 'BR'][np.argmin(q)]
    print(f"  {d['step']:>6d}  {q[0]:6.3f} {q[1]:6.3f} {q[2]:6.3f} {q[3]:6.3f}  "
          f"strong={dominant} weak={weakest}")

print("\nDone.")
