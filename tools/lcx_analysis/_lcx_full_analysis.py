"""Comprehensive LCX Matrix Evolution Analysis."""
import json
import numpy as np

path_lcx = r"S:\AI\work\VRAXION_DEV\Diamond Code\logs\swarm\matrix_history.jsonl"
path_log = r"S:\AI\work\VRAXION_DEV\Diamond Code\logs\swarm\5beings_64d_2layers_rf4_comb_gpu.log"

with open(path_lcx) as f:
    all_entries = [json.loads(l) for l in f]
entries = all_entries[:500]  # first 500 (dedup)

# Parse training log
losses, bit_accs, overall_accs = [], [], []
with open(path_log) as f:
    for line in f:
        if line.startswith("step "):
            parts = line.split("|")
            loss_val = float(parts[1].strip().split()[1])
            md = {}
            for kv in parts[2].strip().split():
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    try: md[k] = float(v)
                    except: pass
            losses.append(loss_val)
            bit_accs.append(md.get("bit_acc", 0))
            overall_accs.append(md.get("overall", 0))

losses = np.array(losses)
bit_accs = np.array(bit_accs)
norms = np.array([e["lcx_norm"] for e in entries])

all_lcx = np.array([np.array(e["lcx_after"]) for e in entries])
lcx_0 = all_lcx[0]
lcx_499 = all_lcx[499]
total_delta = lcx_499 - lcx_0
abs_delta = np.abs(total_delta)
delta_grid = total_delta.reshape(8, 8)

# Per-step deltas
per_step_deltas = []
for e in entries:
    bef = np.array(e["lcx_before"])
    aft = np.array(e["lcx_after"])
    per_step_deltas.append(np.linalg.norm(aft - bef))
per_step_deltas = np.array(per_step_deltas)

print("=" * 72)
print("  LCX MATRIX EVOLUTION ANALYSIS -- 500-STEP SWARM RUN")
print("=" * 72)

# 1. NORM
print("\n[1] LCX NORM EVOLUTION")
print("-" * 50)
for s in [0, 50, 100, 250, 499]:
    print(f"    step {s:>3d}:  norm = {norms[s]:.4f}")
print(f"    ratio final/initial: {norms[-1]/norms[0]:.2f}x")

# Norm growth rate
for a, b in [(0, 50), (50, 100), (100, 200), (200, 300), (300, 499)]:
    rate = (norms[b] - norms[a]) / (b - a)
    print(f"    steps {a:>3d}-{b:>3d}:  {rate:+.5f}/step")

# 2. COSINE SIMILARITY
print("\n[2] COSINE SIMILARITY")
print("-" * 50)
for s in [50, 100, 250, 499]:
    lcx_s = all_lcx[s]
    cos = np.dot(lcx_0, lcx_s) / (np.linalg.norm(lcx_0) * np.linalg.norm(lcx_s) + 1e-12)
    print(f"    step 0 vs {s:>3d}:  {cos:.4f}")
cos_250_499 = np.dot(all_lcx[250], lcx_499) / (np.linalg.norm(all_lcx[250]) * np.linalg.norm(lcx_499) + 1e-12)
print(f"    step 250 vs 499:  {cos_250_499:.4f}")
cos_0_499 = np.dot(lcx_0, lcx_499) / (np.linalg.norm(lcx_0) * np.linalg.norm(lcx_499) + 1e-12)

# 3. PER-STEP DELTA
print("\n[3] PER-STEP DELTA (L2 norm)")
print("-" * 50)
print(f"    Overall mean: {per_step_deltas.mean():.6f}")
print(f"    Max: {per_step_deltas.max():.6f} (step {per_step_deltas.argmax()})")
for name, a, b in [("Early 0-49", 0, 50), ("Mid 150-299", 150, 300), ("Late 450-499", 450, 500)]:
    print(f"    {name}:  {per_step_deltas[a:b].mean():.6f}")

# 4. CELL CHANGES
print("\n[4] CELL-WISE CHANGE (step 0 -> 499)")
print("-" * 50)
for t in [0.05, 0.1, 0.2, 0.5]:
    n = (abs_delta > t).sum()
    print(f"    |delta| > {t:.2f}:  {n:>2d}/64 ({100*n/64:.0f}%)")

went_up = (total_delta > 0.05).sum()
went_down = (total_delta < -0.05).sum()
stayed = 64 - went_up - went_down
print(f"    UP: {went_up}  DOWN: {went_down}  STABLE: {stayed}")

# 5. SPATIAL PATTERNS
print("\n[5] SPATIAL PATTERNS")
print("-" * 50)
print("    Delta heatmap (++ >+0.3, + >+0.1, -- <-0.3, - <-0.1, . stable):")
for r in range(8):
    row_str = "    "
    for c in range(8):
        v = delta_grid[r, c]
        if v > 0.3: row_str += " ++ "
        elif v > 0.1: row_str += " +  "
        elif v > 0.05: row_str += " .+ "
        elif v < -0.3: row_str += " -- "
        elif v < -0.1: row_str += " -  "
        elif v < -0.05: row_str += " .- "
        else: row_str += " .  "
    print(row_str)

# Row/col activity
row_act = [np.abs(delta_grid[r, :]).mean() for r in range(8)]
col_act = [np.abs(delta_grid[:, c]).mean() for c in range(8)]
print(f"    Most active row: {np.argmax(row_act)} ({max(row_act):.4f})")
print(f"    Least active row: {np.argmin(row_act)} ({min(row_act):.4f})")
print(f"    Most active col: {np.argmax(col_act)} ({max(col_act):.4f})")
print(f"    Least active col: {np.argmin(col_act)} ({min(col_act):.4f})")

# 6. LOSS-LCX CORRELATION
print("\n[6] LOSS-LCX CORRELATION")
print("-" * 50)
n_min = min(len(norms), len(bit_accs))
corr_norm_acc = np.corrcoef(norms[:n_min], bit_accs[:n_min])[0, 1]
print(f"    LCX norm vs bit_acc:  r = {corr_norm_acc:.4f}")
corr_norm_loss = np.corrcoef(norms[:n_min], losses[:n_min])[0, 1]
print(f"    LCX norm vs loss:     r = {corr_norm_loss:.4f}")

# 7. LATE STABILITY
print("\n[7] LATE STABILITY (steps 400-499)")
print("-" * 50)
late_lcx = all_lcx[400:500]
cell_std = late_lcx.std(axis=0)
print(f"    Mean cell std: {cell_std.mean():.6f}")
cos_late = []
for i in range(400, 499):
    a, b = all_lcx[i], all_lcx[i+1]
    cos_late.append(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
cos_late = np.array(cos_late)
print(f"    Consecutive cosine: mean={cos_late.mean():.6f}  min={cos_late.min():.6f}")

# 8. TRAJECTORY CHARACTER
print("\n[8] TRAJECTORY CHARACTER")
print("-" * 50)
cell_diffs = np.diff(all_lcx, axis=0)
sign_persist = []
for c in range(64):
    diffs = cell_diffs[:, c]
    same_sign = np.sum(diffs[:-1] * diffs[1:] > 0)
    sign_persist.append(same_sign / (len(diffs) - 1))
sign_persist = np.array(sign_persist)
print(f"    Sign persistence: mean={sign_persist.mean():.4f}")
print(f"    (0.5=random walk, >0.5=drift, <0.5=oscillation)")
print(f"    Drifters (>0.55): {(sign_persist > 0.55).sum()}/64")
print(f"    Oscillators (<0.45): {(sign_persist < 0.45).sum()}/64")

# 9. EFFECTIVE RANK
print("\n[9] EFFECTIVE RANK OF LCX (8x8 matrix)")
print("-" * 50)
for s in [0, 100, 250, 499]:
    mat = all_lcx[s].reshape(8, 8)
    svd = np.linalg.svd(mat, compute_uv=False)
    svd_n = svd / (svd.sum() + 1e-12)
    svd_n = svd_n[svd_n > 1e-10]
    eff_rank = np.exp(-np.sum(svd_n * np.log(svd_n + 1e-12)))
    print(f"    step {s:>3d}: rank={np.linalg.matrix_rank(mat, tol=0.01)}, eff_rank={eff_rank:.2f}, top_sv={svd[0]:.3f}")

# 10. TOP CHANGING CELLS
print("\n[10] TOP-10 CHANGED CELLS")
print("-" * 50)
sorted_cells = np.argsort(-abs_delta)
for idx in sorted_cells[:10]:
    r, c = idx // 8, idx % 8
    d = "UP" if total_delta[idx] > 0 else "DN"
    print(f"    [{r},{c}] {lcx_0[idx]:+.4f} -> {lcx_499[idx]:+.4f}  ({total_delta[idx]:+.4f}) {d}")

# 11. LOSS MILESTONES
print("\n[11] TRAINING MILESTONES")
print("-" * 50)
print("    Step   Loss     BitAcc")
for s in [0, 25, 50, 100, 200, 300, 400, 499]:
    if s < len(losses):
        print(f"    {s:>4d}   {losses[s]:.4f}   {bit_accs[s]:.4f}")

for thresh in [0.6, 0.7, 0.8]:
    crossed = np.where(bit_accs >= thresh)[0]
    if len(crossed) > 0:
        print(f"    bit_acc >= {thresh:.1f} first at step {crossed[0]}")
    else:
        print(f"    bit_acc >= {thresh:.1f} NEVER reached")

# VERDICT
print("\n" + "=" * 72)
print("  VERDICT")
print("=" * 72)
n_active = (abs_delta > 0.1).sum()
signals = 0
checks = []

c1 = abs(corr_norm_acc) > 0.5
signals += c1
checks.append(f"[{'x' if c1 else ' '}] Norm tracks accuracy (r={corr_norm_acc:.3f})")

c2 = cos_0_499 > 0.5
signals += c2
checks.append(f"[{'x' if c2 else ' '}] Stable direction (cos={cos_0_499:.3f})")

c3 = n_active > 20
signals += c3
checks.append(f"[{'x' if c3 else ' '}] Many cells active ({n_active}/64)")

c4 = cos_late.mean() > 0.95
signals += c4
checks.append(f"[{'x' if c4 else ' '}] Late convergence (cos={cos_late.mean():.4f})")

c5 = sign_persist.mean() > 0.52
signals += c5
checks.append(f"[{'x' if c5 else ' '}] Drifting cells (persist={sign_persist.mean():.3f})")

for ch in checks:
    print(f"    {ch}")

print(f"\n    Score: {signals}/5")
if signals >= 4:
    print("    -> USEFUL: LCX learns genuine structure.")
elif signals >= 2:
    print("    -> PARTIALLY USEFUL: Some structure, needs investigation.")
else:
    print("    -> LIKELY NOISE: Insufficient evidence of meaningful LCX content.")
print()
