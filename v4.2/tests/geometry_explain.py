"""
Visualize what each geometry actually looks like as input_projection patterns.
For each geometry: show which hidden neurons light up for input 0, 5, 13, 26.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from tests.geometry_sweep import (
    geom_random, geom_concentric_rings, geom_2d_grid_walls,
    geom_fourier, geom_spiral, geom_sparse_binary, geom_clustered,
    geom_simplex, geom_hypercube, geom_3d_helix, geom_identity_block,
    V, H, SCALE
)

SEED = 42
INPUTS_TO_SHOW = [0, 5, 13, 26]

GEOMETRIES = {
    'random':       (geom_random, "84% — Teljesen véletlen irányok H dimenzióban"),
    'concentric':   (geom_concentric_rings, "82% — Gyűrűk különböző sugárral"),
    'fourier':      (geom_fourier, "81% — Minden input más frekvencia"),
    'sparse-8':     (geom_sparse_binary, "76% — Pontosan 8 random neuron aktív"),
    'simplex':      (geom_simplex, "74% — Szimplex csúcsok H dimenzióban"),
    'clustered':    (geom_clustered, "68% — 3 klaszterbe csoportosítva"),
    'spiral':       (geom_spiral, "44% — 2D spirál → H dimenzió"),
    'grid-vert':    (geom_2d_grid_walls, "33% — 9×9 rácson függőleges falak"),
    'identity':     (geom_identity_block, "76% — 1:1 egy neuron per input"),
}

for name, (geom_fn, desc) in GEOMETRIES.items():
    input_projection, output_projection = geom_fn(SEED)

    print(f"\n{'='*80}")
    print(f"  {name}: {desc}")
    print(f"{'='*80}")
    print(f"  input_projection shape: {input_projection.shape}, scale: [{input_projection.min():.2f}, {input_projection.max():.2f}]")

    # Overlap analysis: how similar are different inputs?
    # Cosine similarity between all pairs
    norms = np.linalg.norm(input_projection, axis=1, keepdims=True)
    norms[norms == 0] = 1
    W_normed = input_projection / norms
    cosim = W_normed @ W_normed.T  # V×V
    np.fill_diagonal(cosim, 0)  # ignore self
    avg_sim = np.abs(cosim).mean()
    max_sim = np.abs(cosim).max()
    print(f"  Input-pár cosine hasonlóság: avg={avg_sim:.3f} max={max_sim:.3f}")
    print(f"  (Alacsony = jó szeparáció, Magas = hasonló inputok = rossz)")

    # How many neurons does each input activate above threshold?
    THRESH = 0.5
    for inp_idx in INPUTS_TO_SHOW:
        row = input_projection[inp_idx]
        above = (np.abs(row) > THRESH)
        n_above = above.sum()
        strong = np.where(above)[0]

        # Show activation pattern as 9×9 grid (for visual intuition)
        print(f"\n  input[{inp_idx:2d}]: {n_above:2d}/{H} neurons > threshold")

        # 9×9 visual
        grid = ""
        for r in range(9):
            line = "    "
            for c in range(9):
                h = r * 9 + c
                val = row[h]
                if val > THRESH:
                    line += " ██"
                elif val < -THRESH:
                    line += " ░░"
                elif abs(val) > 0.1:
                    line += " ··"
                else:
                    line += "   "
            grid += line + "\n"
        print(grid, end="")

    # Show the key diagnostic: effective rank of input_projection
    U, S, Vt = np.linalg.svd(input_projection, full_matrices=False)
    S_norm = S / S.sum()
    eff_rank = np.exp(-np.sum(S_norm * np.log(S_norm + 1e-10)))
    energy_90 = np.searchsorted(np.cumsum(S**2) / (S**2).sum(), 0.9) + 1

    print(f"  Effektív rang: {eff_rank:.1f} / {V}")
    print(f"  90% energia: {energy_90} dimenzióban")

print(f"\n\n{'='*80}")
print(f"  ÖSSZEFOGLALÓ: MI SZÁMÍT?")
print(f"{'='*80}")
print(f"""
  ┌─────────────────────┬───────┬──────────┬──────────┐
  │ Geometry            │ Score │ Eff.Rank │ Avg Sim  │
  ├─────────────────────┼───────┼──────────┼──────────┤""")

for name, (geom_fn, desc) in GEOMETRIES.items():
    input_projection, output_projection = geom_fn(SEED)
    norms = np.linalg.norm(input_projection, axis=1, keepdims=True)
    norms[norms == 0] = 1
    W_normed = input_projection / norms
    cosim = W_normed @ W_normed.T
    np.fill_diagonal(cosim, 0)
    avg_sim = np.abs(cosim).mean()
    U, S, Vt = np.linalg.svd(input_projection, full_matrices=False)
    S_norm = S / S.sum()
    eff_rank = np.exp(-np.sum(S_norm * np.log(S_norm + 1e-10)))
    score_str = desc.split("—")[0].strip()
    print(f"  │ {name:<19s} │ {score_str:>5s} │ {eff_rank:>7.1f}  │ {avg_sim:>7.3f}  │")

print(f"  └─────────────────────┴───────┴──────────┴──────────┘")
print(f"\n  Korreláció: magas rang + alacsony hasonlóság = magas score")
