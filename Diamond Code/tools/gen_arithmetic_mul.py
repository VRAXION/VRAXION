#!/usr/bin/env python3
"""Generate arithmetic multiplication theme file.

Produces a JSONL theme file with A*B=C pairs.
Includes commutative equivalents (A*B and B*A) in sets.
Difficulty scaled by operand size.

Usage:
  python tools/gen_arithmetic_mul.py data/themes/arithmetic_mul.jsonl
  python tools/gen_arithmetic_mul.py data/themes/arithmetic_mul.jsonl --n_curriculum 5000 --n_gist 200
"""

import json
import random
import sys
from pathlib import Path


def difficulty_for_range(max_val: int) -> int:
    """Map operand range to difficulty 1-6."""
    if max_val <= 9:
        return 1
    elif max_val <= 99:
        return 2
    elif max_val <= 999:
        return 3
    elif max_val <= 9999:
        return 4
    elif max_val <= 99999:
        return 5
    else:
        return 6


def generate_pairs(n: int, difficulty: int, seed: int) -> list:
    """Generate up to n unique multiplication pairs at given difficulty."""
    rng = random.Random(seed)
    ranges = {
        1: (0, 9),
        2: (0, 99),
        3: (0, 999),
        4: (0, 9999),
        5: (0, 99999),
        6: (0, 999999),
    }
    lo, hi = ranges[difficulty]
    pool_size = (hi - lo + 1) * (hi - lo + 2) // 2
    cap = min(n, pool_size)

    pairs = []
    seen = set()
    max_attempts = cap * 20

    attempts = 0
    while len(pairs) < cap and attempts < max_attempts:
        a = rng.randint(lo, hi)
        b = rng.randint(lo, hi)
        key = (min(a, b), max(a, b))
        attempts += 1
        if key in seen:
            continue
        seen.add(key)
        c = a * b
        pairs.append((a, b, c))

    return pairs


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate arithmetic_mul theme")
    parser.add_argument("output", type=str, help="Output .jsonl path")
    parser.add_argument("--n_curriculum", type=int, default=5000,
                        help="Number of curriculum pairs (default: 5000)")
    parser.add_argument("--n_gist", type=int, default=200,
                        help="Number of gist pairs (default: 200)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Distribution across difficulties for curriculum
    curriculum_dist = {
        1: int(args.n_curriculum * 0.30),
        2: int(args.n_curriculum * 0.25),
        3: int(args.n_curriculum * 0.20),
        4: int(args.n_curriculum * 0.12),
        5: int(args.n_curriculum * 0.08),
        6: int(args.n_curriculum * 0.05),
    }

    # Gist is all difficulty 1 (golden essentials)
    gist_pairs = generate_pairs(args.n_gist, difficulty=1, seed=args.seed)

    lines = []

    total = sum(curriculum_dist.values()) * 2 + args.n_gist
    gist_ratio = args.n_gist / total if total > 0 else 0.10
    header = {
        "_meta": True,
        "theme": "arithmetic_mul",
        "version": 1,
        "encoding": "utf8",
        "gist_ratio": round(gist_ratio, 4),
        "active": True,
        "generator": "gen_arithmetic_mul.py",
        "seed": args.seed,
    }
    lines.append(json.dumps(header))

    pair_id = 0

    # Curriculum pairs with commutative sets
    for diff in sorted(curriculum_dist.keys()):
        n = curriculum_dist[diff]
        pairs = generate_pairs(n, diff, seed=args.seed + diff * 1000)

        for a, b, c in pairs:
            set_id = f"am-set-{pair_id:06d}"

            # A * B = C
            lines.append(json.dumps({
                "id": f"am-{pair_id:06d}a",
                "s": "c",
                "d": diff,
                "set": set_id,
                "in": f"{a}*{b}",
                "out": f"{c}",
            }))

            # B * A = C (commutative equivalent)
            if a != b:
                lines.append(json.dumps({
                    "id": f"am-{pair_id:06d}b",
                    "s": "c",
                    "d": diff,
                    "set": set_id,
                    "in": f"{b}*{a}",
                    "out": f"{c}",
                }))

            pair_id += 1

    # Gist pairs
    for i, (a, b, c) in enumerate(gist_pairs):
        lines.append(json.dumps({
            "id": f"am-g-{i:05d}",
            "s": "g",
            "d": 1,
            "in": f"{a}*{b}",
            "out": f"{c}",
        }))

    with open(output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    n_written = len(lines) - 1
    file_size = output.stat().st_size
    print(f"Generated {output.name}: {n_written} pairs, {file_size / 1024:.1f} KB")
    print(f"  curriculum: {n_written - len(gist_pairs)} (with commutative variants)")
    print(f"  gist: {len(gist_pairs)}")
    print(f"  seed: {args.seed}")


if __name__ == "__main__":
    main()
