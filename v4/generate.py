"""Training data generator — tiered difficulty ladder for INSTNCT v4.

.traindat files are raw bytes with no header, no encoding, no metadata.
Each byte is a value 0-255. The model converts bytes to 8-bit binary
vectors during loading. File size = exactly N bytes of training signal.

Difficulty tiers test progressively harder capabilities:

  Tier 1  Pattern copy     — can the model reproduce what it just saw?
  Tier 2  Cross-byte       — can it track patterns across byte boundaries?
  Tier 3  Arithmetic       — can it compute (not just copy)?
  Tier 4  Temporal memory  — can it remember two steps back?
  Tier 5  Working memory   — can it recall across a gap of noise?

Each tier isolates one capability. A model that passes tier N but fails
tier N+1 tells you exactly what's missing. No ambiguity.

Usage:
    python generate.py                        # 100 MB each, into data/
    python generate.py --size 1048576         # 1 MB each (bytes)
    python generate.py --out v4/data/         # custom output directory
    python generate.py --seed 0              # different reproducible seed
"""

import argparse
import os
import random
import yaml
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────
# All constants loaded from vraxion_config.yaml (data section).
_yaml_path = Path(__file__).parent / 'vraxion_config.yaml'
with open(_yaml_path, encoding='utf-8') as _f:
    _cfg = yaml.safe_load(_f)['data']

BLOCK       = _cfg['block']        # task pattern granularity (bytes per chunk)
ECHO_REPEAT = _cfg['echo_repeat']  # echo: repeat each block N times
DELAY_GAP   = _cfg['delay_gap']    # delay_echo: filler blocks between original and echo
FLIP_PROB   = _cfg['flip_prob']    # denoise: bit-flip probability

del _yaml_path, _cfg, _f


# ── Tier 1: Pattern Copy ────────────────────────────────────────
# These tasks test the most basic capability: can the model reproduce
# a pattern it just saw? If tier 1 fails, the architecture is broken.

def gen_echo(size):
    """Next block = current block. Each block repeated ECHO_REPEAT times.

    87.5% of consecutive block-pairs are identity (same block again).
    12.5% are random transitions (new block starts). The model gets
    strong gradient signal for learning "copy what you just read"
    without it being trivially all-zeros."""

    data = bytearray()
    while len(data) < size:
        block = bytes(random.randint(0, 255) for _ in range(BLOCK))
        for _ in range(ECHO_REPEAT):
            data.extend(block)
    return bytes(data[:size])


def gen_not(size):
    """Next block = bitwise NOT of current. Alternating [A][~A][A][~A]...

    Every position is 100% deterministic — no random transitions at all.
    The model must learn a simple transformation (flip all bits) rather
    than just copying. Tests that the compute path can invert, not just echo."""

    data = bytearray()
    while len(data) < size:
        block = bytes(random.randint(0, 255) for _ in range(BLOCK))
        inv = bytes(b ^ 0xFF for b in block)
        # repeat the A/~A pair 4 times for consistent signal
        for _ in range(4):
            data.extend(block)
            data.extend(inv)
    return bytes(data[:size])


# ── Tier 2: Cross-Byte ──────────────────────────────────────────
# Tier 1 patterns are block-local: each block depends only on its
# immediate predecessor. Tier 2 requires tracking structure that
# spans multiple bytes within a block — positional awareness.

def gen_shift(size):
    """Byte-rotations of a seed block. Each block is the previous one
    rotated by 1 byte position. After BLOCK steps, it wraps back.

    The model must learn that byte[i] in the next block equals byte[i-1]
    in the current block — a cross-position dependency within the block."""

    data = bytearray()
    while len(data) < size:
        seed = [random.randint(0, 255) for _ in range(BLOCK)]
        for rot in range(BLOCK):
            rotated = seed[rot:] + seed[:rot]
            data.extend(bytes(rotated))
    return bytes(data[:size])


def gen_count(size):
    """Big-endian counter incrementing by 1 each block.

    Each block is a BLOCK-byte integer, and the next block is that integer
    plus one. The model must learn carry propagation: when the last byte
    overflows, it ripples to the byte before it. This is nontrivial —
    predicting byte[15] is easy (+1 mod 256), but predicting byte[14]
    requires knowing whether byte[15] was 255."""

    data = bytearray()
    max_val = 2 ** (BLOCK * 8)
    counter = random.randint(0, max_val - 1)
    while len(data) < size:
        data.extend(counter.to_bytes(BLOCK, 'big'))
        counter = (counter + 1) % max_val
    return bytes(data[:size])


# ── Tier 3: Arithmetic ─────────────────────────────────────────
# Pure computation: the output is a mathematical function of two inputs.
# No memory, no temporal dependency — just "can the model compute?"

def gen_add(size):
    """[A (8 bytes)][B (8 bytes)][(A+B) mod 256 per byte (8 bytes)][|A-B| per byte (8 bytes)].

    Four consecutive 8-byte chunks form one problem. The model reads A and B,
    then must produce their bytewise sum and absolute difference. This requires
    actual arithmetic in the weights — no copying or pattern matching suffices."""

    data = bytearray()
    while len(data) < size:
        a = bytes(random.randint(0, 255) for _ in range(8))
        b = bytes(random.randint(0, 255) for _ in range(8))
        add_r = bytes((ai + bi) % 256 for ai, bi in zip(a, b))
        sub_r = bytes(abs(ai - bi) for ai, bi in zip(a, b))
        data.extend(a + b + add_r + sub_r)
    return bytes(data[:size])


# ── Tier 4: Temporal Memory ─────────────────────────────────────
# The output depends on TWO previous blocks, not just one.
# The model must maintain state across at least two timesteps.

def gen_fib(size):
    """Fibonacci bytes: block[t] = (block[t-1] + block[t-2]) mod 256.

    Each block is the bytewise sum of the previous two. The model must
    remember block[t-2] while processing block[t-1] — a two-step temporal
    dependency. 30-step chains before re-seeding keep the task learnable
    (longer chains would saturate to 0 due to mod-256 arithmetic)."""

    data = bytearray()
    while len(data) < size:
        prev2 = [random.randint(0, 255) for _ in range(BLOCK)]
        prev1 = [random.randint(0, 255) for _ in range(BLOCK)]
        data.extend(bytes(prev2))
        data.extend(bytes(prev1))
        for _ in range(30):
            curr = [(a + b) % 256 for a, b in zip(prev1, prev2)]
            data.extend(bytes(curr))
            prev2, prev1 = prev1, curr
    return bytes(data[:size])


# ── Tier 5: Working Memory ──────────────────────────────────────
# The hardest tier: the model must store information in ring memory,
# survive a gap of unpredictable noise, then recall the stored pattern.
# This is the capability that separates INSTNCT from feedforward models.

def gen_delay_echo(size):
    """[A][random filler × N][A] — delayed echo across a noise gap.

    The original block A appears, then N filler blocks of pure random noise,
    then A again. With DELAY_GAP=4 and BLOCK=16, the gap is 64 bytes.
    When the model's sliding window is shorter than this gap, the original A
    is no longer visible in the input — the only way to predict the echo
    is to have written A into ring memory and read it back."""

    data = bytearray()
    while len(data) < size:
        original = bytes(random.randint(0, 255) for _ in range(BLOCK))
        data.extend(original)
        for _ in range(DELAY_GAP):
            filler = bytes(random.randint(0, 255) for _ in range(BLOCK))
            data.extend(filler)
        data.extend(original)  # echo — must match the original exactly
    return bytes(data[:size])


def gen_denoise(size):
    """[noisy_A][clean_A] — read corrupted version, predict clean original.

    A clean block is generated, then a noisy copy is made by flipping each
    bit with 10% probability. The noisy block comes FIRST, then the clean one.
    The model reads corrupted data and must predict what the uncorrupted
    version looks like — a denoising autoencoder task at the byte level.
    Ring memory helps: store the noisy version, cross-reference during clean."""

    data = bytearray()
    while len(data) < size:
        clean = bytes(random.randint(0, 255) for _ in range(BLOCK))
        noisy = bytearray(clean)
        for i in range(len(noisy)):
            for bit in range(8):
                if random.random() < FLIP_PROB:
                    noisy[i] ^= (1 << bit)
        data.extend(bytes(noisy))
        data.extend(clean)
    return bytes(data[:size])


# ── Generator Registry ──────────────────────────────────────────
# Ordered by difficulty tier. Name = filename stem. The .traindat
# extension and 256 suffix are conventions from KB 01 (Training Data).

TASKS = [
    ('echo256',       gen_echo),        # tier 1 — pattern copy
    ('not256',        gen_not),         # tier 1 — pattern invert
    ('shift256',      gen_shift),       # tier 2 — cross-byte rotation
    ('count256',      gen_count),       # tier 2 — counter with carry
    ('add256',        gen_add),         # tier 3 — bytewise arithmetic
    ('fib256',        gen_fib),         # tier 4 — two-step memory
    ('delay_echo256', gen_delay_echo),  # tier 5 — recall across noise gap
    ('denoise256',    gen_denoise),     # tier 5 — pattern recovery
]


# ── CLI ──────────────────────────────────────────────────────────

def parse_size(s):
    """Parse human-readable size string: '1MB', '100MB', '1048576', etc."""
    s = s.strip().upper()
    multipliers = {'KB': 1024, 'MB': 1024**2, 'GB': 1024**3}
    for suffix, mult in multipliers.items():
        if s.endswith(suffix):
            return int(float(s[:-len(suffix)]) * mult)
    return int(s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate .traindat difficulty ladder for INSTNCT v4.'
    )
    parser.add_argument('--size', default='100MB',
                        help='target file size (e.g. 1MB, 100MB, 1073741824). Default: 100MB')
    parser.add_argument('--out', default='data',
                        help='output directory. Default: data/')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for reproducibility. Default: 42')
    args = parser.parse_args()

    target = parse_size(args.size)
    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    print(f'Generating {len(TASKS)} tasks, {target:,} bytes each, seed={args.seed}')
    print(f'Output: {os.path.abspath(args.out)}/')
    print()

    for name, fn in TASKS:
        filename = f'{name}.traindat'
        path = os.path.join(args.out, filename)
        data = fn(target)
        with open(path, 'wb') as f:
            f.write(data)
        print(f'  {filename:<25s} {len(data):>12,} bytes')

    print()
    print('Done.')
