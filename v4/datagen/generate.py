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

import argparse                      # CLI argument parsing (--size, --task, --seed, --out)
import os                            # file/directory ops (makedirs, path.join, path.abspath)
import random                        # PRNG — randbytes, getrandbits, randint, random()
import time                          # wall-clock timing for gen/io/total benchmarks
import yaml                          # vraxion_config.yaml loader (BLOCK, ECHO_REPEAT, etc.)
from pathlib import Path             # config file resolution relative to this script

# ── Config ───────────────────────────────────────────────────────
# Source: vraxion_config.yaml → 'data' section.
# Loaded once at import. Shared pattern with instnct.py.

def _load_yaml(section: str) -> dict:
    """Load a named section from vraxion_config.yaml. Fail loud on any defect."""
    cfg_path = Path(__file__).parent.parent / 'config' / 'vraxion_config.yaml'  # datagen/ → v4/ → config/
    if not cfg_path.exists():                                     # missing file — fatal
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    try:
        with open(cfg_path, encoding='utf-8') as f:
            root = yaml.safe_load(f)                              # parse YAML into dict
    except yaml.YAMLError as e:                                   # malformed syntax — fatal
        raise RuntimeError(f"Corrupted YAML in {cfg_path}: {e}")
    if not isinstance(root, dict):                                # empty or non-dict — fatal
        raise RuntimeError(f"Expected dict in {cfg_path}, got {type(root).__name__}")
    if section not in root:                                       # missing section — fatal, show what exists
        raise KeyError(f"Missing '{section}' in {cfg_path} (have: {list(root.keys())})")
    return root[section]

_cfg = _load_yaml('data')

BLOCK       = int(_cfg['block'])         # bytes per chunk — all generators structure data in BLOCK-sized pieces
ECHO_REPEAT = int(_cfg['echo_repeat'])   # echo repetitions — each block appears this many times consecutively
DELAY_GAP   = int(_cfg['delay_gap'])     # filler blocks between original and echo in delay_echo task
FLIP_PROB   = float(_cfg['flip_prob'])   # per-byte corruption probability in denoise task (0.125 = ~12.5%)

del _cfg

# ── Lookup Tables & Constants ──────────────────────────────────
_INV_TABLE    = bytes(b ^ 0xFF for b in range(256))  # 256-byte NOT table — C-level translate() uses this
_COUNTER_MASK = (1 << (BLOCK * 8)) - 1               # 2^128-1 bitmask — AND wrap replaces modulo in func_countbyte_byt


# ── Tier 1: Pattern Copy ────────────────────────────────────────
# Can the model reproduce what it just saw?
# If tier 1 fails, the architecture is broken.

def func_echorepat_byt(size):
    """[A][A][A]...[A][B][B][B]...[B]... — each block repeated ECHO_REPEAT times.

    7/8 consecutive pairs are identity (same block). 1/8 are random transitions.
    Strong copy signal without being trivially constant.
    Mask: first block of each chunk = 0 (seed), remaining repeats = 1."""

    chunk = BLOCK * ECHO_REPEAT                    # bytes per full pattern cycle (16×8 = 128)
    n_chunks = -(-size // chunk)                    # ceiling div — exact count, no +1 overshoot
    unit_mask = b'\x00' * BLOCK + b'\x01' * (BLOCK * (ECHO_REPEAT - 1))  # seed=0, copies=1
    data = bytearray()                             # accumulator
    mask = bytearray()                             # supervision mask
    for _ in range(n_chunks):
        block = random.randbytes(BLOCK)            # one random BLOCK-byte chunk (single C call)
        data.extend(block * ECHO_REPEAT)           # repeat via C-level bytes multiply
        mask.extend(unit_mask)                     # pre-calc mask pattern per chunk
    del data[size:]                                # trim overshoot in-place
    del mask[size:]                                # trim mask in sync
    return bytes(data), bytes(mask)


def func_invrtbits_byt(size):
    """[A][~A][A][~A]... — bitwise NOT, each A/~A pair repeated 4× per seed.

    100% deterministic — no random transitions. The model must learn
    to invert all bits, not just copy. Tests compute path, not memory.
    Mask: first A = 0 (seed), all ~A and repeated A/~A = 1."""

    pair_repeat = 4                                # A/~A pairs per seed block (4×2×16 = 128 bytes/cycle)
    cycle = BLOCK * 2 * pair_repeat                # bytes per full cycle
    n_cycles = -(-size // cycle)                    # ceiling div — exact count, no +1 overshoot
    unit_mask = b'\x00' * BLOCK + b'\x01' * (BLOCK * (2 * pair_repeat - 1))  # seed=0, rest=1
    data = bytearray()                             # accumulator
    mask = bytearray()                             # supervision mask
    for _ in range(n_cycles):
        block = random.randbytes(BLOCK)            # random BLOCK-byte chunk (single C call)
        inv = block.translate(_INV_TABLE)           # bitwise NOT via C-level lookup (no Python loop)
        data.extend((block + inv) * pair_repeat)   # 4 A/~A pairs via C-level bytes multiply
        mask.extend(unit_mask)                     # pre-calc mask pattern per cycle
    del data[size:]                                # trim overshoot in-place
    del mask[size:]                                # trim mask in sync
    return bytes(data), bytes(mask)


# TODO: gen_xor — [A][B][A^B], two-input binary logic (natural func_invrtbits_byt pair)
# TODO: gen_mask — [A][M][A&M], selective bit masking

# ── Tier 2: Cross-Byte ──────────────────────────────────────────
# Tier 1 patterns are block-local: each block depends only on its
# immediate predecessor. Tier 2 requires tracking structure that
# spans multiple bytes within a block — positional awareness.

def func_byterotat_byt(size):
    """Byte-rotations of a seed block. Each block is the previous one
    left-rotated by 1 byte position. After BLOCK steps, it wraps back.

    The model must learn that byte[i] in the next block equals byte[i+1]
    in the current block — a cross-position dependency within the block.
    Mask: seed (rot=0) = 0, all rotations = 1."""

    cycle = BLOCK * BLOCK                            # one full rotation cycle = BLOCK² bytes (16×16 = 256)
    n_cycles = -(-size // cycle)                     # ceiling div — exact count, no +1 overshoot
    unit_mask = b'\x00' * BLOCK + b'\x01' * (BLOCK * (BLOCK - 1))  # seed=0, rotations=1
    data = bytearray()                               # accumulator
    mask = bytearray()                               # supervision mask
    for _ in range(n_cycles):
        seed = random.randbytes(BLOCK)               # one random BLOCK-byte chunk (single C call)
        doubled = seed * 2                           # doubled buffer — sliding window replaces slice+concat
        for rot in range(BLOCK):                     # BLOCK rotations per seed
            data.extend(doubled[rot:rot + BLOCK])    # single slice from doubled buffer (no concat)
        mask.extend(unit_mask)                       # pre-calc mask pattern per cycle
    del data[size:]                                  # trim overshoot in-place
    del mask[size:]                                  # trim mask in sync
    return bytes(data), bytes(mask)


def func_countbyte_byt(size):
    """Big-endian counter incrementing by 1 each block.

    Each block is a BLOCK-byte integer, and the next block is that integer
    plus one. The model must learn carry propagation: when the last byte
    overflows, it ripples to the byte before it. This is nontrivial —
    predicting byte[15] is easy (+1 mod 256), but predicting byte[14]
    requires knowing whether byte[15] was 255.
    Mask: first block = 0 (random start), all increments = 1."""

    n_blocks = -(-size // BLOCK)                       # ceiling div — exact count, no +1 overshoot
    counter = random.randint(0, _COUNTER_MASK)         # random 128-bit start point
    data = bytearray()                                 # accumulator
    mask = bytearray()                                 # supervision mask
    mask.extend(b'\x00' * BLOCK)                       # first block = seed (unpredictable)
    for _ in range(n_blocks):
        data.extend(counter.to_bytes(BLOCK, 'big'))    # BLOCK-byte big-endian encoding (C-level)
        counter = (counter + 1) & _COUNTER_MASK        # increment + bitmask wrap (no modulo)
    mask.extend(b'\x01' * (BLOCK * (n_blocks - 1)))    # all increments = supervised
    del data[size:]                                    # trim overshoot in-place
    del mask[size:]                                    # trim mask in sync
    return bytes(data), bytes(mask)


# TODO: gen_reverse — emit block in reversed byte order (positional remapping)
# TODO: gen_sort — emit block with bytes sorted ascending (ordering task)

# ── Tier 3: Arithmetic ─────────────────────────────────────────
# Short-range memory + compute: the model recalls A and B (up to 16 bytes
# back) then produces a mathematical function of both. Tests arithmetic
# in the weights — copying or pattern matching alone won't suffice.

def func_addandsub_byt(size):
    """[A][B][(A+B) mod 256 per byte][|A-B| per byte] — four OP_LEN chunks.

    The model reads A and B, then must produce their bytewise sum and
    absolute difference. Requires recalling inputs across a short window
    (2 × OP_LEN bytes) and computing per-byte arithmetic.
    Mask: A,B = 0 (random operands), A+B and |A-B| = 1."""

    OP_LEN = 8                                       # operand length in bytes — independent of BLOCK
    UNIT = OP_LEN * 4                                # one full problem: A + B + add + sub (32 bytes)
    n_units = -(-size // UNIT)                       # ceiling div — exact count
    unit_mask = b'\x00' * (OP_LEN * 2) + b'\x01' * (OP_LEN * 2)  # operands=0, results=1
    data = bytearray()                               # accumulator
    mask = bytearray()                               # supervision mask
    for _ in range(n_units):
        ab = random.randbytes(OP_LEN * 2)            # A and B in one C call (16 bytes)
        a, b = ab[:OP_LEN], ab[OP_LEN:]             # split into operands
        add_r = bytes((ai + bi) & 0xFF for ai, bi in zip(a, b))   # bytewise sum — bitmask, no modulo
        sub_r = bytes(abs(ai - bi) for ai, bi in zip(a, b))       # bytewise absolute difference
        data.extend(ab + add_r + sub_r)              # one problem = 32 bytes
        mask.extend(unit_mask)                       # pre-calc mask pattern per unit
    del data[size:]                                  # trim overshoot in-place
    del mask[size:]                                  # trim mask in sync
    return bytes(data), bytes(mask)


# TODO: gen_mul — bytewise (A*B) & 0xFF, or gen_mod — A mod B per byte

# ── Tier 4: Temporal Memory ─────────────────────────────────────
# The output depends on TWO previous blocks, not just one.
# The model must maintain state across at least two timesteps.

def func_fiboseqnc_byt(size):
    """Fibonacci bytes: block[t] = (block[t-1] + block[t-2]) mod 256.

    Each block is the bytewise sum of the previous two. The model must
    remember block[t-2] while processing block[t-1] — a two-step temporal
    dependency. 30-step chains before re-seeding keep the task learnable
    (longer chains stay periodic via Pisano period, not saturating).
    Mask: 2 seed blocks = 0, 30 fib steps = 1."""

    CHAIN_LEN = 30                                     # steps per chain before re-seeding
    chain = (2 + CHAIN_LEN) * BLOCK                    # bytes per full chain (2 seeds + 30 fib steps)
    n_chains = -(-size // chain)                        # ceiling div — exact count, no +1 overshoot
    unit_mask = b'\x00' * (BLOCK * 2) + b'\x01' * (BLOCK * CHAIN_LEN)  # seeds=0, fib=1
    data = bytearray()                                  # accumulator
    mask = bytearray()                                  # supervision mask
    for _ in range(n_chains):
        prev2 = random.randbytes(BLOCK)                 # seed block t=0 (single C call)
        prev1 = random.randbytes(BLOCK)                 # seed block t=1 (single C call)
        data.extend(prev2)                              # emit seed 0
        data.extend(prev1)                              # emit seed 1
        for _ in range(CHAIN_LEN):                      # 30 Fibonacci steps per chain
            curr = bytes((a + b) & 0xFF for a, b in zip(prev1, prev2))  # bytewise sum — bitmask, no modulo
            data.extend(curr)                           # emit fib block
            prev2, prev1 = prev1, curr                  # slide window forward
        mask.extend(unit_mask)                          # pre-calc mask pattern per chain
    del data[size:]                                     # trim overshoot in-place
    del mask[size:]                                     # trim mask in sync
    return bytes(data), bytes(mask)


# TODO: gen_rle — run-length encoded patterns, variable-length dependency

# ── Tier 5: Working Memory ──────────────────────────────────────
# The hardest tier: the model must store information in ring memory,
# survive a gap of unpredictable noise, then recall the stored pattern.
# This is the capability that separates INSTNCT from feedforward models.

def func_delayecho_byt(size):
    """[A][random filler × N][A] — delayed echo across a noise gap.

    The original block A appears, then N filler blocks of pure random noise,
    then A again. With DELAY_GAP=4 and BLOCK=16, the gap is 64 bytes.
    When the model's sliding window is shorter than this gap, the original A
    is no longer visible in the input — the only way to predict the echo
    is to have written A into ring memory and read it back.
    Mask: original + filler = 0, echo = 1. Only 1/6 is supervised."""

    unit = (2 + DELAY_GAP) * BLOCK                    # bytes per unit: original + gap + echo (96 bytes)
    n_units = -(-size // unit)                         # ceiling div — exact count, no +1 overshoot
    unit_mask = b'\x00' * (BLOCK * (1 + DELAY_GAP)) + b'\x01' * BLOCK  # original+filler=0, echo=1
    data = bytearray()                                 # accumulator
    mask = bytearray()                                 # supervision mask
    for _ in range(n_units):
        original = random.randbytes(BLOCK)             # pattern block (single C call)
        data.extend(original)                          # emit original
        data.extend(random.randbytes(BLOCK * DELAY_GAP))  # emit noise gap — one C call, no loop
        data.extend(original)                          # emit echo — must match original exactly
        mask.extend(unit_mask)                         # pre-calc mask pattern per unit
    del data[size:]                                    # trim overshoot in-place
    del mask[size:]                                    # trim mask in sync
    return bytes(data), bytes(mask)


def func_denoisbyt_byt(size):
    """[noisy_A][clean_A] — read corrupted version, predict clean original.

    A clean block is generated, then ~FLIP_PROB of its bytes are replaced
    with independent random values (true random noise, not bit-flip).
    Selected byte positions get a fresh random byte; unselected stay clean.
    The noisy block comes FIRST, then the clean one.
    Mask: noisy = 0, clean = 1 (loss won't reach 0 — that's the task)."""

    pair = BLOCK * 2                                    # bytes per unit: noisy + clean
    n_pairs = -(-size // pair)                          # ceiling div — exact count, no +1 overshoot
    unit_mask = b'\x00' * BLOCK + b'\x01' * BLOCK       # noisy=0, clean=1
    data = bytearray()                                  # accumulator
    sup = bytearray()                                   # supervision mask
    for _ in range(n_pairs):
        clean = random.randbytes(BLOCK)                 # random BLOCK-byte chunk (single C call)
        noisy = bytearray(clean)                        # start with clean copy
        for i in range(BLOCK):                          # per-byte corruption decision
            if random.random() < FLIP_PROB:             # FLIP_PROB from YAML (default 0.1 = 10%)
                noisy[i] = random.randint(0, 255)       # replace with independent random byte
        data.extend(noisy)                              # emit corrupted version first
        data.extend(clean)                              # emit clean original second
        sup.extend(unit_mask)                           # pre-calc mask pattern per pair
    del data[size:]                                     # trim overshoot in-place
    del sup[size:]                                      # trim mask in sync
    return bytes(data), bytes(sup)


def func_denoisbyt_lite(size):
    """[noisy_A][clean_A] — lite variant using triple-AND bit-flip noise.

    Fixed ~12.5% of bits are flipped via XOR with a triple-AND mask
    (3 × getrandbits AND'd → P(flip)=0.5³). Faster than byte-level replace
    but noise is structured: flipped bits are always the inverse of clean.
    Mask: noisy = 0, clean = 1."""

    BITS = BLOCK * 8                                    # 128 bits per block
    pair = BLOCK * 2                                    # bytes per unit: noisy + clean
    n_pairs = -(-size // pair)                          # ceiling div — exact count, no +1 overshoot
    unit_mask = b'\x00' * BLOCK + b'\x01' * BLOCK       # noisy=0, clean=1
    data = bytearray()                                  # accumulator
    sup = bytearray()                                   # supervision mask
    for _ in range(n_pairs):
        clean = random.randbytes(BLOCK)                 # random BLOCK-byte chunk (single C call)
        clean_int = int.from_bytes(clean, 'big')        # 128-bit integer for bitwise ops
        flip = (random.getrandbits(BITS)                # triple-AND: each layer 50% per bit
                & random.getrandbits(BITS)              #   0.5 × 0.5 × 0.5 = 12.5% flip rate
                & random.getrandbits(BITS))             #   3 C calls, no Python loop
        noisy = (clean_int ^ flip).to_bytes(BLOCK, 'big')  # XOR flips masked bits
        data.extend(noisy)                              # emit corrupted version first
        data.extend(clean)                              # emit clean original second
        sup.extend(unit_mask)                           # pre-calc mask pattern per pair
    del data[size:]                                     # trim overshoot in-place
    del sup[size:]                                      # trim mask in sync
    return bytes(data), bytes(sup)


# TODO: gen_delay_xor — [A][noise][B][noise][A^B], working memory + compute combo

# ── Generator Registry ──────────────────────────────────────────
# Ordered by difficulty tier. Name = filename stem. The .traindat
# extension and 256 suffix are conventions from KB 01 (Training Data).

TASKS = [
    ('echo256',       func_echorepat_byt),          # tier 1 — pattern copy
    ('not256',        func_invrtbits_byt),          # tier 1 — pattern invert
    ('shift256',      func_byterotat_byt),          # tier 2 — cross-byte rotation
    ('count256',      func_countbyte_byt),          # tier 2 — counter with carry
    ('add256',        func_addandsub_byt),          # tier 3 — bytewise arithmetic
    ('fib256',        func_fiboseqnc_byt),          # tier 4 — two-step memory
    ('delay_echo256', func_delayecho_byt),          # tier 5 — recall across noise gap
    ('denoise256',    func_denoisbyt_byt),          # tier 5 — pattern recovery (byte-level replace)
    ('denoise256_lite', func_denoisbyt_lite),       # tier 5 — pattern recovery (triple-AND bit-flip)
]


# ── CLI ──────────────────────────────────────────────────────────

def func_parsesize_int(s):
    """Parse human-readable size string: '1MB', '100MB', '1048576', etc."""
    s = s.strip().upper()
    multipliers = {'KB': 1024, 'MB': 1024**2, 'GB': 1024**3}
    for suffix, mult in multipliers.items():
        if s.endswith(suffix):
            return int(float(s[:-len(suffix)]) * mult)
    return int(s)


def func_generate_all(tasks, target: int, seed: int, out_dir: str, label: str):
    """Generate all tasks with the given seed into out_dir.

    Extracted from __main__ so it can be called twice — once for training
    data and once for eval data — without duplicating the generation loop.

    Args:
        tasks:   list of (name, generator_fn) pairs
        target:  target byte count per task
        seed:    PRNG seed — different seed = different random bytes, same patterns
        out_dir: output directory (created if missing)
        label:   human-readable label for progress output ('training' or 'eval')
    """
    random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    print(f'[{label}] {len(tasks)} tasks, {target:,} bytes each, seed={seed}')
    print(f'  Output: {os.path.abspath(out_dir)}/')
    print()

    t0 = time.time()
    for name, fn in tasks:
        base = os.path.join(out_dir, name)
        gen_t0 = time.time()
        data, sup_mask = fn(target)                    # generate data + supervision mask
        gen_t1 = time.time()
        with open(base + '.traindat', 'wb') as f:
            f.write(data)                              # raw training bytes
        with open(base + '.mask', 'wb') as f:
            f.write(sup_mask)                          # supervision mask (0=noise, 1=supervised)
        io_t1 = time.time()
        gen_s = gen_t1 - gen_t0                        # generation time
        io_s = io_t1 - gen_t1                          # disk write time
        mb = len(data) / 1024 / 1024                   # size in MB
        mbps = mb / gen_s if gen_s > 0 else float('inf')
        # fraction of supervised bytes (use bytes.count() for C-speed; sum() would be O(n) Python)
        mask_frac = sup_mask.count(1) / len(sup_mask) if sup_mask else 0
        print(f'  {name + ".traindat":<25s} {len(data):>12,} bytes  gen {gen_s:>5.2f}s  io {io_s:>5.2f}s  ({mbps:>.0f} MB/s)  signal: {mask_frac:.1%}')

    total_s = time.time() - t0
    print(f'\n  [{label}] Done. {len(tasks)} tasks in {total_s:.2f}s.')


# ── Eval Seed ────────────────────────────────────────────────────
# Eval seed is derived from the training seed by XOR with a fixed constant.
# This guarantees: (a) different random bytes than training, (b) deterministic
# given the same --seed, (c) no collision unless seed == seed ^ EVAL_SALT (impossible
# for any 64-bit int since EVAL_SALT != 0).
_EVAL_SALT = 0x5EED_E7A1


if __name__ == '__main__':
    _task_names = [n for n, _ in TASKS]                # valid task names for argparse choices

    parser = argparse.ArgumentParser(
        description='Generate .traindat difficulty ladder for INSTNCT v4.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--size', default='100MB',
                        help='target file size (e.g. 1MB, 100MB, 1073741824)')
    _v4_root = Path(__file__).parent.parent                             # v4/
    _default_out = str(_v4_root / 'training_data')                     # v4/training_data/
    _default_eval_out = str(_v4_root / 'eval_data')                    # v4/eval_data/
    parser.add_argument('--out', default=_default_out,
                        help='output directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for reproducibility')
    parser.add_argument('--task', nargs='*', choices=_task_names, metavar='TASK',
                        help='generate only these tasks (e.g. --task echo256 denoise256)')
    parser.add_argument('--eval', action='store_true',
                        help='also generate eval data (same tasks, different seed)')
    parser.add_argument('--eval-seed', type=int, default=None,
                        help='eval seed (default: seed XOR 0x5EED_EVA1)')
    parser.add_argument('--eval-out', default=None,
                        help='eval output directory (default: v4/eval_data/)')
    args = parser.parse_args()

    target = func_parsesize_int(args.size)

    # filter tasks if --task specified
    tasks_to_run = TASKS
    if args.task:
        task_dict = dict(TASKS)
        tasks_to_run = [(n, task_dict[n]) for n in args.task]

    print(f'VRAXION v4 — Synthetic Data Generator v1.0')
    print(f'{"=" * 48}')

    # ── Training data ──
    func_generate_all(tasks_to_run, target, args.seed, args.out, 'TRAIN')

    # ── Eval data (optional) ──
    if args.eval:
        eval_seed = args.eval_seed if args.eval_seed is not None else (args.seed ^ _EVAL_SALT)
        eval_out = args.eval_out or _default_eval_out
        print()
        func_generate_all(tasks_to_run, target, eval_seed, eval_out, 'EVAL')
