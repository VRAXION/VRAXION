"""Generate curated traindat difficulty ladder.

Tier 1 (sanity): echo256, not256 — 100% predictable
Tier 2 (simple): shift256, count256 — cross-byte patterns
Tier 3 (logic):  add256 — arithmetic (logic256 already exists)
Tier 4 (memory): fib256 — two-step dependency
Tier 5 (LCX):   delay_echo256, denoise256 — require working memory
"""
import datetime
import json
import os
import random

TARGET = 104_857_600  # 100MB each
BLOCK = 16  # must match bytes_per_pos = num_bits // 8 = 128 // 8
ECHO_REPEAT = 8  # repeat each block N times; identity fraction = (N-1)/N = 87.5%
OUT_DIR = "data/traindat"

# Delay-echo config
DELAY_FILLER_BLOCKS = 4  # filler blocks between original and echo (gap = 4 * 16 = 64 bytes)

# Denoise config
DENOISE_FLIP_PROB = 0.1  # probability of flipping each bit in noisy copy


COPY_RUN_LENGTH = 128  # each byte repeated N times; with seq_len=62, most windows are 100% predictable


def gen_constant(path):
    """All zeros. Ultimate sanity check — model just learns output bias.

    100% predictable. Every byte is 0x00. The model needs to push all 8 sigmoid
    outputs below 0.5. If this fails, the architecture is fundamentally broken.
    """
    data = bytes(TARGET)
    with open(path, 'wb') as f:
        f.write(data)


def gen_copy_echo(path):
    """Each random byte repeated COPY_RUN_LENGTH times. The simplest echo task.

    With run_length=128 and seq_len=62, ~52% of windows are fully within one run
    (100% predictable), and the rest have at most 1 boundary in 62 positions (~98.4%).
    Overall ~99% of positions are 'predict = copy current byte'.

    The temporal dependency is ZERO — the answer is the current input byte, not
    a previous one. This tests pure input→output information flow through the model.
    """
    data = bytearray()
    while len(data) < TARGET:
        b = random.randint(0, 255)
        data.extend(bytes([b]) * COPY_RUN_LENGTH)
    with open(path, 'wb') as f:
        f.write(data[:TARGET])


def gen_echo(path):
    """Next block = current block (identity). Each block repeated ECHO_REPEAT times.

    With REPEAT=8, 7/8 = 87.5% of consecutive pairs are identity (same block),
    and 1/8 = 12.5% are random transitions (between different blocks).
    This gives the model strong gradient signal for learning the copy operation.
    """
    data = bytearray()
    while len(data) < TARGET:
        block = bytes(random.randint(0, 255) for _ in range(BLOCK))
        for _ in range(ECHO_REPEAT):
            data.extend(block)
    with open(path, 'wb') as f:
        f.write(data[:TARGET])


def gen_not(path):
    """Next block = bitwise NOT of current. Alternating [A][~A][A][~A]... pattern.

    Every position is deterministic: even positions predict NOT, odd positions predict NOT-back.
    100% of positions have learnable signal (no random transitions).
    """
    data = bytearray()
    while len(data) < TARGET:
        block = bytes(random.randint(0, 255) for _ in range(BLOCK))
        inv = bytes(b ^ 0xFF for b in block)
        # Repeat the A/~A pair 4 times to give consistent signal
        for _ in range(4):
            data.extend(block)
            data.extend(inv)
    with open(path, 'wb') as f:
        f.write(data[:TARGET])


def gen_shift(path):
    """Sequence of byte-rotations of a seed block."""
    data = bytearray()
    while len(data) < TARGET:
        seed = [random.randint(0, 255) for _ in range(BLOCK)]
        for rot in range(BLOCK):
            rotated = seed[rot:] + seed[:rot]
            data.extend(bytes(rotated))
    with open(path, 'wb') as f:
        f.write(data[:TARGET])


def gen_count(path):
    """BLOCK-byte big-endian counter, incrementing."""
    data = bytearray()
    max_val = 2 ** (BLOCK * 8)
    counter = random.randint(0, max_val - 1)
    while len(data) < TARGET:
        data.extend(counter.to_bytes(BLOCK, 'big'))
        counter = (counter + 1) % max_val
    with open(path, 'wb') as f:
        f.write(data[:TARGET])


def gen_add(path):
    """[A(8B)][B(8B)][(A+B)%256 per byte(8B)][|A-B| per byte(8B)]."""
    data = bytearray()
    while len(data) < TARGET:
        a = bytes(random.randint(0, 255) for _ in range(8))
        b = bytes(random.randint(0, 255) for _ in range(8))
        add_r = bytes((ai + bi) % 256 for ai, bi in zip(a, b))
        sub_r = bytes(abs(ai - bi) for ai, bi in zip(a, b))
        data.extend(a + b + add_r + sub_r)
    with open(path, 'wb') as f:
        f.write(data[:TARGET])


def gen_fib(path):
    """Fibonacci bytes: block[t][i] = (block[t-1][i] + block[t-2][i]) % 256."""
    data = bytearray()
    while len(data) < TARGET:
        prev2 = [random.randint(0, 255) for _ in range(BLOCK)]
        prev1 = [random.randint(0, 255) for _ in range(BLOCK)]
        data.extend(bytes(prev2))
        data.extend(bytes(prev1))
        for _ in range(30):
            curr = [(a + b) % 256 for a, b in zip(prev1, prev2)]
            data.extend(bytes(curr))
            prev2, prev1 = prev1, curr
    with open(path, 'wb') as f:
        f.write(data[:TARGET])


def gen_delay_echo(path):
    """[A][filler*N][A] — delayed echo that requires working memory.

    Structure per group:
        - 1 original block A (random)
        - N filler blocks (random, unpredictable)
        - 1 echo of A (identical copy)

    When the model's sliding window lands on [filler...][A_echo], it can only
    predict A_echo if it memorized A from before the filler gap.

    With DELAY_FILLER_BLOCKS=4 and BLOCK=16: gap = 64 bytes.
    For seq_len=62 (num_bits=8), the original A is outside the attention window
    when the model reaches the echo — LCX memory is required.
    """
    data = bytearray()
    n_filler = DELAY_FILLER_BLOCKS
    while len(data) < TARGET:
        original = bytes(random.randint(0, 255) for _ in range(BLOCK))
        data.extend(original)
        for _ in range(n_filler):
            filler = bytes(random.randint(0, 255) for _ in range(BLOCK))
            data.extend(filler)
        data.extend(original)  # echo
    with open(path, 'wb') as f:
        f.write(data[:TARGET])


def gen_denoise(path):
    """[noisy_A][clean_A] — model reads corrupted version, predicts clean original.

    Structure per pair:
        - 1 noisy block (A with random bit flips at DENOISE_FLIP_PROB per bit)
        - 1 clean block A (the uncorrupted original)

    At the noisy→clean boundary, the next-byte prediction task becomes:
    "given a corrupted byte sequence, predict what the clean version is."

    Within the noisy block, the model learns noisy-byte statistics.
    Within the clean block following it, the model can verify its denoising.
    Memory helps: if the model stores the noisy version in LCX, it can
    cross-reference during the clean block.
    """
    data = bytearray()
    while len(data) < TARGET:
        clean = bytes(random.randint(0, 255) for _ in range(BLOCK))
        # Flip bits with probability DENOISE_FLIP_PROB
        noisy = bytearray(clean)
        for i in range(len(noisy)):
            for bit in range(8):
                if random.random() < DENOISE_FLIP_PROB:
                    noisy[i] ^= (1 << bit)
        data.extend(bytes(noisy))
        data.extend(clean)
    with open(path, 'wb') as f:
        f.write(data[:TARGET])


# --- Metadata ---

GENERATORS = {
    'constant256.traindat': {
        'fn': gen_constant,
        'meta': {
            'task': 'constant',
            'generator': 'gen_constant',
            'block_size': 0,
            'tier': 0,
            'notes': 'All zeros. Sanity check: model learns output bias only.',
        },
    },
    'copy_echo256.traindat': {
        'fn': gen_copy_echo,
        'meta': {
            'task': 'copy_echo',
            'generator': 'gen_copy_echo',
            'run_length': COPY_RUN_LENGTH,
            'tier': 0,
            'notes': f'Each byte repeated {COPY_RUN_LENGTH}x. Copy current byte = ~99% predictable.',
        },
    },
    'echo256.traindat': {
        'fn': gen_echo,
        'meta': {
            'task': 'echo',
            'generator': 'gen_echo',
            'block_size': BLOCK,
            'tier': 1,
            'notes': f'Block repeated {ECHO_REPEAT}x, {(ECHO_REPEAT-1)/ECHO_REPEAT*100:.1f}% identity',
        },
    },
    'not256.traindat': {
        'fn': gen_not,
        'meta': {
            'task': 'bitwise_not',
            'generator': 'gen_not',
            'block_size': BLOCK,
            'tier': 1,
            'notes': 'Alternating [A][~A] pairs, 100% deterministic',
        },
    },
    'shift256.traindat': {
        'fn': gen_shift,
        'meta': {
            'task': 'byte_rotation',
            'generator': 'gen_shift',
            'block_size': BLOCK,
            'tier': 2,
            'notes': 'Byte-rotations of seed block, cross-byte pattern',
        },
    },
    'count256.traindat': {
        'fn': gen_count,
        'meta': {
            'task': 'counter',
            'generator': 'gen_count',
            'block_size': BLOCK,
            'tier': 2,
            'notes': 'Big-endian counter incrementing by 1',
        },
    },
    'add256.traindat': {
        'fn': gen_add,
        'meta': {
            'task': 'arithmetic',
            'generator': 'gen_add',
            'block_size': 8,
            'tier': 3,
            'notes': '[A(8B)][B(8B)][(A+B)%256(8B)][|A-B|(8B)]',
        },
    },
    'fib256.traindat': {
        'fn': gen_fib,
        'meta': {
            'task': 'fibonacci',
            'generator': 'gen_fib',
            'block_size': BLOCK,
            'tier': 4,
            'notes': 'block[t] = (block[t-1] + block[t-2]) % 256, 30-step chains',
        },
    },
    'delay_echo256.traindat': {
        'fn': gen_delay_echo,
        'meta': {
            'task': 'delay_echo',
            'generator': 'gen_delay_echo',
            'block_size': BLOCK,
            'tier': 5,
            'delay_filler_blocks': DELAY_FILLER_BLOCKS,
            'gap_bytes': DELAY_FILLER_BLOCKS * BLOCK,
            'notes': f'[A][filler*{DELAY_FILLER_BLOCKS}][A] — gap={DELAY_FILLER_BLOCKS * BLOCK}B, requires LCX memory when gap > seq_len',
        },
    },
    'denoise256.traindat': {
        'fn': gen_denoise,
        'meta': {
            'task': 'denoise',
            'generator': 'gen_denoise',
            'block_size': BLOCK,
            'tier': 5,
            'flip_prob': DENOISE_FLIP_PROB,
            'notes': f'[noisy_A][clean_A] — {DENOISE_FLIP_PROB*100:.0f}% bit flip rate, tests pattern recovery',
        },
    },
}


if __name__ == '__main__':
    random.seed(42)
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, entry in GENERATORS.items():
        path = os.path.join(OUT_DIR, name)
        print(f"Generating {name}...")
        entry['fn'](path)
        size = os.path.getsize(path)
        print(f"  -> {size:,} bytes")

        # Write .meta.json sidecar (v2.0 format)
        meta = dict(entry['meta'])
        meta['seed'] = 42
        meta['target_bytes'] = TARGET
        meta['actual_bytes'] = size
        meta['version'] = '2.0'
        meta['date'] = datetime.datetime.now().isoformat()[:19]
        meta['pairs_estimate'] = size // (BLOCK * 2)
        meta['script'] = 'generate_traindat_suite.py'
        meta_path = path.replace('.traindat', '.meta.json')
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"  -> {os.path.basename(meta_path)}")
    print("Done!")
