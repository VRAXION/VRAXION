"""Generate curated traindat difficulty ladder.

Tier 1 (sanity): echo256, not256 — 100% predictable
Tier 2 (simple): shift256, count256 — cross-byte patterns
Tier 3 (logic):  add256 — arithmetic (logic256 already exists)
Tier 4 (memory): fib256 — two-step dependency
"""
import os
import random

TARGET = 1_048_576  # 1MB each
BLOCK = 32
OUT_DIR = "data/traindat"


def gen_echo(path):
    """Next block = current block (identity)."""
    data = bytearray()
    while len(data) < TARGET:
        block = bytes(random.randint(0, 255) for _ in range(BLOCK))
        data.extend(block)
        data.extend(block)
    with open(path, 'wb') as f:
        f.write(data[:TARGET])


def gen_not(path):
    """Next block = bitwise NOT of current."""
    data = bytearray()
    while len(data) < TARGET:
        block = bytes(random.randint(0, 255) for _ in range(BLOCK))
        inv = bytes(b ^ 0xFF for b in block)
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
    """32-byte big-endian counter, incrementing."""
    data = bytearray()
    counter = random.randint(0, 2**256 - 1)
    while len(data) < TARGET:
        data.extend(counter.to_bytes(BLOCK, 'big'))
        counter = (counter + 1) % (2**256)
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


if __name__ == '__main__':
    random.seed(42)
    os.makedirs(OUT_DIR, exist_ok=True)
    generators = {
        'echo256.traindat': gen_echo,
        'not256.traindat': gen_not,
        'shift256.traindat': gen_shift,
        'count256.traindat': gen_count,
        'add256.traindat': gen_add,
        'fib256.traindat': gen_fib,
    }
    for name, fn in generators.items():
        path = os.path.join(OUT_DIR, name)
        print(f"Generating {name}...")
        fn(path)
        print(f"  -> {os.path.getsize(path):,} bytes")
    print("Done!")
