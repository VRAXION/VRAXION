"""
Generate Golden Disc .traindat files for Diamond Code training.

Golden Disc format: structured byte streams with ASCII control separators.
See golden_disc.py for format spec.

Usage:
    python generate_golden_discs.py              # generate all
    python generate_golden_discs.py add1         # generate just add1
    python generate_golden_discs.py add1 sub1    # generate specific discs
"""

import os
import sys
import random
import time

from golden_disc import FS, RS, US, encode_type_byte, validate_content

TARGET = 104_857_600  # 100 MB
TARGET_GENESIS = 63_897_600  # 61 MB (phi ratio)
OUT_DIR = "data/traindat"


def gen_gold_math_add1(path):
    """Single-digit addition: '3+4' -> '7'.

    Type byte: 0x01 (numeric, easy)
    All 100 valid pairs (0+0 through 9+9), uniformly sampled.
    ~16M rows in 100MB, ~10 examples per 62-byte window.
    """
    type_byte = encode_type_byte(modality=0x0, complexity=0x1)
    data = bytearray()
    data.append(FS)
    data.append(type_byte)
    while len(data) < TARGET:
        a = random.randint(0, 9)
        b = random.randint(0, 9)
        c = a + b
        data.append(RS)
        data.extend(f"{a}+{b}".encode('ascii'))
        data.append(US)
        data.extend(str(c).encode('ascii'))
    # Truncate on RS boundary (never mid-row)
    last_rs = data.rfind(RS, 0, TARGET)
    if last_rs > 0:
        data = data[:last_rs]
    with open(path, 'wb') as f:
        f.write(data)


def gen_gold_math_sub1(path):
    """Single-digit subtraction (non-negative): '7-3' -> '4'.

    Type byte: 0x01 (numeric, easy)
    Only pairs where a >= b (result >= 0).
    """
    type_byte = encode_type_byte(modality=0x0, complexity=0x1)
    data = bytearray()
    data.append(FS)
    data.append(type_byte)
    while len(data) < TARGET:
        a = random.randint(0, 9)
        b = random.randint(0, a)  # b <= a
        c = a - b
        data.append(RS)
        data.extend(f"{a}-{b}".encode('ascii'))
        data.append(US)
        data.extend(str(c).encode('ascii'))
    last_rs = data.rfind(RS, 0, TARGET)
    if last_rs > 0:
        data = data[:last_rs]
    with open(path, 'wb') as f:
        f.write(data)


def gen_gold_math_mul1(path):
    """Single-digit multiplication: '3*4' -> '12'.

    Type byte: 0x02 (numeric, easy+)
    All 100 pairs (0*0 through 9*9).
    """
    type_byte = encode_type_byte(modality=0x0, complexity=0x2)
    data = bytearray()
    data.append(FS)
    data.append(type_byte)
    while len(data) < TARGET:
        a = random.randint(0, 9)
        b = random.randint(0, 9)
        c = a * b
        data.append(RS)
        data.extend(f"{a}*{b}".encode('ascii'))
        data.append(US)
        data.extend(str(c).encode('ascii'))
    last_rs = data.rfind(RS, 0, TARGET)
    if last_rs > 0:
        data = data[:last_rs]
    with open(path, 'wb') as f:
        f.write(data)


def gen_gold_logic_bool(path):
    """Boolean logic: 'T AND F' -> 'F', 'T OR T' -> 'T', 'NOT T' -> 'F'.

    Type byte: 0x20 (logic, trivial)
    Three operations: AND, OR, NOT.
    """
    type_byte = encode_type_byte(modality=0x2, complexity=0x0)
    vals = {True: b'T', False: b'F'}
    data = bytearray()
    data.append(FS)
    data.append(type_byte)
    ops = ['AND', 'OR', 'NOT']
    while len(data) < TARGET:
        op = random.choice(ops)
        if op == 'NOT':
            a = random.choice([True, False])
            q = b'NOT ' + vals[a]
            ans = vals[not a]
        else:
            a = random.choice([True, False])
            b_val = random.choice([True, False])
            if op == 'AND':
                result = a and b_val
            else:
                result = a or b_val
            q = vals[a] + b' ' + op.encode() + b' ' + vals[b_val]
            ans = vals[result]
        data.append(RS)
        data.extend(q)
        data.append(US)
        data.extend(ans)
    last_rs = data.rfind(RS, 0, TARGET)
    if last_rs > 0:
        data = data[:last_rs]
    with open(path, 'wb') as f:
        f.write(data)


def gen_gold_origin_echo(path):
    """Dot echo: 'ooo' -> 'ooo'. Mirror test for proof of life.

    Type byte: 0x00 (numeric, trivial)
    Dot count 1-9, uniform random. Only 5 unique bytes in output.
    ~5.3M rows in 61MB, 3-15 complete rows per 62-byte window.
    """
    type_byte = encode_type_byte(modality=0x0, complexity=0x0)
    data = bytearray()
    data.append(FS)
    data.append(type_byte)
    while len(data) < TARGET_GENESIS:
        n = random.randint(1, 9)
        dots = b'o' * n
        data.append(RS)
        data.extend(dots)
        data.append(US)
        data.extend(dots)
    last_rs = data.rfind(RS, 0, TARGET_GENESIS)
    if last_rs > 0:
        data = data[:last_rs]
    with open(path, 'wb') as f:
        f.write(data)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
GENERATORS = {
    'gold_origin_echo': ('gold_origin_echo.traindat', gen_gold_origin_echo),
    'gold_math_add1': ('gold_math_add1.traindat', gen_gold_math_add1),
    'gold_math_sub1': ('gold_math_sub1.traindat', gen_gold_math_sub1),
    'gold_math_mul1': ('gold_math_mul1.traindat', gen_gold_math_mul1),
    'gold_logic_bool': ('gold_logic_bool.traindat', gen_gold_logic_bool),
}


def main():
    random.seed(42)
    os.makedirs(OUT_DIR, exist_ok=True)

    # Filter to requested discs (or all)
    requested = sys.argv[1:] if len(sys.argv) > 1 else list(GENERATORS.keys())
    for name in requested:
        if name not in GENERATORS:
            print(f"Unknown disc: {name}")
            print(f"Available: {', '.join(GENERATORS.keys())}")
            sys.exit(1)

    for name in requested:
        fname, gen_fn = GENERATORS[name]
        path = os.path.join(OUT_DIR, fname)
        print(f"Generating {fname}...", end=' ', flush=True)
        t0 = time.time()
        gen_fn(path)
        elapsed = time.time() - t0
        size = os.path.getsize(path)
        print(f"{size:,} bytes in {elapsed:.1f}s")

    print("Done!")


if __name__ == '__main__':
    main()
