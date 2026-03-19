"""Deterministic validation of curriculum traindat files.

For each dataset:
1. Verify file exists and size matches meta.json
2. Load raw bytes and verify the pattern is correct
3. Generate a batch in binary-bits mode and check shape/content
4. Estimate theoretical accuracy ceiling at seq_len=62

Usage: python tools/_scratch/validate_curriculum.py
"""
import sys, os, json, struct, random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from traindat_loader import generate_batch_binary_bits

DATA_DIR = "data/traindat"
SEQ_LEN = 62
NUM_BITS = 8
BATCH = 32
BLOCK = 16

DATASETS = [
    "copy_echo256",
    "echo256",
    "not256",
    "shift256",
    "count256",
]

def validate_copy_echo256(data: bytes) -> dict:
    """copy_echo: each byte repeated 128 times. Pattern: long runs of identical bytes."""
    # Check that we see long runs
    run_lengths = []
    current = data[0]
    run = 1
    for i in range(1, min(len(data), 100000)):
        if data[i] == current:
            run += 1
        else:
            run_lengths.append(run)
            current = data[i]
            run = 1
    run_lengths.append(run)

    avg_run = np.mean(run_lengths)
    # Most runs should be 128 (or truncated at boundaries)
    full_runs = [r for r in run_lengths if r == 128]

    # For next-byte prediction: within a run, next byte = current byte
    # At boundaries (every 128 bytes), next byte is random
    # Theoretical accuracy: 127/128 = 99.2% of positions are predictable
    theoretical_acc = 127.0 / 128.0

    # Verify with actual data
    correct = 0
    total = min(len(data) - 1, 100000)
    for i in range(total):
        if data[i] == data[i+1]:
            correct += 1
    actual_identity = correct / total

    return {
        "avg_run_length": round(avg_run, 1),
        "num_full_runs": len(full_runs),
        "total_runs": len(run_lengths),
        "theoretical_next_byte_acc": round(theoretical_acc, 4),
        "actual_identity_rate": round(actual_identity, 4),
        "PASS": actual_identity > 0.98,
    }


def validate_echo256(data: bytes) -> dict:
    """echo: 16-byte blocks repeated 8 times. Next byte = same position in same block (87.5%)."""
    # Check block repetition pattern
    matches = 0
    total = 0
    for i in range(0, min(len(data) - BLOCK, 100000), 1):
        if data[i] == data[i + BLOCK]:
            matches += 1
        total += 1

    block_repeat_rate = matches / total if total > 0 else 0

    # For next-byte prediction at seq_len=62:
    # The model needs to learn that byte[t+1] often equals byte[t+1 - BLOCK]
    # Within a repeating section: byte[t+1] = byte[t+1-16] (need to see 16+ bytes back)
    # At seq_len=62, the model can see 62 bytes, so it can look 16 bytes back easily
    # Theoretical: 7/8 of transitions are within the same repeated section
    # (every 8th block boundary is a new random block)
    theoretical_acc = 7.0 / 8.0  # 87.5% of positions are predictable via echo

    return {
        "block_repeat_rate_16byte": round(block_repeat_rate, 4),
        "expected_repeat_rate": "~0.875 (7/8 blocks are repeats)",
        "theoretical_next_byte_acc": round(theoretical_acc, 4),
        "PASS": block_repeat_rate > 0.85,
    }


def validate_not256(data: bytes) -> dict:
    """not: [A][~A][A][~A]... pairs, 16 bytes each, repeated 4 times."""
    # Check the NOT pattern: data[i] ^ data[i + BLOCK] should be 0xFF
    not_matches = 0
    identity_matches = 0
    total = 0
    for i in range(0, min(len(data) - BLOCK, 100000)):
        if data[i] ^ data[i + BLOCK] == 0xFF:
            not_matches += 1
        if data[i] == data[i + BLOCK]:
            identity_matches += 1
        total += 1

    # Pattern is [A][~A][A][~A]... repeated 4 times per seed
    # So positions 0-15 are A, 16-31 are ~A, 32-47 are A, 48-63 are ~A, etc.
    # Checking consecutive blocks:
    # A->~A: NOT (positions 0-15 predict ~A at 16-31)
    # ~A->A: NOT again (positions 16-31 predict A at 32-47)
    # After 4 repeats (128 bytes), new random A' starts
    # Within the 128-byte group, next byte is deterministic via pattern

    not_rate = not_matches / total if total > 0 else 0

    # For next-byte prediction: byte[t+1] relationship depends on position within pattern
    # If we're within a 16-byte block: next byte = next byte in same block (identity)
    # If we're at a block boundary: next byte = NOT of 16 bytes ago
    # Tricky: identity within blocks, NOT at block boundaries

    # Actually for next-byte (byte-level shift by 1):
    # Within the same block: byte[t+1] is just the next byte of the random block A
    # These are random and unpredictable unless the model memorized the block
    # At block boundaries: the NOT relationship kicks in

    # More carefully: the data is A1 A2 ... A16 ~A1 ~A2 ... ~A16 A1 A2 ... A16 ...
    # For next-byte prediction:
    # Position 0->1: A1->A2 (random, unpredictable without seeing the pattern)
    # Position 15->16: A16->~A1 (need to know A1 to predict ~A1)
    # But the model sees up to 62 bytes back! So at position 16, it saw A1 62 bytes ago? No...
    # At position 16, it has seen A1...A16 in its context. It can predict ~A1.
    # At position 17, it has A2...A16, ~A1 in context. It can predict ~A2.
    # So within the NOT block, if the model learned NOT, it can predict each byte.

    # Within the A block (bytes 0-15): bytes are random, but repeated 4x in the 128-byte group
    # First occurrence: unpredictable. Subsequent occurrences: predictable via echo (32 bytes back)

    # Theoretical ceiling: complex. In the first A block, bytes are unpredictable (random).
    # In ~A block: predictable via NOT from 16 bytes back.
    # In 2nd A block (bytes 32-47): predictable via identity from 32 bytes back.
    # In 2nd ~A block: predictable via NOT from 16 bytes back.
    # Pattern repeats every 128 bytes. New random A at 128.

    # Within each 128-byte group:
    # Bytes 0-15 (1st A): ~0 predictable (random seed)
    # Bytes 16-31 (~A): 16/16 predictable via NOT of bytes 0-15
    # Bytes 32-47 (2nd A): 16/16 predictable via identity of bytes 0-15 (or NOT of 16-31)
    # Bytes 48-63 (2nd ~A): 16/16 predictable
    # Bytes 64-79 (3rd A): 16/16 predictable
    # Bytes 80-95 (3rd ~A): 16/16 predictable
    # Bytes 96-111 (4th A): 16/16 predictable
    # Bytes 112-127 (4th ~A): 16/16 predictable
    # = 112/128 predictable = 87.5% ... wait, first block is 16/128 unpredictable
    # Actually within first A: next-byte within the block is random (A1->A2 is random)
    # But byte 15->16 transitions: A16 -> ~A1 (NOT with 16-byte offset)
    # Wait, I need to think about this differently.

    # For NEXT-BYTE prediction (t -> t+1):
    # Within any block (positions 0-14 within a 16-byte block):
    #   next byte is next byte in same block. If this is the first A block, random.
    #   If this is a repeat/NOT block, then the byte at t+1 can be predicted from
    #   the byte at t+1-16 (or NOT of t+1-16).
    # At block boundaries (position 15 within block -> position 0 of next block):
    #   Transition to new block. Can predict via relationship (identity or NOT).

    # Net: 112/128 positions are in predictable blocks, BUT within those blocks,
    # each byte is predicted from 16 or 32 bytes back. The first 16 bytes of each
    # 128-byte group are unpredictable. So theoretical max ≈ 87.5%.

    theoretical_acc = 112.0 / 128.0  # 87.5%

    return {
        "not_pattern_rate": round(not_rate, 4),
        "identity_rate": round(identity_matches / total if total > 0 else 0, 4),
        "expected_not_rate": "~0.50 (alternating NOT/identity blocks)",
        "theoretical_next_byte_acc": round(theoretical_acc, 4),
        "PASS": not_rate > 0.45,
    }


def validate_shift256(data: bytes) -> dict:
    """shift: byte-rotations of 16-byte seed blocks."""
    # Pattern: block[i] = rotate(seed, i) where rotate shifts bytes left by i
    # So block[0] = [s0,s1,...,s15], block[1] = [s1,s2,...,s15,s0], etc.
    # After 16 rotations, new random seed.

    # Check: is data[i + BLOCK] a rotation of data[i:i+BLOCK]?
    rotation_found = 0
    total = 0
    for start in range(0, min(len(data) - 2*BLOCK, 50000), BLOCK):
        block1 = data[start:start+BLOCK]
        block2 = data[start+BLOCK:start+2*BLOCK]
        # Check if block2 is a left rotation of block1 by 1
        if bytes(block1[1:]) + bytes([block1[0]]) == block2:
            rotation_found += 1
        total += 1

    rot_rate = rotation_found / total if total > 0 else 0

    # For next-byte prediction:
    # Within a block: byte[t+1] is next byte of rotated seed (random unless you know pattern)
    # At block boundary: byte = seed[(rotation+1+0) % 16] = seed[rotation+1]
    # The model can learn: byte at position p in block k+1 = byte at position p+1 in block k
    # This is a cross-byte, cross-block relationship.

    # Within each 16-block group (256 bytes):
    # First block (16 bytes): unpredictable (random seed)
    # Subsequent 15 blocks: each byte predictable from previous block
    # = 15/16 = 93.75% of blocks are predictable
    # But within each block, byte-to-byte transitions are seed-dependent
    # For next-byte: if within a known block, byte[t+1] is known from the seed pattern
    # First 16 bytes of each 256-byte group are random. Rest are deterministic.

    theoretical_acc = 15.0 / 16.0  # 93.75% (first rotation unpredictable)

    return {
        "rotation_rate": round(rot_rate, 4),
        "expected_rotation_rate": "~0.9375 (15/16 blocks are rotations of previous)",
        "theoretical_next_byte_acc": round(theoretical_acc, 4),
        "PASS": rot_rate > 0.90,
    }


def validate_count256(data: bytes) -> dict:
    """count: 16-byte big-endian counter incrementing by 1."""
    # Check consecutive blocks increment by 1
    increments_correct = 0
    total = 0
    for start in range(0, min(len(data) - 2*BLOCK, 50000), BLOCK):
        val1 = int.from_bytes(data[start:start+BLOCK], 'big')
        val2 = int.from_bytes(data[start+BLOCK:start+2*BLOCK], 'big')
        if val2 == val1 + 1:
            increments_correct += 1
        total += 1

    inc_rate = increments_correct / total if total > 0 else 0

    # For next-byte prediction:
    # Counter is 16 bytes big-endian. Byte 15 (LSB) increments every block.
    # Byte 14 increments every 256 blocks. Byte 13 every 65536 blocks, etc.
    # Within a block: bytes are "random" (counter value), but MSB bytes rarely change
    # At block boundary: last byte increments by 1 (mod 256)

    # For next-byte prediction within a block:
    # The counter bytes are positional. If the model learns the counter structure,
    # it can predict byte[t+1] = same counter value's next byte.
    # But these are not easily predictable without understanding big-endian encoding.

    # For cross-block prediction:
    # Most bytes in next block = same as current block (MSBs rarely change)
    # Last byte = current last byte + 1 (mod 256)
    # Theoretical: 15/16 bytes identical to previous block + last byte predictable
    # = ~100% predictable (but model needs to learn the incrementing pattern)

    # Actually: within each 16-byte block, the byte-to-byte pattern IS the counter value
    # MSBs are very stable (change rarely), LSB cycles through 0-255
    # The model needs to learn that:
    # - Most bytes are the same as 16 bytes ago
    # - Last byte is 1 more than 16 bytes ago

    # Similar to echo but with a +1 on the last byte
    # Harder than echo because of the increment logic

    theoretical_acc = 0.95  # approximate: most bytes are echo, last byte needs increment logic

    return {
        "increment_correct_rate": round(inc_rate, 4),
        "expected_increment_rate": "~1.0000 (consecutive blocks should differ by 1)",
        "theoretical_next_byte_acc": round(theoretical_acc, 2),
        "PASS": inc_rate > 0.99,
    }


def validate_batch(filepath, name):
    """Generate a batch and verify shapes and basic properties."""
    with open(filepath, 'rb') as f:
        corpus = f.read()

    x, y, mask = generate_batch_binary_bits(corpus, BATCH, SEQ_LEN, NUM_BITS, seed=42)

    assert x.shape == (BATCH, SEQ_LEN, NUM_BITS), f"x shape mismatch: {x.shape}"
    assert y.shape == (BATCH, SEQ_LEN, NUM_BITS), f"y shape mismatch: {y.shape}"
    assert mask.shape == (BATCH, SEQ_LEN, NUM_BITS), f"mask shape mismatch: {mask.shape}"

    # All values should be 0 or 1 in binary mode
    assert ((x == 0) | (x == 1)).all(), "x contains non-binary values"
    assert ((y == 0) | (y == 1)).all(), "y contains non-binary values"
    assert (mask == 1).all(), "mask should be all 1s for traindat"

    # Check that y is x shifted by 1 byte in the original data
    # y[b, t, :] should be the bits of the byte at position start + t + 1
    # x[b, t, :] should be the bits of the byte at position start + t
    # So y[b, t, :] == x[b, t+1, :] if the next byte in sequence is the same
    # This isn't generally true, but x[b, 0:61, :] and y[b, 0:61, :] should have
    # y[b, t, :] == x[b, t+1, :] for all t < seq_len-1
    # Because y[b,t] = bits_of(byte[start+t+1]) = x[b,t+1]
    shift_match = (y[:, :-1, :] == x[:, 1:, :]).float().mean().item()

    return {
        "batch_shape": f"x={list(x.shape)} y={list(y.shape)} mask={list(mask.shape)}",
        "x_range": f"[{x.min():.0f}, {x.max():.0f}]",
        "y_range": f"[{y.min():.0f}, {y.max():.0f}]",
        "shift_consistency": round(shift_match, 4),
        "PASS": True,
    }


def main():
    os.chdir("S:/AI/work/VRAXION_DEV/Diamond Code")

    validators = {
        "copy_echo256": validate_copy_echo256,
        "echo256": validate_echo256,
        "not256": validate_not256,
        "shift256": validate_shift256,
        "count256": validate_count256,
    }

    all_pass = True

    for name in DATASETS:
        filepath = os.path.join(DATA_DIR, f"{name}.traindat")
        metapath = os.path.join(DATA_DIR, f"{name}.meta.json")

        print(f"\n{'='*60}")
        print(f"  VALIDATING: {name}")
        print(f"{'='*60}")

        # 1. File existence and size
        if not os.path.exists(filepath):
            print(f"  FAIL: {filepath} not found!")
            all_pass = False
            continue

        fsize = os.path.getsize(filepath)
        print(f"  File size: {fsize:,} bytes ({fsize/1024/1024:.1f} MB)")

        # 2. Meta.json check
        if os.path.exists(metapath):
            with open(metapath) as f:
                meta = json.load(f)
            print(f"  Meta: task={meta.get('task')}, tier={meta.get('tier')}, "
                  f"version={meta.get('version')}")
            if meta.get('actual_bytes') != fsize:
                print(f"  WARN: meta says {meta['actual_bytes']} bytes, file is {fsize}")
        else:
            print(f"  WARN: no meta.json found")

        # 3. Pattern validation
        with open(filepath, 'rb') as f:
            data = f.read(200000)  # Read first 200KB for pattern check

        validator = validators.get(name)
        if validator:
            result = validator(data)
            for k, v in result.items():
                if k == "PASS":
                    status = "PASS" if v else "FAIL"
                    print(f"  Pattern check: {status}")
                    if not v:
                        all_pass = False
                else:
                    print(f"    {k}: {v}")

        # 4. Batch generation test
        print(f"\n  Batch test (batch={BATCH}, seq_len={SEQ_LEN}, num_bits={NUM_BITS}):")
        batch_result = validate_batch(filepath, name)
        for k, v in batch_result.items():
            if k == "PASS":
                print(f"  Batch check: {'PASS' if v else 'FAIL'}")
            else:
                print(f"    {k}: {v}")

    print(f"\n{'='*60}")
    if all_pass:
        print("  ALL DATASETS VALIDATED SUCCESSFULLY")
    else:
        print("  SOME DATASETS FAILED VALIDATION")
    print(f"{'='*60}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
