"""
Generate structured .traindat files for 256-bit Diamond Code training.

Format: Each block = 32 raw bytes, aligned to num_bits=256:
  Bytes  0-7:  operand A (8 random bytes)
  Bytes  8-15: operand B (8 random bytes)
  Bytes 16-23: A XOR B (bytewise)
  Bytes 24-31: A AND B (bytewise)

All 256 bits per position are structured and learnable.
The model can discover: bits 128-191 = XOR(bits 0-63, bits 64-127)
                         bits 192-255 = AND(bits 0-63, bits 64-127)

Usage:
    python generate_logic_traindat.py --output data/traindat/ --size_mb 1
"""

import random
import shutil
import argparse
from pathlib import Path

BLOCK_SIZE = 32  # bytes per position (must match num_bits // 8)


def generate_structured_file(output_path: str, target_bytes: int, seed: int = 42):
    """Generate a .traindat file with 32-byte structured blocks."""
    random.seed(seed)
    data = bytearray()

    while len(data) < target_bytes:
        # 8 random bytes for operand A
        a = bytes(random.randint(0, 255) for _ in range(8))
        # 8 random bytes for operand B
        b = bytes(random.randint(0, 255) for _ in range(8))
        # 8 bytes: A XOR B (bytewise)
        xor_result = bytes(a_i ^ b_i for a_i, b_i in zip(a, b))
        # 8 bytes: A AND B (bytewise)
        and_result = bytes(a_i & b_i for a_i, b_i in zip(a, b))

        data.extend(a)
        data.extend(b)
        data.extend(xor_result)
        data.extend(and_result)

    with open(output_path, 'wb') as f:
        f.write(bytes(data))

    n_blocks = len(data) // BLOCK_SIZE
    print(f"  {Path(output_path).name}: {len(data):,} bytes, {n_blocks:,} blocks of {BLOCK_SIZE}B")


def main():
    parser = argparse.ArgumentParser(description='Generate .traindat files for 256-bit training')
    parser.add_argument('--output', type=str, default='data/traindat/',
                        help='Output directory')
    parser.add_argument('--size_mb', type=float, default=1.0,
                        help='Target size per file in MB')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_bytes = int(args.size_mb * 1024 * 1024)

    print(f"Generating .traindat files in {output_dir}/ (32-byte structured blocks)")
    print(f"Target: ~{args.size_mb}MB per file")
    print(f"Format: 32B/block = [A(8B)][B(8B)][A^B(8B)][A&B(8B)]")
    print()

    # Single structured file containing all operations
    generate_structured_file(
        str(output_dir / "logic256.traindat"),
        target_bytes,
        seed=args.seed,
    )

    # Copy shakespeare as .traindat
    shakespeare_src = Path(__file__).parent / "data" / "shakespeare.txt"
    if shakespeare_src.exists():
        shakespeare_dst = output_dir / "shakespeare.traindat"
        shutil.copy2(shakespeare_src, shakespeare_dst)
        print(f"  shakespeare.traindat: {shakespeare_src.stat().st_size:,} bytes (copied)")
    else:
        print(f"  WARNING: {shakespeare_src} not found, skipping")

    print(f"\nDone. {len(list(output_dir.glob('*.traindat')))} files ready.")


if __name__ == "__main__":
    main()
