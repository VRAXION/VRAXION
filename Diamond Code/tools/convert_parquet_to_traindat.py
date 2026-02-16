"""
Convert FineWeb-Edu parquet files to .traindat format.

Reads parquet text data, shuffles documents, concatenates as raw UTF-8 bytes
with \n\n separators, writes flat binary .traindat file.

Usage:
    python tools/convert_parquet_to_traindat.py
"""

import os
import random
import pandas as pd

PARQUET_DIR = r"S:\AI\MESSY TRAINING DATA - INPUT ONLY\Fineweb edu 10B"
OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "traindat", "fineweb.traindat"
)
TARGET_BYTES = 104_857_600  # 100 MB


def main():
    # Use first parquet file only (726K rows, way more than needed for 100 MB)
    parquet_file = os.path.join(PARQUET_DIR, "000_00000.parquet")
    print(f"Reading {parquet_file}...")
    df = pd.read_parquet(parquet_file, columns=["text", "int_score"])
    print(f"  {len(df)} rows loaded")

    # Filter for educational quality
    df = df[df["int_score"] >= 3]
    print(f"  {len(df)} rows after int_score >= 3 filter")

    # Shuffle
    texts = df["text"].tolist()
    random.shuffle(texts)
    print(f"  Shuffled {len(texts)} documents")

    # Concatenate as UTF-8 bytes until we hit target size
    buf = bytearray()
    doc_count = 0
    for text in texts:
        encoded = text.encode("utf-8", errors="replace")
        buf.extend(encoded)
        buf.extend(b"\n\n")
        doc_count += 1
        if len(buf) >= TARGET_BYTES:
            break

    # Truncate to exact target
    buf = buf[:TARGET_BYTES]

    # Write
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        f.write(buf)

    print(f"\nDone:")
    print(f"  Documents: {doc_count}")
    print(f"  Output:    {OUTPUT_PATH}")
    print(f"  Size:      {len(buf):,} bytes ({len(buf) / 1_048_576:.1f} MB)")


if __name__ == "__main__":
    main()
