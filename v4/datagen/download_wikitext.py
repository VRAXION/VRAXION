"""Download WikiText-103-raw from HuggingFace and convert to .traindat + .mask.

Uses the dataset's own train/validation/test splits. Each text entry is
concatenated with the standard 6-byte separator (mask=0). Content bytes
get mask=1.

Output:
    training_data/real_wikitext/shard_000.traindat + .mask + .meta.json
    eval_data/real_wikitext/shard_000.traindat + .mask + .meta.json

Usage:
    python download_wikitext.py                     # download + convert
    python download_wikitext.py --dataset wikitext-2-raw-v1   # tiny version
    python download_wikitext.py --out ./my_data --eval-out ./my_eval
"""

import argparse
import hashlib
import json
import os
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

CONVERTER_VERSION = '1.0.0'
SEPARATOR = b'\xff\xfe\x00\x00\xfe\xff'


def clean_text(text: str) -> bytes:
    """Apply the same preprocessing as convert.py: CRLF→LF, NFKC, UTF-8."""
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = unicodedata.normalize('NFKC', text)
    return text.encode('utf-8')


def write_shard(out_dir: Path, name: str, data: bytes, mask: bytes,
                meta: dict):
    """Write a single shard: .traindat + .mask + .meta.json."""
    os.makedirs(out_dir, exist_ok=True)
    traindat_path = out_dir / f'{name}.traindat'
    mask_path = out_dir / f'{name}.mask'
    meta_path = out_dir / f'{name}.meta.json'

    with open(traindat_path, 'wb') as f:
        f.write(data)
    with open(mask_path, 'wb') as f:
        f.write(mask)

    meta['sha256'] = hashlib.sha256(data).hexdigest()
    meta['size_bytes'] = len(data)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return traindat_path


def convert_split(dataset_split, out_dir: Path, split_name: str,
                  shard_size: int):
    """Convert one HF dataset split to sharded .traindat + .mask files."""
    print(f'\n[{split_name.upper()}] Converting {len(dataset_split)} entries ...')
    t0 = time.perf_counter()

    buf_data = bytearray()
    buf_mask = bytearray()
    shard_idx = 0
    total_bytes = 0
    entries_in_shard = 0
    is_first = True
    skipped = 0

    def flush():
        nonlocal buf_data, buf_mask, shard_idx, total_bytes, entries_in_shard, is_first
        if not buf_data:
            return
        name = f'shard_{shard_idx:03d}'
        shard_dir = out_dir / 'real_wikitext'
        meta = {
            'converter_version': CONVERTER_VERSION,
            'source': 'Salesforce/wikitext (wikitext-103-raw-v1)',
            'split': split_name,
            'shard_index': shard_idx,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'separator_hex': SEPARATOR.hex(' '),
            'mask_semantics': {'0': 'separator', '1': 'content'},
            'num_entries': entries_in_shard,
        }
        path = write_shard(shard_dir, name, bytes(buf_data), bytes(buf_mask),
                           meta)
        mb = len(buf_data) / (1024 ** 2)
        print(f'  shard {shard_idx:03d}  {entries_in_shard:>6d} entries  '
              f'{mb:>8.1f} MB  {path}')
        total_bytes += len(buf_data)
        shard_idx += 1
        buf_data = bytearray()
        buf_mask = bytearray()
        entries_in_shard = 0
        is_first = True

    for i, row in enumerate(dataset_split):
        text = row['text']
        if not text or not text.strip():
            skipped += 1
            continue

        clean = clean_text(text)
        if not clean:
            skipped += 1
            continue

        # Separator between entries
        if not is_first:
            buf_data.extend(SEPARATOR)
            buf_mask.extend(b'\x00' * len(SEPARATOR))

        is_first = False
        buf_data.extend(clean)
        buf_mask.extend(b'\x01' * len(clean))
        entries_in_shard += 1

        if len(buf_data) >= shard_size:
            flush()

        # Heartbeat
        if (i + 1) % 50000 == 0:
            elapsed = time.perf_counter() - t0
            print(f'  [{split_name}] {i+1}/{len(dataset_split)} ... {elapsed:.1f}s')

    flush()

    elapsed = time.perf_counter() - t0
    total_mb = total_bytes / (1024 ** 2)
    print(f'  Total: {shard_idx} shards, {total_mb:.1f} MB, '
          f'{skipped} empty entries skipped ({elapsed:.1f}s)')


def main():
    parser = argparse.ArgumentParser(
        description='Download WikiText-103-raw and convert to .traindat + .mask')

    v4_root = Path(__file__).parent.parent
    parser.add_argument('--dataset', default='wikitext-103-raw-v1',
                        help='HF dataset config (default: wikitext-103-raw-v1)')
    parser.add_argument('--out', default=str(v4_root / 'training_data'),
                        help='training output directory')
    parser.add_argument('--eval-out', default=str(v4_root / 'eval_data'),
                        help='eval output directory')
    parser.add_argument('--shard-size', default='256MB',
                        help='max bytes per shard (default: 256MB)')
    args = parser.parse_args()

    # Parse shard size
    s = args.shard_size.strip().upper()
    multipliers = {'KB': 1024, 'MB': 1024**2, 'GB': 1024**3}
    shard_size = None
    for suffix, mult in multipliers.items():
        if s.endswith(suffix):
            shard_size = int(float(s[:-len(suffix)]) * mult)
            break
    if shard_size is None:
        shard_size = int(s)

    print(f'VRAXION v4 — WikiText Downloader + Converter v{CONVERTER_VERSION}')
    print(f'{"=" * 56}')
    print(f'Dataset: Salesforce/wikitext ({args.dataset})')
    print(f'Shard size: {shard_size / (1024**2):.0f} MB')

    # Download from HuggingFace
    print('\nDownloading from HuggingFace (first run caches locally) ...')
    from datasets import load_dataset
    ds = load_dataset('Salesforce/wikitext', args.dataset)

    print(f'  train:      {len(ds["train"]):>8d} entries')
    print(f'  validation: {len(ds["validation"]):>8d} entries')
    print(f'  test:       {len(ds["test"]):>8d} entries')

    out_dir = Path(args.out)
    eval_out = Path(args.eval_out)

    # Convert train split
    convert_split(ds['train'], out_dir, 'train', shard_size)

    # Merge validation + test for eval
    from datasets import concatenate_datasets
    eval_ds = concatenate_datasets([ds['validation'], ds['test']])
    convert_split(eval_ds, eval_out, 'eval', shard_size)

    print('\nDone.')


if __name__ == '__main__':
    main()
