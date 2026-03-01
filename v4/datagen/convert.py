"""Real-world data converter — text/code files to .traindat + .mask for INSTNCT v4.

Converts UTF-8 text and code files into the byte-level training format used
by INSTNCT v4. Output is identical in structure to generate.py .traindat files,
with the addition of a parallel .mask file (supervision mask) and a .meta.json
provenance file per shard.

Preprocessing pipeline (per file):
    read raw bytes → detect encoding from BOM (UTF-8/16/32) + strip BOM
    → decode with detected encoding (errors=replace)
    → CRLF→LF + CR→LF → NFKC normalize → re-encode UTF-8

Files are concatenated with a 6-byte separator (0xFF 0xFE 0x00 0x00 0xFE 0xFF)
that is impossible in valid UTF-8. The supervision mask marks content bytes as 1
(supervised) and separator bytes as 0 (skip in loss computation).

Usage:
    python convert.py --input ./corpus --out ./training_data --domain text
    python convert.py --input ./repos  --out ./training_data --domain code --shard-size 512MB
    python convert.py --input ./mix    --out ./training_data --domain text --ext .txt,.md,.rst
    python convert.py --list-extensions text
    python convert.py --dry-run --input ./corpus --domain text
"""

import argparse                          # CLI argument parsing
import codecs                            # BOM constants (BOM_UTF8, BOM_UTF16_*, BOM_UTF32_*)
import hashlib                           # SHA-256 shard checksums for provenance
import json                              # .meta.json output
import os                                # makedirs, path ops
import time                              # wall-clock timing + heartbeat
import unicodedata                       # NFKC normalization
from datetime import datetime, timezone  # ISO timestamps for .meta.json
from pathlib import Path                 # recursive file discovery


# ── Constants ──────────────────────────────────────────────────────

CONVERTER_VERSION = '1.0.0'

# 6-byte separator — 0xFF and 0xFE never appear in valid UTF-8 (RFC 3629).
# Placed between files in a shard. Mask=0 at these positions.
SEPARATOR = b'\xff\xfe\x00\x00\xfe\xff'

TEXT_EXTENSIONS = frozenset({
    '.txt', '.md', '.rst', '.csv', '.tsv',
    '.json', '.jsonl', '.xml', '.html', '.htm',
    '.yaml', '.yml', '.log', '.ini', '.cfg',
})

CODE_EXTENSIONS = frozenset({
    '.py', '.js', '.ts', '.jsx', '.tsx',
    '.c', '.cpp', '.cc', '.h', '.hpp',
    '.go', '.rs', '.java', '.rb', '.sh',
    '.cs', '.lua', '.toml', '.sql', '.r',
    '.jl', '.zig', '.nim', '.pl', '.swift',
    '.kt', '.scala', '.hs', '.ml', '.ex',
    '.erl', '.clj', '.vim', '.ps1', '.bat',
    '.cmake', '.dockerfile', '.proto', '.graphql',
})

DOMAIN_EXTENSIONS = {
    'text': TEXT_EXTENSIONS,
    'code': CODE_EXTENSIONS,
}

# Directories to skip during recursive scan.
PRUNE_DIRS = frozenset({
    '__pycache__', 'node_modules', '.tox', 'venv', '.venv', 'env',
    '.mypy_cache', '.pytest_cache', '.eggs', 'dist', 'build',
})


# ── Helpers ────────────────────────────────────────────────────────

def parse_size(s):
    """Parse human-readable size string: '256MB', '1GB', '1048576', etc."""
    s = s.strip().upper()
    multipliers = {'KB': 1024, 'MB': 1024**2, 'GB': 1024**3}
    for suffix, mult in multipliers.items():
        if s.endswith(suffix):
            return int(float(s[:-len(suffix)]) * mult)
    return int(s)


def detect_bom(raw: bytes) -> tuple[bytes, str]:
    """Detect BOM prefix, strip it, and return the encoding name.

    Check UTF-32 before UTF-16 (ordering matters — UTF-32-LE BOM
    FF FE 00 00 starts with UTF-16-LE BOM FF FE).

    Returns (stripped_bytes, encoding_name) where encoding_name is
    a valid Python codec name. Default: 'utf-8' when no BOM found."""
    bom_map = [
        (codecs.BOM_UTF32_LE, 'utf-32-le'),
        (codecs.BOM_UTF32_BE, 'utf-32-be'),
        (codecs.BOM_UTF16_LE, 'utf-16-le'),
        (codecs.BOM_UTF16_BE, 'utf-16-be'),
        (codecs.BOM_UTF8,     'utf-8'),
    ]
    for bom, encoding in bom_map:
        if raw.startswith(bom):
            return raw[len(bom):], encoding
    return raw, 'utf-8'


def process_file(path: Path) -> bytes | None:
    """Read a text file and apply the full preprocessing pipeline.

    Pipeline:
        1. Read raw bytes
        2. Detect encoding from BOM (UTF-8/16/32) + strip BOM
        3. Decode with detected encoding (default UTF-8, invalid → U+FFFD)
        4. CRLF → LF, then CR-only → LF (on text, works for all encodings)
        5. NFKC normalize (collapse Unicode variants)
        6. Re-encode to UTF-8

    Returns cleaned bytes, or None on failure (file is skipped)."""
    try:
        raw = path.read_bytes()
        if not raw:
            return None
        raw, encoding = detect_bom(raw)                               # step 2: detect encoding + strip BOM
        text = raw.decode(encoding, errors='replace')                 # step 3: decode (detected encoding)
        text = text.replace('\r\n', '\n')                             # step 4: CRLF → LF
        text = text.replace('\r', '\n')                               # step 4b: CR-only → LF (old Mac)
        text = unicodedata.normalize('NFKC', text)                    # step 5: NFKC normalization
        clean = text.encode('utf-8')                                  # step 6: back to UTF-8 bytes
        return clean if clean else None
    except OSError as e:
        print(f'  SKIP (read error): {path} — {e}')
        return None
    except Exception as e:
        print(f'  SKIP (unexpected): {path} — {type(e).__name__}: {e}')
        return None


def discover_files(root: Path, extensions: frozenset,
                   min_bytes: int, max_bytes: int) -> list[Path]:
    """Recursively discover text/code files under root.

    Filters by extension, size, and prunes hidden/build directories.
    Returns a deterministically sorted list of Paths."""
    found = []
    for path in root.rglob('*'):
        # Skip non-files
        if not path.is_file():
            continue
        # Prune hidden dirs and known junk dirs
        parts = path.relative_to(root).parts
        if any(p.startswith('.') or p in PRUNE_DIRS for p in parts[:-1]):
            continue
        # Extension filter
        if path.suffix.lower() not in extensions:
            continue
        # Size filter
        try:
            sz = path.stat().st_size
        except OSError:
            continue
        if sz < min_bytes or sz > max_bytes:
            continue
        found.append(path)
    found.sort()
    return found


# ── Shard Writer ───────────────────────────────────────────────────

class ShardWriter:
    """Accumulates processed file bytes and flushes to sharded output files.

    Output per shard:
        real_{domain}/shard_{NNN}.traindat  — raw training bytes
        real_{domain}/shard_{NNN}.mask      — supervision mask (0=separator, 1=content)
        real_{domain}/shard_{NNN}.meta.json — provenance metadata
    """

    def __init__(self, out_dir: Path, domain: str, shard_size: int):
        self.out_dir = out_dir / f'real_{domain}'
        self.domain = domain
        self.shard_size = shard_size
        self._reset()
        self.shard_idx = 0
        # Grand totals
        self.total_bytes = 0
        self.total_files = 0
        self.total_shards = 0

    def _reset(self):
        """Reset buffers for the next shard."""
        self.buf_data = bytearray()
        self.buf_mask = bytearray()
        self.sources = []
        self.is_first_in_shard = True

    def add_file(self, path: Path, clean_bytes: bytes, raw_size: int):
        """Add a processed file to the current shard.

        Inserts a separator (mask=0) between files. If the file is larger
        than shard_size, it is chunked across multiple shards. Small files
        that overflow the current shard trigger a normal flush (old behavior).
        Chunk metadata (chunk index + byte range) is added to sources."""
        # Separator between files (never between chunks of the same file)
        if not self.is_first_in_shard:
            self.buf_data.extend(SEPARATOR)
            self.buf_mask.extend(b'\x00' * len(SEPARATOR))

        self.is_first_in_shard = False
        self.total_files += 1
        total = len(clean_bytes)

        if total <= self.shard_size:
            # ── Normal path: file fits within one shard ──
            self.buf_data.extend(clean_bytes)
            self.buf_mask.extend(b'\x01' * total)
            self.sources.append({
                'path': str(path),
                'size_raw': raw_size,
                'size_clean': total,
            })
            if len(self.buf_data) >= self.shard_size:
                self.flush()
        else:
            # ── Large file: chunk across shard boundaries ──
            print(f'  CHUNK: {path.name} ({total / 1024**2:.1f} MB)'
                  f' — splitting across shards')
            offset = 0
            chunk_idx = 0
            while offset < total:
                space = self.shard_size - len(self.buf_data)
                if space <= 0:
                    self.flush()
                    space = self.shard_size

                end = min(offset + space, total)
                self.buf_data.extend(clean_bytes[offset:end])
                self.buf_mask.extend(b'\x01' * (end - offset))

                self.sources.append({
                    'path': str(path),
                    'size_raw': raw_size,
                    'size_clean': total,
                    'chunk': chunk_idx,
                    'byte_range': [offset, end],
                })

                offset = end
                chunk_idx += 1

                if len(self.buf_data) >= self.shard_size:
                    self.flush()

    def flush(self):
        """Write current buffers as a shard and reset."""
        if not self.buf_data:
            return

        os.makedirs(self.out_dir, exist_ok=True)

        name = f'shard_{self.shard_idx:03d}'
        traindat_path = self.out_dir / f'{name}.traindat'
        mask_path     = self.out_dir / f'{name}.mask'
        meta_path     = self.out_dir / f'{name}.meta.json'

        data_bytes = bytes(self.buf_data)
        mask_bytes = bytes(self.buf_mask)

        # Write .traindat
        with open(traindat_path, 'wb') as f:
            f.write(data_bytes)

        # Write .mask
        with open(mask_path, 'wb') as f:
            f.write(mask_bytes)

        # Compute SHA-256 for provenance
        sha = hashlib.sha256(data_bytes).hexdigest()

        # Write .meta.json
        meta = {
            'converter_version': CONVERTER_VERSION,
            'domain': self.domain,
            'shard_index': self.shard_idx,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'separator_hex': SEPARATOR.hex(' '),
            'mask_semantics': {'0': 'separator (do not predict)', '1': 'content (supervise)'},
            'size_bytes': len(data_bytes),
            'sha256': sha,
            'num_files': len(self.sources),
            'processing': [
                'detect_bom', 'decode_detected_encoding',
                'normalize_line_endings', 'nfkc_normalize', 'utf8_encode',
            ],
            'sources': self.sources,
        }
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        shard_mb = len(data_bytes) / (1024 ** 2)
        print(f'  shard {self.shard_idx:03d}  {len(self.sources):>5d} files  '
              f'{shard_mb:>8.1f} MB  {traindat_path}')

        # Update totals
        self.total_bytes += len(data_bytes)
        # total_files is tracked in add_file(), not here
        self.total_shards += 1
        self.shard_idx += 1
        self._reset()

    def finalize(self):
        """Flush remaining data and print grand totals."""
        self.flush()
        total_mb = self.total_bytes / (1024 ** 2)
        print(f'\n  Total: {self.total_shards} shards, '
              f'{self.total_files} files, {total_mb:.1f} MB')


# ── Main Conversion Logic ─────────────────────────────────────────

def _process_files(files: list[Path], writer: 'ShardWriter', domain: str,
                    heartbeat_s: float, t0: float) -> list[str]:
    """Process a list of files through the ShardWriter.

    Extracted so it can be called once for training files and once for eval
    files without duplicating the processing loop.

    Returns a list of skipped file paths."""
    skipped = []
    last_report = time.perf_counter()

    for i, fpath in enumerate(files):
        # Heartbeat progress
        now = time.perf_counter()
        if heartbeat_s > 0 and (now - last_report) >= heartbeat_s:
            pct = 100 * (i + 1) / len(files)
            elapsed = now - t0
            print(f'  [{domain}] {i+1}/{len(files)} ({pct:.0f}%) ... {elapsed:.1f}s')
            last_report = now

        # stat() before process_file() — TOCTOU guard: file may vanish between
        # discovery and processing. Without this try, a deleted file crashes
        # the entire pipeline instead of skipping gracefully.
        try:
            raw_size = fpath.stat().st_size
        except OSError:
            skipped.append(str(fpath))
            continue

        clean = process_file(fpath)
        if clean is None:
            skipped.append(str(fpath))
            continue

        writer.add_file(fpath, clean, raw_size)

    writer.finalize()
    return skipped


def _print_skipped(skipped: list[str]):
    """Print skipped file summary."""
    print(f'\n  Skipped: {len(skipped)} files')
    if skipped and len(skipped) <= 20:
        for s in skipped:
            print(f'    {s}')
    elif skipped:
        for s in skipped[:10]:
            print(f'    {s}')
        print(f'    ... and {len(skipped) - 10} more')


def convert(input_dir: Path, out_dir: Path, domain: str,
            extensions: frozenset, shard_size: int,
            min_bytes: int, max_bytes: int, heartbeat_s: float,
            dry_run: bool, eval_split: float = 0.0,
            eval_out: Path | None = None):
    """Discover files, preprocess, and write sharded training data.

    If eval_split > 0, the discovered files are split deterministically:
    first (1 - eval_split) fraction → training, last eval_split fraction → eval.
    Files are sorted alphabetically before splitting, so the split is stable
    across runs. The eval set is written to eval_out (or out_dir/../eval_data/)."""

    t0 = time.perf_counter()

    # ── Discovery ──
    print(f'\nScanning {input_dir} for {domain} files ...')
    files = discover_files(input_dir, extensions, min_bytes, max_bytes)

    if not files:
        print('  0 files found. Nothing to do.')
        return

    raw_total = sum(f.stat().st_size for f in files)
    raw_mb = raw_total / (1024 ** 2)
    shard_mb = shard_size / (1024 ** 2)
    print(f'  {len(files)} files found ({raw_mb:.1f} MB raw)')
    print(f'  Shard size: {shard_mb:.0f} MB')

    if dry_run:
        # Show extension breakdown
        ext_counts: dict[str, int] = {}
        for f in files:
            ext = f.suffix.lower()
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
        print('\n  Extension breakdown:')
        for ext, count in sorted(ext_counts.items(), key=lambda x: -x[1]):
            print(f'    {ext:>12s}  {count:>6d} files')
        if eval_split > 0:
            split_idx = int(len(files) * (1.0 - eval_split))
            print(f'\n  Eval split: {eval_split:.0%} -- {split_idx} train, {len(files) - split_idx} eval')
        elapsed = time.perf_counter() - t0
        print(f'\n  Dry run complete ({elapsed:.1f}s). No files written.')
        return

    # ── Split ──
    # files are already sorted by discover_files(). If eval_split > 0,
    # deterministic file-level split: first N → train, rest → eval.
    # Minimum 1 file per split to avoid empty sets.
    if eval_split > 0:
        split_idx = max(1, min(len(files) - 1, int(len(files) * (1.0 - eval_split))))
        train_files = files[:split_idx]
        eval_files = files[split_idx:]
        if eval_out is None:
            eval_out = Path(__file__).parent.parent / 'eval_data'
        print(f'\n  Eval split: {len(train_files)} train + {len(eval_files)} eval files')
    else:
        train_files = files
        eval_files = []

    # ── Training data ──
    print(f'\n[TRAIN] Converting {len(train_files)} files ...')
    train_writer = ShardWriter(out_dir, domain, shard_size)
    skipped = _process_files(train_files, train_writer, domain, heartbeat_s, t0)
    _print_skipped(skipped)

    # ── Eval data (optional) ──
    if eval_files:
        assert eval_out is not None  # guaranteed by the split branch above; narrowing for Pylance
        print(f'\n[EVAL] Converting {len(eval_files)} files ...')
        eval_writer = ShardWriter(eval_out, domain, shard_size)
        eval_skipped = _process_files(eval_files, eval_writer, domain, heartbeat_s, t0)
        _print_skipped(eval_skipped)

    elapsed = time.perf_counter() - t0
    print(f'\n  Wall time: {elapsed:.1f}s')


# ── CLI ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert real-world text/code files to .traindat + .mask for INSTNCT v4.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python convert.py --input ./corpus --domain text
  python convert.py --input ./repos  --domain code --shard-size 512MB
  python convert.py --input ./mix    --domain text --ext .txt,.md,.rst
  python convert.py --input ./corpus --domain text --eval-split 0.2
  python convert.py --list-extensions code
  python convert.py --dry-run --input ./corpus --domain text""",
    )

    _v4_root = Path(__file__).parent.parent
    _default_out = str(_v4_root / 'training_data')
    _default_eval_out = str(_v4_root / 'eval_data')

    parser.add_argument('--input', default=None,
                        help='source directory to scan recursively')
    parser.add_argument('--out', default=_default_out,
                        help=f'output directory (default: {_default_out})')
    parser.add_argument('--domain', default=None, choices=['text', 'code'],
                        help='domain: text or code (determines extension filter and output prefix)')
    parser.add_argument('--shard-size', default='256MB',
                        help='max bytes per shard (e.g. 256MB, 512MB, 1GB). Default: 256MB')
    parser.add_argument('--ext', default=None,
                        help='override extension list (comma-separated, e.g. .py,.js)')
    parser.add_argument('--list-extensions', metavar='DOMAIN', nargs='?', const='all',
                        help='print default extensions for a domain and exit')
    parser.add_argument('--dry-run', action='store_true',
                        help='scan and report files/sizes without writing anything')
    parser.add_argument('--heartbeat', type=float, default=5.0,
                        help='progress interval in seconds (0 disables). Default: 5')
    parser.add_argument('--min-bytes', type=int, default=10,
                        help='skip files smaller than this. Default: 10')
    parser.add_argument('--max-bytes', default='50MB',
                        help='skip files larger than this. Default: 50MB')
    parser.add_argument('--eval-split', type=float, default=0.0,
                        help='fraction of files to reserve for eval (0.0 = no split, 0.2 = 20%% eval)')
    parser.add_argument('--eval-out', default=None,
                        help='eval output directory (default: v4/eval_data/)')

    args = parser.parse_args()

    # ── Handle --list-extensions ──
    if args.list_extensions is not None:
        if args.list_extensions == 'all':
            for d, exts in sorted(DOMAIN_EXTENSIONS.items()):
                print(f'{d}: {" ".join(sorted(exts))}')
        elif args.list_extensions in DOMAIN_EXTENSIONS:
            print(' '.join(sorted(DOMAIN_EXTENSIONS[args.list_extensions])))
        else:
            parser.error(f"Unknown domain: {args.list_extensions}")
        raise SystemExit(0)

    # ── Validate required args (not needed for --list-extensions) ──
    if args.input is None:
        parser.error("the following arguments are required: --input")
    if args.domain is None:
        parser.error("the following arguments are required: --domain")

    # ── Resolve args ──
    input_dir = Path(args.input)
    if not input_dir.is_dir():
        parser.error(f"Input directory does not exist: {input_dir}")

    out_dir = Path(args.out)

    extensions = DOMAIN_EXTENSIONS[args.domain]
    if args.ext:
        extensions = frozenset(
            e.strip() if e.strip().startswith('.') else f'.{e.strip()}'
            for e in args.ext.split(',') if e.strip()
        )

    shard_size = parse_size(args.shard_size)
    max_bytes = parse_size(args.max_bytes)

    print(f'VRAXION v4 — Real-World Data Converter v{CONVERTER_VERSION}')
    print(f'{"=" * 52}')

    eval_out = Path(args.eval_out) if args.eval_out else None

    convert(
        input_dir=input_dir,
        out_dir=out_dir,
        domain=args.domain,
        extensions=extensions,
        shard_size=shard_size,
        min_bytes=args.min_bytes,
        max_bytes=max_bytes,
        heartbeat_s=args.heartbeat,
        dry_run=args.dry_run,
        eval_split=args.eval_split,
        eval_out=eval_out,
    )
