"""
Universal .traindat loader for Diamond Code swarm training.

.traindat files are raw bytes -- no header, no encoding.
Gray code scalar encoding: each byte -> 1 scalar in (0, 1).
num_bits channels per position = num_bits bytes of data per position.

Also includes ThemeLoader for JSONL paired training (dialogue mode).

Usage:
    loader = TraindatLoader("data/traindat/")
    x, y = loader.sample_batch(batch_size=1, seq_len=62, num_bits=8, seed=42)

    theme = ThemeLoader("data/themes/", num_bits=6184)
    x, y, mask = theme.sample_batch(batch_size=10, seq_len=6, seed=42)
"""

import json
import base64
import random
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union


# --- Gray code lookup tables (256-entry, precomputed) ---
# GRAY_ENCODE[position] = byte value at that position in Gray code order
GRAY_ENCODE = [i ^ (i >> 1) for i in range(256)]
# GRAY_POSITION[byte_value] = position in Gray code order (inverse of GRAY_ENCODE)
GRAY_POSITION = [0] * 256
for _i in range(256):
    GRAY_POSITION[GRAY_ENCODE[_i]] = _i
# GRAY_SCALAR[byte_value] = scalar in (0.00195, 0.998) via (position + 0.5) / 256
GRAY_SCALAR = [(_p + 0.5) / 256.0 for _p in GRAY_POSITION]


def byte_to_gray_scalar(byte_val: int) -> float:
    """Convert a byte value (0-255) to its Gray-coded scalar in (0, 1)."""
    return float(GRAY_SCALAR[byte_val])


def gray_scalar_to_byte(scalar: float) -> int:
    """Convert a Gray-coded scalar back to the original byte value."""
    pos = max(0, min(255, round(scalar * 256.0 - 0.5)))
    return GRAY_ENCODE[pos]


def generate_batch_binary_bits(
    corpus: bytes,
    n_samples: int,
    seq_len: int = 16,
    num_bits: int = 8,
    seed=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate next-chunk prediction batch using binary bit encoding.

    Each position = bytes_per_pos bytes (num_bits // 8), each decomposed into 8 bits.
    Input = bits[0:seq_len], Target = bits[1:seq_len+1] (shifted by bytes_per_pos bytes).

    Args:
        num_bits: Total bits per position. Must be multiple of 8.
            num_bits=8  -> 1 byte/pos (original behavior)
            num_bits=16 -> 2 bytes/pos (16 binary bits)
            num_bits=64 -> 8 bytes/pos (64 binary bits, covers 128 bytes in 16 positions)

    Returns:
        (x, y, mask) where x, y are [n_samples, seq_len, num_bits] float tensors
        and mask is [n_samples, seq_len, num_bits] float tensor (1.0=real data, 0.0=N/A).
        For traindat, mask is always all-1s (all bits are real data).
    """
    assert num_bits % 8 == 0, f"num_bits must be multiple of 8, got {num_bits}"
    bytes_per_pos = num_bits // 8

    if seed is not None:
        random.seed(seed)

    chunk_bytes = (seq_len + 1) * bytes_per_pos  # seq_len chunks + 1 shifted target
    max_start = len(corpus) - chunk_bytes

    if max_start < 0:
        raise ValueError(
            f"Corpus too small ({len(corpus)} bytes) for "
            f"seq_len={seq_len}, bytes_per_pos={bytes_per_pos} (need >= {chunk_bytes})"
        )

    # Vectorized: gather all chunks into numpy array, unpack bits in bulk
    corpus_arr = np.frombuffer(corpus, dtype=np.uint8)
    starts = np.array([random.randint(0, max_start) for _ in range(n_samples)])

    # Build offset array: [0, 1, ..., chunk_bytes-1] for each sample
    offsets = np.arange(chunk_bytes)
    # indices[i, j] = starts[i] + j  -> shape [n_samples, chunk_bytes]
    indices = starts[:, None] + offsets[None, :]
    # Gather all bytes: [n_samples, chunk_bytes]
    all_bytes = corpus_arr[indices]

    # Reshape to [n_samples, seq_len+1, bytes_per_pos]
    all_bytes = all_bytes.reshape(n_samples, seq_len + 1, bytes_per_pos)

    # Unpack bits: numpy unpackbits along last axis, MSB first
    # [n_samples, seq_len+1, bytes_per_pos] -> [n_samples, seq_len+1, num_bits]
    all_bits = np.unpackbits(all_bytes, axis=2).astype(np.float32)

    # Input = positions [0:seq_len], target = positions [1:seq_len+1]
    x = torch.from_numpy(all_bits[:, :seq_len, :].copy())
    y = torch.from_numpy(all_bits[:, 1:seq_len + 1, :].copy())

    # Bit mask: 1.0 = real data, 0.0 = N/A (empty/padding).
    # For traindat, every bit is real data. Future: sparse encodings set padding to 0.
    mask = torch.ones_like(y)

    return x, y, mask


def load_batch_from_file(
    filepath: str,
    n_samples: int,
    seq_len: int,
    num_bits: int,
    seed: int = None,
    binary_bits_mode: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Load a random batch from a specific .traindat file (for dream rehearsal).

    Returns (x, y, mask). mask is None for Gray mode, [B,T,num_bits] for binary.
    Raises FileNotFoundError if file doesn't exist.
    """
    with open(filepath, 'rb') as f:
        corpus = f.read()
    if binary_bits_mode:
        return generate_batch_binary_bits(corpus, n_samples, seq_len, num_bits, seed)
    else:
        x, y = generate_batch_from_bytes(corpus, n_samples, seq_len, num_bits, seed)
        return x, y, None


def generate_batch_byte_tokens(
    corpus: bytes,
    n_samples: int,
    seq_len: int = 16,
    seed=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate next-byte prediction batch from raw bytes (discrete token mode).

    Each position = 1 byte (0-255) as a Long tensor.
    Input = bytes[0:seq_len], Target = bytes[1:seq_len+1] (shifted by 1 byte).
    Every unique byte maps to a unique integer — lossless by construction.

    Returns:
        (x, y) where x, y are [n_samples, seq_len] Long tensors
    """
    if seed is not None:
        random.seed(seed)

    chunk_len = seq_len + 1  # one extra byte for shifted target
    max_start = len(corpus) - chunk_len

    if max_start < 0:
        raise ValueError(
            f"Corpus too small ({len(corpus)} bytes) for "
            f"seq_len={seq_len} (need >= {chunk_len})"
        )

    x = torch.zeros(n_samples, seq_len, dtype=torch.long)
    y = torch.zeros(n_samples, seq_len, dtype=torch.long)

    for i in range(n_samples):
        start = random.randint(0, max_start)
        chunk = corpus[start:start + chunk_len]

        for t in range(seq_len):
            x[i, t] = chunk[t]
            y[i, t] = chunk[t + 1]

    return x, y


def generate_batch_from_bytes(
    corpus: bytes,
    n_samples: int,
    seq_len: int = 16,
    num_bits: int = 64,
    seed=None,
    role_map: Optional[np.ndarray] = None,
) -> Union[Tuple[torch.Tensor, torch.Tensor],
           Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Generate next-chunk prediction batch from raw bytes using Gray code scalar encoding.

    Each position holds num_bits bytes, each Gray-coded to a scalar in (0, 1).
    Input = chunk[0:seq_len], Target = chunk[1:seq_len+1] (shifted by num_bits bytes).

    Args:
        role_map: Optional numpy array from golden_disc.build_role_map().
            If provided, returns a 3-tuple with mask tensor.

    Returns:
        Without role_map: (x, y)
        With role_map: (x, y, mask) where mask is [n_samples, seq_len] uint8
            mask values: 0=separator, 1=question, 2=answer, 3=type, 4=freetext
            mask[i,t] = role of the TARGET chunk at position t.
    """
    if seed is not None:
        random.seed(seed)

    bytes_per_pos = num_bits  # each channel = 1 byte (was num_bits // 8)
    chunk_len = (seq_len + 1) * bytes_per_pos
    max_start = len(corpus) - chunk_len - bytes_per_pos

    if max_start < 0:
        raise ValueError(
            f"Corpus too small ({len(corpus)} bytes) for "
            f"seq_len={seq_len}, num_bits={num_bits} (need >= {chunk_len + bytes_per_pos})"
        )

    x = torch.zeros(n_samples, seq_len, num_bits)
    y = torch.zeros(n_samples, seq_len, num_bits)
    if role_map is not None:
        mask = torch.zeros(n_samples, seq_len, dtype=torch.uint8)

    for i in range(n_samples):
        start = random.randint(0, max_start // bytes_per_pos) * bytes_per_pos
        chunk = corpus[start:start + chunk_len + bytes_per_pos]

        for t in range(seq_len):
            offset = t * bytes_per_pos
            for b in range(num_bits):
                x[i, t, b] = GRAY_SCALAR[chunk[offset + b]]

            target_offset = offset + bytes_per_pos
            for b in range(num_bits):
                y[i, t, b] = GRAY_SCALAR[chunk[target_offset + b]]

            # Mask: role of the target chunk (what model is predicting)
            if role_map is not None:
                target_byte_pos = start + target_offset
                if target_byte_pos < len(role_map):
                    mask[i, t] = role_map[target_byte_pos]

    if role_map is not None:
        return x, y, mask
    return x, y


class TraindatLoader:
    """
    Loads .traindat files from a directory with weighted sampling.

    Files are lazily loaded into memory and cached.
    Weights can be updated mid-run via update_weights() (live controls hook).
    """

    def __init__(self, data_dir: str, weights: Optional[Dict[str, float]] = None):
        self.data_dir = Path(data_dir)
        self._corpora: Dict[str, bytes] = {}
        self._role_maps: Dict[str, np.ndarray] = {}
        self._file_list: List[str] = []
        self._weights: Dict[str, float] = weights or {}
        self._explicit_weights = weights is not None  # if True, unlisted files get weight=0
        self._sequential = False
        self._seq_steps = 100
        self._seq_index = 0
        self._seq_counter = 0
        self._seq_active: List[str] = []
        self._discover_files()

    def _discover_files(self):
        """Scan directory for .traindat files."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        self._file_list = sorted([
            f.name for f in self.data_dir.iterdir()
            if f.suffix == '.traindat' and f.is_file()
        ])

        if not self._file_list:
            raise FileNotFoundError(f"No .traindat files in {self.data_dir}")

        for fname in self._file_list:
            if fname not in self._weights:
                self._weights[fname] = 0.0 if self._explicit_weights else 1.0

    def _load_corpus(self, filename: str) -> bytes:
        """Lazily load and cache a .traindat file."""
        if filename not in self._corpora:
            path = self.data_dir / filename
            with open(path, 'rb') as f:
                self._corpora[filename] = f.read()
        return self._corpora[filename]

    def _get_role_map(self, filename: str) -> np.ndarray:
        """Lazily build and cache role map for a .traindat file."""
        if filename not in self._role_maps:
            from golden_disc import build_role_map
            corpus = self._load_corpus(filename)
            self._role_maps[filename] = build_role_map(corpus)
        return self._role_maps[filename]

    def update_weights(self, new_weights: Dict[str, float]):
        """Update sampling weights (from live controls). Re-scans directory for new files."""
        self._discover_files()
        self._weights.update(new_weights)
        # Clean stale cached corpora for deleted files
        stale = [k for k in self._corpora if k not in self._file_list]
        for k in stale:
            del self._corpora[k]
        # Rebuild sequential active list when weights change
        if self._sequential:
            self._rebuild_seq_active()

    def set_sequential(self, enabled: bool, steps_per_dataset: int = 100):
        """Enable/disable sequential dataset cycling.

        When enabled, pick_file() cycles through active datasets (weight > 0)
        in sorted order, spending steps_per_dataset steps on each before
        advancing. When the last dataset is reached, wraps back to first.
        """
        was_enabled = self._sequential
        self._sequential = enabled
        self._seq_steps = max(1, steps_per_dataset)
        if enabled and not was_enabled:
            self._seq_index = 0
            self._seq_counter = 0
            self._rebuild_seq_active()

    def _rebuild_seq_active(self):
        """Rebuild the sorted list of active files for sequential mode."""
        active = sorted([f for f in self._file_list if self._weights.get(f, 0) > 0])
        if active != self._seq_active:
            self._seq_active = active
            if self._seq_index >= len(self._seq_active):
                self._seq_index = 0
            if self._seq_active:
                print(f"  [SEQ] datasets: {' -> '.join(f.replace('.traindat','') for f in self._seq_active)} "
                      f"({self._seq_steps} steps each, cycling)")

    @property
    def current_dataset(self) -> Optional[str]:
        """Return the current dataset name in sequential mode, or the sole active dataset in random mode."""
        if self._sequential and self._seq_active:
            return self._seq_active[self._seq_index]
        # In random mode: if exactly 1 dataset has weight > 0, return it
        active = [f for f, w in self._weights.items() if w > 0]
        if len(active) == 1:
            return active[0]
        return None

    @property
    def files(self) -> List[str]:
        return list(self._file_list)

    @property
    def weights(self) -> Dict[str, float]:
        return dict(self._weights)

    def pick_file(self) -> str:
        """Select next file. Sequential mode cycles through active datasets;
        random mode uses weighted sampling. Falls back to uniform if all weights are zero."""
        if self._sequential and self._seq_active:
            fname = self._seq_active[self._seq_index]
            self._seq_counter += 1
            if self._seq_counter >= self._seq_steps:
                self._seq_counter = 0
                old_idx = self._seq_index
                self._seq_index = (self._seq_index + 1) % len(self._seq_active)
                old_name = self._seq_active[old_idx].replace('.traindat', '')
                new_name = self._seq_active[self._seq_index].replace('.traindat', '')
                print(f"  [SEQ] switching: {old_name} -> {new_name} "
                      f"({self._seq_index + 1}/{len(self._seq_active)})")
            return fname
        names = self._file_list
        w = [self._weights.get(n, 1.0) for n in names]
        if sum(w) <= 0:
            return random.choice(names)
        return random.choices(names, weights=w, k=1)[0]

    def sample_batch(
        self,
        n_samples: int,
        seq_len: int,
        num_bits: int,
        seed: int = None,
        return_mask: bool = False,
        byte_token_mode: bool = False,
        binary_bits_mode: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor],
               Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Sample a batch from a weighted-random .traindat file.

        Args:
            return_mask: If True, returns (x, y, mask) with Golden Disc role mask.
                mask is [n_samples, seq_len] uint8 with role of each target byte.
            byte_token_mode: If True, returns (x, y) as [B, seq] Long tensors
                of raw byte values (0-255). Each position = 1 discrete byte token.
            binary_bits_mode: If True, returns (x, y) as [B, seq, 8] float tensors.
                Each byte decomposed to 8 binary bits (0.0/1.0). 1 byte per position.

        Returns (x, y) or (x, y, mask) if return_mask=True.
        """
        filename = self.pick_file()
        corpus = self._load_corpus(filename)
        if byte_token_mode:
            return generate_batch_byte_tokens(corpus, n_samples, seq_len, seed)
        if binary_bits_mode:
            # Always returns (x, y, mask) — mask is [B, T, num_bits] float
            return generate_batch_binary_bits(corpus, n_samples, seq_len, num_bits, seed)
        rm = self._get_role_map(filename) if return_mask else None
        return generate_batch_from_bytes(corpus, n_samples, seq_len, num_bits, seed, role_map=rm)


# ---------------------------------------------------------------------------
# ThemeLoader — JSONL paired training (dialogue mode)
# ---------------------------------------------------------------------------

def _encode_theme_field(value: str, encoding: str) -> bytes:
    """Convert a theme pair field to raw bytes using the declared encoding."""
    if encoding == "utf8":
        return value.encode("utf-8")
    elif encoding == "hex":
        return bytes.fromhex(value)
    elif encoding == "base64":
        return base64.b64decode(value)
    else:
        raise ValueError(f"unknown encoding: {encoding}")


def _text_to_bits(text: str, num_bits: int, encoding: str = "utf8") -> np.ndarray:
    """Convert a text field to a binary bit vector of length num_bits.

    Encodes text → bytes → zero-padded to bytes_per_pos → unpackbits.
    Returns float32 array of shape (num_bits,) with values 0.0 / 1.0.
    """
    raw = _encode_theme_field(text, encoding)
    bytes_per_pos = num_bits // 8

    # Pad or truncate to exactly bytes_per_pos bytes
    if len(raw) < bytes_per_pos:
        raw = raw + b'\x00' * (bytes_per_pos - len(raw))
    else:
        raw = raw[:bytes_per_pos]

    arr = np.frombuffer(raw, dtype=np.uint8)
    bits = np.unpackbits(arr).astype(np.float32)

    # unpackbits may produce more bits than needed if bytes_per_pos * 8 > num_bits
    return bits[:num_bits]


class ThemeLoader:
    """Load JSONL theme files for dialogue-mode paired training.

    Reads .jsonl files from a themes directory. Each file has a header line
    followed by input→output pairs with state ("c"=curriculum, "g"=gist)
    and difficulty (1-6).

    Dialogue mode: seq_len positions alternate input/output.
    For seq_len=6: positions 0,2,4 are inputs; 1,3,5 are outputs.
    Loss is computed only on output positions (returned via mask).

    Mixing rule: ONE active theme provides curriculum pairs.
    ALL themes contribute their gist pairs, mixed in equally.
    """

    def __init__(
        self,
        themes_dir: str,
        num_bits: int = 6184,
        active_theme: Optional[str] = None,
        gist_ratio: float = 0.10,
    ):
        """
        Args:
            themes_dir: Path to directory containing .jsonl theme files.
            num_bits: Bits per position (must be multiple of 8).
            active_theme: Theme name for curriculum pairs. If None, uses first found.
            gist_ratio: Fraction of each batch drawn from gist pairs (0.0-1.0).
        """
        assert num_bits % 8 == 0, f"num_bits must be multiple of 8, got {num_bits}"
        self.themes_dir = Path(themes_dir)
        self.num_bits = num_bits
        self.gist_ratio = gist_ratio

        # Storage: theme_name -> {meta, curriculum, gist, encoding}
        self._themes: Dict[str, dict] = {}
        self._active_theme: Optional[str] = active_theme

        self._load_all_themes()

        if self._active_theme is None and self._themes:
            self._active_theme = sorted(self._themes.keys())[0]

    def _load_all_themes(self):
        """Scan themes directory and load all .jsonl files."""
        if not self.themes_dir.exists():
            raise FileNotFoundError(f"Themes directory not found: {self.themes_dir}")

        files = sorted(self.themes_dir.glob("*.jsonl"))
        if not files:
            raise FileNotFoundError(f"No .jsonl files in {self.themes_dir}")

        for fp in files:
            self._load_theme_file(fp)

    def _load_theme_file(self, filepath: Path):
        """Parse a single .jsonl theme file into curriculum + gist lists."""
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if not lines:
            return

        meta = json.loads(lines[0])
        if not meta.get("_meta"):
            return

        theme_name = meta.get("theme", filepath.stem)
        encoding = meta.get("encoding", "utf8")

        curriculum = []
        gist = []

        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            pair = json.loads(line)
            if pair.get("_meta"):
                continue

            entry = {
                "id": pair["id"],
                "in": pair["in"],
                "out": pair["out"],
                "d": pair.get("d", 1),
                "set": pair.get("set"),
            }

            if pair["s"] == "g":
                gist.append(entry)
            else:
                curriculum.append(entry)

        self._themes[theme_name] = {
            "meta": meta,
            "encoding": encoding,
            "curriculum": curriculum,
            "gist": gist,
            "filepath": str(filepath),
        }

    @property
    def active_theme(self) -> Optional[str]:
        return self._active_theme

    @active_theme.setter
    def active_theme(self, name: str):
        if name not in self._themes:
            raise ValueError(f"Theme '{name}' not found. Available: {list(self._themes.keys())}")
        self._active_theme = name

    @property
    def theme_names(self) -> List[str]:
        return sorted(self._themes.keys())

    def stats(self) -> dict:
        """Return summary statistics for all loaded themes."""
        result = {}
        for name, t in self._themes.items():
            result[name] = {
                "curriculum": len(t["curriculum"]),
                "gist": len(t["gist"]),
                "encoding": t["encoding"],
                "active": name == self._active_theme,
            }
        return result

    def _all_gist(self) -> List[Tuple[dict, str]]:
        """Collect all gist pairs from all themes, with their encoding."""
        out = []
        for name, t in self._themes.items():
            enc = t["encoding"]
            for pair in t["gist"]:
                out.append((pair, enc))
        return out

    def sample_batch(
        self,
        n_samples: int,
        seq_len: int = 6,
        seed: Optional[int] = None,
        difficulty_max: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a batch of dialogue-mode sequences.

        Each sequence has seq_len positions, alternating input/output.
        For seq_len=6: [in0, out0, in1, out1, in2, out2]
        The model sees all seq_len positions as input x.
        Target y is the same as x (next-position prediction is identity in dialogue mode).
        Loss mask marks which positions to compute loss on (output positions only).

        Args:
            n_samples: Batch size.
            seq_len: Sequence length (must be even). Default 6.
            seed: Random seed for reproducibility.
            difficulty_max: If set, only use curriculum pairs with d <= this value.

        Returns:
            x: [n_samples, seq_len, num_bits] float32 — input positions
            y: [n_samples, seq_len, num_bits] float32 — target positions
            loss_mask: [n_samples, seq_len] float32 — 1.0 for output positions, 0.0 for input
        """
        assert seq_len % 2 == 0, f"seq_len must be even for dialogue mode, got {seq_len}"
        n_pairs_per_seq = seq_len // 2

        rng = random.Random(seed)

        if self._active_theme is None:
            raise RuntimeError("No active theme set and no themes loaded")

        theme = self._themes[self._active_theme]
        curriculum = theme["curriculum"]
        curriculum_enc = theme["encoding"]

        if difficulty_max is not None:
            curriculum = [p for p in curriculum if p["d"] <= difficulty_max]

        all_gist = self._all_gist()

        if not curriculum and not all_gist:
            raise RuntimeError(f"No pairs available (curriculum empty, no gist)")

        # Pre-allocate output tensors
        x = torch.zeros(n_samples, seq_len, self.num_bits, dtype=torch.float32)
        y = torch.zeros(n_samples, seq_len, self.num_bits, dtype=torch.float32)
        loss_mask = torch.zeros(n_samples, seq_len, dtype=torch.float32)

        # Mark output positions in loss mask (positions 1, 3, 5, ...)
        for t in range(seq_len):
            if t % 2 == 1:
                loss_mask[:, t] = 1.0

        for i in range(n_samples):
            pairs_for_seq = []

            for _ in range(n_pairs_per_seq):
                # Decide: gist or curriculum for this pair
                use_gist = (rng.random() < self.gist_ratio) and all_gist
                if use_gist:
                    pair, enc = rng.choice(all_gist)
                elif curriculum:
                    pair = rng.choice(curriculum)
                    enc = curriculum_enc
                elif all_gist:
                    pair, enc = rng.choice(all_gist)
                else:
                    continue  # shouldn't reach here

                pairs_for_seq.append((pair, enc))

            # Fill sequence positions: input at even slots, output at odd slots
            for j, (pair, enc) in enumerate(pairs_for_seq):
                in_bits = _text_to_bits(pair["in"], self.num_bits, enc)
                out_bits = _text_to_bits(pair["out"], self.num_bits, enc)
                x[i, j * 2, :] = torch.from_numpy(in_bits)
                x[i, j * 2 + 1, :] = torch.from_numpy(out_bits)

        # y = shifted x: y[t] = x[t+1], y[last] = 0
        # Autoregressive target: model at position t predicts position t+1
        # Loss mask ensures we only train on output positions (1, 3, 5)
        y[:, :seq_len - 1, :] = x[:, 1:, :]
        # y[:, seq_len-1, :] stays zero (already initialized)

        return x, y, loss_mask

    def reload(self):
        """Re-scan the themes directory and reload all files."""
        self._themes.clear()
        self._load_all_themes()
        if self._active_theme and self._active_theme not in self._themes:
            self._active_theme = sorted(self._themes.keys())[0] if self._themes else None
