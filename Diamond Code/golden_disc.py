"""
Golden Disc — Universal Training Data Format for Diamond Code.

Byte-level structured data using ASCII control character separators:
    0x1C (FS) = document/database boundary
    0x1D (GS) = reserved
    0x1E (RS) = row/example boundary
    0x1F (US) = field separator (input -> output)

Type byte (one byte after FS): high nibble = modality, low nibble = complexity.

Usage:
    # Write
    writer = GoldenDiscWriter(modality=0x0, complexity=0x1)
    writer.add_row(b"3+4", b"7")
    writer.add_row(b"5+2", b"7")
    data = writer.to_bytes()

    # Parse
    blocks = GoldenDiscParser.parse(data)

    # Role map for answer masking
    role_map = build_role_map(data)
    # 0=separator, 1=question, 2=answer, 3=type_byte, 4=freetext
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Separator constants
# ---------------------------------------------------------------------------
FS = 0x1C  # File Separator   — document boundary
GS = 0x1D  # Group Separator  — reserved
RS = 0x1E  # Record Separator — row boundary
US = 0x1F  # Unit Separator   — field separator

SEPARATORS = frozenset({FS, GS, RS, US})

# ---------------------------------------------------------------------------
# Role map values
# ---------------------------------------------------------------------------
ROLE_SEPARATOR = 0
ROLE_QUESTION  = 1
ROLE_ANSWER    = 2
ROLE_TYPE_BYTE = 3
ROLE_FREETEXT  = 4

# ---------------------------------------------------------------------------
# Type byte: modality (high nibble) + complexity (low nibble)
# ---------------------------------------------------------------------------
MODALITY = {
    0x0: 'numeric',
    0x1: 'text',
    0x2: 'logic',
    0x3: 'code',
    0x4: 'audio',
    0x5: 'image',
    0x6: 'video',
    0x7: 'game',
    0x8: 'science',
    0x9: 'structured',
    0xA: 'dialogue',
    0xB: 'translation',
}

MODALITY_REV = {v: k for k, v in MODALITY.items()}


def encode_type_byte(modality: int, complexity: int) -> int:
    """Combine modality (0-F) and complexity (0-F) into one type byte."""
    if not (0 <= modality <= 0xF):
        raise ValueError(f"modality must be 0-15, got {modality}")
    if not (0 <= complexity <= 0xF):
        raise ValueError(f"complexity must be 0-15, got {complexity}")
    return (modality << 4) | complexity


def decode_type_byte(byte: int) -> Tuple[int, int]:
    """Decode type byte into (modality, complexity)."""
    return (byte >> 4) & 0xF, byte & 0xF


def modality_name(code: int) -> str:
    """Human-readable modality name."""
    return MODALITY.get(code, f'unknown_{code:X}')


# ---------------------------------------------------------------------------
# Content validation
# ---------------------------------------------------------------------------
def validate_content(data: bytes) -> bool:
    """Check that data contains no separator bytes (0x1C-0x1F).

    Raises ValueError with position of first violation.
    Returns True if clean.
    """
    for i, b in enumerate(data):
        if b in SEPARATORS:
            raise ValueError(
                f"Separator byte 0x{b:02X} found at position {i} in content"
            )
    return True


# ---------------------------------------------------------------------------
# GoldenDiscWriter
# ---------------------------------------------------------------------------
class GoldenDiscWriter:
    """Accumulates rows and serializes to Golden Disc byte format.

    For small datasets and validation. For 100MB+ files, use streaming
    generation (see generate_golden_discs.py).
    """

    def __init__(self, modality: int, complexity: int):
        self.type_byte = encode_type_byte(modality, complexity)
        self._rows: List[List[bytes]] = []
        self._num_fields: Optional[int] = None

    def add_row(self, *fields: bytes):
        """Add a row with one or more fields.

        All rows in a writer must have the same number of fields.
        Fields must not contain separator bytes.
        """
        if self._num_fields is None:
            self._num_fields = len(fields)
        elif len(fields) != self._num_fields:
            raise ValueError(
                f"Row has {len(fields)} fields, expected {self._num_fields}"
            )
        for i, f in enumerate(fields):
            validate_content(f)
        self._rows.append(list(fields))

    def to_bytes(self) -> bytearray:
        """Serialize to Golden Disc format: FS + type + (RS + fields joined by US)..."""
        out = bytearray()
        out.append(FS)
        out.append(self.type_byte)
        for row in self._rows:
            out.append(RS)
            for i, field in enumerate(row):
                if i > 0:
                    out.append(US)
                out.extend(field)
        return out

    @property
    def num_rows(self) -> int:
        return len(self._rows)

    @property
    def num_fields(self) -> Optional[int]:
        return self._num_fields


# ---------------------------------------------------------------------------
# GoldenDiscParser
# ---------------------------------------------------------------------------
class GoldenBlock:
    """One parsed document block (between FS markers)."""

    __slots__ = ('modality', 'complexity', 'type_byte', 'rows', 'num_fields')

    def __init__(self, type_byte: int, rows: List[List[bytes]]):
        self.type_byte = type_byte
        self.modality, self.complexity = decode_type_byte(type_byte)
        self.rows = rows
        self.num_fields = len(rows[0]) if rows else 0


class GoldenDiscParser:
    """Parse raw bytes into GoldenBlock objects."""

    @staticmethod
    def parse(data: bytes) -> List[GoldenBlock]:
        """Split on FS, then RS, then US. Returns list of GoldenBlock."""
        if not data or data[0] != FS:
            raise ValueError("Data must start with FS (0x1C)")

        blocks = []
        # Split into FS-delimited chunks (skip first empty split)
        raw_blocks = data.split(bytes([FS]))
        for raw in raw_blocks:
            if not raw:
                continue
            type_byte = raw[0]
            body = raw[1:]
            # Split on RS
            raw_rows = body.split(bytes([RS]))
            rows = []
            for raw_row in raw_rows:
                if not raw_row:
                    continue
                fields = raw_row.split(bytes([US]))
                rows.append(fields)
            if rows:
                # Validate consistent field count
                expected = len(rows[0])
                for i, row in enumerate(rows):
                    if len(row) != expected:
                        raise ValueError(
                            f"Block type 0x{type_byte:02X}: row {i} has "
                            f"{len(row)} fields, expected {expected}"
                        )
                blocks.append(GoldenBlock(type_byte, rows))
        return blocks


# ---------------------------------------------------------------------------
# Role map builder
# ---------------------------------------------------------------------------
def build_role_map(corpus: bytes) -> np.ndarray:
    """Build a byte-level role map for answer masking.

    Returns numpy uint8 array of same length as corpus:
        0 = separator byte (FS, GS, RS, US)
        1 = question byte (content after RS, before first US in row)
        2 = answer byte (content after US, before next RS or FS)
        3 = type byte (byte immediately after FS)
        4 = free text (content in rows with no US)

    For non-golden-disc data (no FS found): returns all ROLE_FREETEXT.
    """
    n = len(corpus)
    roles = np.full(n, ROLE_FREETEXT, dtype=np.uint8)

    if n == 0:
        return roles

    # Quick check: if no FS in corpus, it's not a golden disc
    if FS not in (corpus[i] for i in range(min(n, 100))):
        # Check more thoroughly but cheaply
        has_fs = False
        for i in range(n):
            if corpus[i] == FS:
                has_fs = True
                break
        if not has_fs:
            return roles  # All freetext

    # State machine scan
    # States: 'start', 'after_fs', 'question', 'answer', 'freetext_row'
    state = 'start'
    row_has_us = False

    for i in range(n):
        b = corpus[i]

        if b == FS:
            roles[i] = ROLE_SEPARATOR
            state = 'after_fs'
            row_has_us = False
        elif b == RS:
            roles[i] = ROLE_SEPARATOR
            state = 'question'
            row_has_us = False
        elif b == US:
            roles[i] = ROLE_SEPARATOR
            state = 'answer'
            row_has_us = True
        elif b == GS:
            roles[i] = ROLE_SEPARATOR
        elif state == 'after_fs':
            roles[i] = ROLE_TYPE_BYTE
            state = 'question'  # Next content is question
        elif state == 'question':
            roles[i] = ROLE_QUESTION
        elif state == 'answer':
            roles[i] = ROLE_ANSWER
        else:
            roles[i] = ROLE_FREETEXT

    return roles


def role_map_summary(role_map: np.ndarray) -> Dict[str, int]:
    """Count bytes per role."""
    names = {
        ROLE_SEPARATOR: 'separator',
        ROLE_QUESTION: 'question',
        ROLE_ANSWER: 'answer',
        ROLE_TYPE_BYTE: 'type_byte',
        ROLE_FREETEXT: 'freetext',
    }
    counts = {}
    for val, name in names.items():
        counts[name] = int((role_map == val).sum())
    return counts
