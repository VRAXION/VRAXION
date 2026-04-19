"""Pack the exact single-W hybrid-K artifact with a practical Huffman bitstream.

This is a serializer/deserializer for the already exact generator artifact:
  output/merger_single_w_hybrid_k/final_hybrid.json

Format choices:
  - Fixed architecture / component order; shapes are implicit.
  - W, b1, c19_c, c19_rho use:
      fp16 generators
      1-bit mode bitmap (encoded vs fallback)
      1-bit sign bitmap for encoded cells
      canonical Huffman on coef stream
      canonical Huffman on generator-index stream
      fp16 fallback stream
  - b2 is stored raw fp16 because the hybrid form is larger there.

The script writes a compact binary, decodes it back, and verifies exact 100% lossless
on the full 65536 pair space.
"""
from __future__ import annotations

import collections
import heapq
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import sys

sys.path.insert(0, str(Path(__file__).parent))
from diag_byte_pair_merger_single_w_mirror import SingleWMirror, load_byte_pairs, metrics, DEVICE


SRC = Path("output/merger_single_w_hybrid_k/final_hybrid.json")
OUT_DIR = Path("output/merger_single_w_huffman_pack")
OUT_DIR.mkdir(parents=True, exist_ok=True)

COMPONENTS = [
    ("W", 2592, False),
    ("b1", 81, False),
    ("b2", 32, True),
    ("c19_c", 81, False),
    ("c19_rho", 81, False),
]


class BitWriter:
    def __init__(self) -> None:
        self.buf = bytearray()
        self.acc = 0
        self.nbits = 0

    def write(self, code: int, length: int) -> None:
        while length > 0:
            take = min(8 - self.nbits, length)
            shift = length - take
            chunk = (code >> shift) & ((1 << take) - 1)
            self.acc = (self.acc << take) | chunk
            self.nbits += take
            length -= take
            if self.nbits == 8:
                self.buf.append(self.acc)
                self.acc = 0
                self.nbits = 0

    def finish(self) -> bytes:
        if self.nbits:
            self.buf.append(self.acc << (8 - self.nbits))
            self.acc = 0
            self.nbits = 0
        return bytes(self.buf)


class BitReader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0
        self.acc = 0
        self.nbits = 0

    def read(self, length: int) -> int:
        out = 0
        while length > 0:
            if self.nbits == 0:
                if self.pos >= len(self.data):
                    raise EOFError("bitstream exhausted")
                self.acc = self.data[self.pos]
                self.pos += 1
                self.nbits = 8
            take = min(self.nbits, length)
            shift = self.nbits - take
            chunk = (self.acc >> shift) & ((1 << take) - 1)
            out = (out << take) | chunk
            self.nbits -= take
            self.acc &= (1 << self.nbits) - 1 if self.nbits else 0
            length -= take
        return out


def pack_bits(bits: list[int]) -> bytes:
    w = BitWriter()
    for b in bits:
        w.write(int(b), 1)
    return w.finish()


def unpack_bits(data: bytes, n: int) -> list[int]:
    r = BitReader(data)
    return [r.read(1) for _ in range(n)]


def pack_nibbles(vals: list[int]) -> bytes:
    out = bytearray()
    for i in range(0, len(vals), 2):
        lo = vals[i] & 0xF
        hi = vals[i + 1] & 0xF if i + 1 < len(vals) else 0
        out.append(lo | (hi << 4))
    return bytes(out)


def unpack_nibbles(data: bytes, n: int) -> list[int]:
    out: list[int] = []
    for b in data:
        out.append(b & 0xF)
        if len(out) == n:
            break
        out.append((b >> 4) & 0xF)
        if len(out) == n:
            break
    return out


def huffman_lengths(counter: collections.Counter[int]) -> dict[int, int]:
    items = [(w, {sym: 0}) for sym, w in counter.items() if w > 0]
    if not items:
        return {}
    if len(items) == 1:
        sym = next(iter(items[0][1]))
        return {sym: 1}
    heap: list[tuple[int, int, dict[int, int]]] = []
    uid = 0
    for w, mapping in items:
        heap.append((w, uid, mapping))
        uid += 1
    heapq.heapify(heap)
    while len(heap) > 1:
        w1, _, m1 = heapq.heappop(heap)
        w2, _, m2 = heapq.heappop(heap)
        merged: dict[int, int] = {}
        for k, v in m1.items():
            merged[k] = v + 1
        for k, v in m2.items():
            merged[k] = v + 1
        heapq.heappush(heap, (w1 + w2, uid, merged))
        uid += 1
    return heap[0][2]


def canonical_codes(lengths: dict[int, int]) -> dict[int, tuple[int, int]]:
    if not lengths:
        return {}
    items = sorted(((ln, sym) for sym, ln in lengths.items()), key=lambda x: (x[0], x[1]))
    code = 0
    prev_len = items[0][0]
    out: dict[int, tuple[int, int]] = {}
    for ln, sym in items:
        code <<= ln - prev_len
        out[sym] = (code, ln)
        code += 1
        prev_len = ln
    return out


def canonical_decode_table(lengths: dict[int, int]) -> dict[tuple[int, int], int]:
    return {(ln, code): sym for sym, (code, ln) in canonical_codes(lengths).items()}


def encode_symbols(symbols: list[int], lengths: dict[int, int]) -> bytes:
    codes = canonical_codes(lengths)
    w = BitWriter()
    for s in symbols:
        code, ln = codes[s]
        w.write(code, ln)
    return w.finish()


def decode_symbols(data: bytes, n: int, lengths: dict[int, int]) -> list[int]:
    if n == 0:
        return []
    table = canonical_decode_table(lengths)
    max_len = max(lengths.values())
    r = BitReader(data)
    out: list[int] = []
    for _ in range(n):
        code = 0
        matched = False
        for ln in range(1, max_len + 1):
            code = (code << 1) | r.read(1)
            key = (ln, code)
            if key in table:
                out.append(table[key])
                matched = True
                break
        if not matched:
            raise ValueError("invalid Huffman stream")
    return out


@dataclass
class PackedComponent:
    name: str
    data: bytes
    decoded: np.ndarray


def pack_component(name: str, n: int, raw_only: bool, artifact: dict) -> PackedComponent:
    arr = np.array(artifact[f"{name}_final"], dtype=np.float32).reshape(-1)
    if raw_only:
        raw = arr.astype(np.float16).view(np.uint16).tobytes()
        decoded = np.frombuffer(raw, dtype=np.uint16).view(np.float16).astype(np.float32)
        return PackedComponent(name, raw, decoded)

    enc = {int(k): tuple(v) for k, v in artifact["encodings"][name].items()}
    gens = np.array(artifact["generators"][name], dtype=np.float32).astype(np.float16)
    gens16 = gens.view(np.uint16)

    mode_bits: list[int] = []
    sign_bits: list[int] = []
    coef_syms: list[int] = []
    idx_syms: list[int] = []
    fallback_u16: list[int] = []
    decoded = np.empty(n, dtype=np.float32)

    for i in range(n):
        if i in enc:
            s, c, g = enc[i]
            mode_bits.append(1)
            sign_bits.append(1 if s > 0 else 0)
            coef_syms.append(int(c))
            idx_syms.append(int(g))
            decoded[i] = np.float32((1 if s > 0 else -1) * int(c) * np.float32(gens[g]))
        else:
            mode_bits.append(0)
            u16 = np.float16(arr[i]).view(np.uint16).item()
            fallback_u16.append(u16)
            decoded[i] = np.float32(np.array([u16], dtype=np.uint16).view(np.float16)[0])

    mode_blob = pack_bits(mode_bits)
    sign_blob = pack_bits(sign_bits)

    coef_counter = collections.Counter(coef_syms)
    idx_counter = collections.Counter(idx_syms)
    coef_lengths = huffman_lengths(coef_counter)
    idx_lengths = huffman_lengths(idx_counter)
    coef_len_nibbles = pack_nibbles([coef_lengths.get(sym, 0) for sym in range(1, 8)])
    idx_len_nibbles = pack_nibbles([idx_lengths.get(sym, 0) for sym in range(len(gens))])
    coef_blob = encode_symbols(coef_syms, coef_lengths)
    idx_blob = encode_symbols(idx_syms, idx_lengths)
    fb_blob = np.array(fallback_u16, dtype=np.uint16).tobytes()

    # Fixed-format section:
    #   G:uint8
    #   gens:uint16[G]
    #   mode bitmap
    #   sign bitmap
    #   coef lengths (7 nibbles -> 4B)
    #   idx lengths (G nibbles -> ceil(G/2)B)
    #   coef_blob_len:uint16
    #   idx_blob_len:uint16
    #   coef blob
    #   idx blob
    #   fallback fp16 stream
    out = bytearray()
    out.append(len(gens))
    out += gens16.tobytes()
    out += mode_blob
    out += sign_blob
    out += coef_len_nibbles
    out += idx_len_nibbles
    out += struct.pack("<HH", len(coef_blob), len(idx_blob))
    out += coef_blob
    out += idx_blob
    out += fb_blob
    return PackedComponent(name, bytes(out), decoded)


def unpack_component(name: str, n: int, raw_only: bool, payload: bytes, offset: int) -> tuple[np.ndarray, int]:
    if raw_only:
        end = offset + n * 2
        arr = np.frombuffer(payload[offset:end], dtype=np.uint16).view(np.float16).astype(np.float32)
        return arr, end

    g = payload[offset]
    offset += 1
    gens_end = offset + g * 2
    gens = np.frombuffer(payload[offset:gens_end], dtype=np.uint16).view(np.float16).astype(np.float32)
    offset = gens_end

    mode_nbytes = math.ceil(n / 8)
    mode_bits = unpack_bits(payload[offset:offset + mode_nbytes], n)
    offset += mode_nbytes
    n_enc = sum(mode_bits)
    n_fb = n - n_enc

    sign_nbytes = math.ceil(n_enc / 8)
    sign_bits = unpack_bits(payload[offset:offset + sign_nbytes], n_enc)
    offset += sign_nbytes

    coef_len_nbytes = math.ceil(7 / 2)
    coef_lens_raw = unpack_nibbles(payload[offset:offset + coef_len_nbytes], 7)
    offset += coef_len_nbytes
    coef_lengths = {sym: ln for sym, ln in zip(range(1, 8), coef_lens_raw) if ln > 0}

    idx_len_nbytes = math.ceil(g / 2)
    idx_lens_raw = unpack_nibbles(payload[offset:offset + idx_len_nbytes], g)
    offset += idx_len_nbytes
    idx_lengths = {sym: ln for sym, ln in zip(range(g), idx_lens_raw) if ln > 0}

    coef_nbytes, idx_nbytes = struct.unpack("<HH", payload[offset:offset + 4])
    offset += 4
    coef_blob = payload[offset:offset + coef_nbytes]
    offset += coef_nbytes
    idx_blob = payload[offset:offset + idx_nbytes]
    offset += idx_nbytes
    coef_syms = decode_symbols(coef_blob, n_enc, coef_lengths)
    idx_syms = decode_symbols(idx_blob, n_enc, idx_lengths)

    fb_end = offset + n_fb * 2
    fallback = np.frombuffer(payload[offset:fb_end], dtype=np.uint16).view(np.float16).astype(np.float32)
    offset = fb_end

    arr = np.empty(n, dtype=np.float32)
    ie = 0
    ifb = 0
    for i, m in enumerate(mode_bits):
        if m:
            s = 1 if sign_bits[ie] else -1
            c = coef_syms[ie]
            gi = idx_syms[ie]
            arr[i] = np.float32(s * c * gens[gi])
            ie += 1
        else:
            arr[i] = fallback[ifb]
            ifb += 1
    return arr, offset


def build_model_from_state(state: dict[str, np.ndarray]) -> SingleWMirror:
    model = SingleWMirror(32, 81, DEVICE).to(DEVICE)
    with torch.no_grad():
        model.W.copy_(torch.tensor(state["W"].reshape(32, 81), dtype=torch.float32, device=DEVICE))
        model.b1.copy_(torch.tensor(state["b1"], dtype=torch.float32, device=DEVICE))
        model.b2.copy_(torch.tensor(state["b2"], dtype=torch.float32, device=DEVICE))
        model.c19.c_raw.copy_(torch.tensor(state["c19_c"], dtype=torch.float32, device=DEVICE))
        model.c19.rho_raw.copy_(torch.tensor(state["c19_rho"], dtype=torch.float32, device=DEVICE))
    return model


def main() -> None:
    artifact = json.loads(SRC.read_text())

    parts: list[PackedComponent] = []
    for name, n, raw_only in COMPONENTS:
        parts.append(pack_component(name, n, raw_only, artifact))

    blob = bytearray()
    blob += b"VGH1"
    for part in parts:
        blob += part.data

    packed_path = OUT_DIR / "packed_model.bin"
    packed_path.write_bytes(blob)

    payload = packed_path.read_bytes()
    if payload[:4] != b"VGH1":
        raise ValueError("bad magic")
    offset = 4
    state: dict[str, np.ndarray] = {}
    for name, n, raw_only in COMPONENTS:
        arr, offset = unpack_component(name, n, raw_only, payload, offset)
        state[name] = arr
    if offset != len(payload):
        raise ValueError(f"trailing bytes: {len(payload) - offset}")

    model = build_model_from_state(state)
    data = load_byte_pairs().to(DEVICE)
    lossless, per_dim, bad_pairs = metrics(model, data)

    summary = {
        "packed_bytes": len(payload),
        "packed_kb": len(payload) / 1024,
        "lossless": float(lossless),
        "per_dim": float(per_dim),
        "bad_pairs": int(bad_pairs),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
