"""Exact, space-aware word tokenizer (GPT protocol).

Token types, in priority order:
  ▁WORD           — a word with space-prefix (SentencePiece style)
  WORD            — same word without leading space (start of text)
  PUNCT           — single punctuation character (.,;:!?()[]{}"'/\\<>=+-*&|^~%$#@`)
  WS_RUN_n        — run of whitespace (tabs/newlines/multiple spaces) length n (up to 8, then byte fallback)
  BYTE_0xXX       — literal single byte fallback (256 slots reserved)

Guarantees:
  - encode(text) -> list[int]
  - decode(ids)  -> bytes
  - decode(encode(text)) == text   (lossless)

Vocab layout (ids in order):
  [0 .. 255]         : BYTE_0xXX (always present)
  [256 .. 256+P-1]   : PUNCT single-char tokens (P of them)
  [256+P .. 256+P+W-1] : WS_RUN_2 .. WS_RUN_8 (7 slots)
  [vocab_start ..]   : learned ▁WORD / WORD tokens sorted by frequency

No neural net yet. This is Phase 1 of the Brain-side pipeline.
"""
from __future__ import annotations
import argparse, json, re, sys, time
from collections import Counter
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

CORPUS_CANDIDATES = [
    Path("instnct-core/tests/fixtures/code_corpus.txt"),
    Path("instnct-core/tests/fixtures/alice_corpus.txt"),
    Path("instnct-core/tests/fixtures/beta_smoke_corpus.txt"),
]
OUT_DIR = Path("output/word_tokenizer")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Punctuation chars that get their own single-char token
PUNCT_CHARS = b".,;:!?()[]{}\"'/\\<>=+-*&|^~%$#@`"
# Word chars: letters, digits, underscore
WORD_CHARS = set(b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
WS_CHARS = set(b" \t\n\r")
MAX_WS_RUN = 8  # WS_RUN_2 .. WS_RUN_8


def is_word(b): return b in WORD_CHARS
def is_ws(b): return b in WS_CHARS
def is_punct(b): return b in PUNCT_CHARS


class Tokenizer:
    def __init__(self, word_tokens=None):
        # Reserved slots
        self.BYTE_BASE = 0          # 0..255
        self.PUNCT_BASE = 256       # 256..256+P-1
        self.punct_list = list(PUNCT_CHARS)
        self.WS_BASE = self.PUNCT_BASE + len(self.punct_list)
        # WS_RUN_2..WS_RUN_MAX_WS_RUN
        self.WORD_BASE = self.WS_BASE + (MAX_WS_RUN - 1)

        self.id_to_token = {}  # id -> tuple (kind, payload)
        # Build reserved
        for i, b in enumerate(range(256)):
            self.id_to_token[i] = ("BYTE", b)
        for i, c in enumerate(self.punct_list):
            self.id_to_token[self.PUNCT_BASE + i] = ("PUNCT", c)
        for i, n in enumerate(range(2, MAX_WS_RUN + 1)):
            self.id_to_token[self.WS_BASE + i] = ("WS_RUN", n)

        self.punct_to_id = {c: self.PUNCT_BASE + i for i, c in enumerate(self.punct_list)}

        # Learned ▁WORD / WORD tokens
        # word_tokens: list of (token_bytes, has_space_prefix: bool) sorted by desired id order
        self.word_to_id = {}   # (bytes, has_prefix) -> id
        if word_tokens:
            for i, (tok_bytes, has_prefix) in enumerate(word_tokens):
                tid = self.WORD_BASE + i
                self.id_to_token[tid] = ("WORD", (tok_bytes, has_prefix))
                self.word_to_id[(bytes(tok_bytes), has_prefix)] = tid

    @property
    def vocab_size(self):
        return len(self.id_to_token)

    def encode(self, data):
        """Greedy encode bytes -> list[int] ids.

        Algorithm:
          walk through `data`:
            if space (+ optional leading-space bytes): try to match ▁WORD
            elif word char run: try match WORD
            elif punct: PUNCT token
            elif whitespace (tab/nl/run): WS_RUN or BYTE fallback
            else: BYTE fallback
        """
        if isinstance(data, str):
            data = data.encode("utf-8")
        ids = []
        i = 0
        n = len(data)
        while i < n:
            b = data[i]
            # --- space-prefixed word ---
            if b == 0x20:  # single space
                j = i + 1
                if j < n and is_word(data[j]):
                    # collect word bytes
                    k = j
                    while k < n and is_word(data[k]):
                        k += 1
                    word = data[j:k]
                    key = (word, True)
                    if key in self.word_to_id:
                        ids.append(self.word_to_id[key])
                        i = k
                        continue
                    # fallback: space as BYTE, then try plain WORD
                    ids.append(self.BYTE_BASE + 0x20)
                    key2 = (word, False)
                    if key2 in self.word_to_id:
                        ids.append(self.word_to_id[key2])
                    else:
                        # bytes of the word
                        for bb in word:
                            ids.append(self.BYTE_BASE + bb)
                    i = k
                    continue
                # just a space -> fall through to WS handling
            # --- whitespace run (space / tab / nl / cr) ---
            if is_ws(b):
                k = i
                while k < n and is_ws(data[k]) and (k - i) < MAX_WS_RUN:
                    k += 1
                run_len = k - i
                # If it's all space and we'd rather emit WS_RUN (>=2), do so
                all_same = all(data[m] == b for m in range(i, k))
                if run_len >= 2 and all_same and b == 0x20:
                    ids.append(self.WS_BASE + (run_len - 2))
                    i = k
                    continue
                # Otherwise emit bytes individually
                for m in range(i, k):
                    ids.append(self.BYTE_BASE + data[m])
                i = k
                continue
            # --- word (no space prefix) ---
            if is_word(b):
                k = i
                while k < n and is_word(data[k]):
                    k += 1
                word = data[i:k]
                key = (word, False)
                if key in self.word_to_id:
                    ids.append(self.word_to_id[key])
                else:
                    for bb in word:
                        ids.append(self.BYTE_BASE + bb)
                i = k
                continue
            # --- punctuation ---
            if is_punct(b):
                ids.append(self.punct_to_id[b])
                i += 1
                continue
            # --- raw byte fallback ---
            ids.append(self.BYTE_BASE + b)
            i += 1
        return ids

    def decode(self, ids):
        """Decode list[int] -> bytes.  Reverse of encode, lossless."""
        out = bytearray()
        for tid in ids:
            kind, payload = self.id_to_token[tid]
            if kind == "BYTE":
                out.append(payload)
            elif kind == "PUNCT":
                out.append(payload)
            elif kind == "WS_RUN":
                out.extend(b" " * payload)
            elif kind == "WORD":
                word_bytes, has_prefix = payload
                if has_prefix:
                    out.append(0x20)
                out.extend(word_bytes)
            else:
                raise ValueError(f"unknown token kind: {kind}")
        return bytes(out)

    def to_dict(self):
        items = []
        for i in range(self.vocab_size):
            kind, payload = self.id_to_token[i]
            if kind == "BYTE":
                items.append({"id": i, "kind": "BYTE", "byte": payload})
            elif kind == "PUNCT":
                items.append({"id": i, "kind": "PUNCT", "char": chr(payload)})
            elif kind == "WS_RUN":
                items.append({"id": i, "kind": "WS_RUN", "len": payload})
            else:
                word_bytes, has_prefix = payload
                items.append({
                    "id": i, "kind": "WORD",
                    "text": word_bytes.decode("utf-8", errors="replace"),
                    "space_prefix": has_prefix,
                })
        return items


# --- Vocab building from corpus ---

def scan_words(corpus_bytes):
    """Count (word_bytes, has_space_prefix) occurrences across corpus."""
    counter = Counter()
    n = len(corpus_bytes)
    i = 0
    while i < n:
        b = corpus_bytes[i]
        if b == 0x20 and i + 1 < n and is_word(corpus_bytes[i + 1]):
            j = i + 1
            k = j
            while k < n and is_word(corpus_bytes[k]):
                k += 1
            word = bytes(corpus_bytes[j:k])
            counter[(word, True)] += 1
            i = k
            continue
        if is_word(b):
            j = i
            while j < n and is_word(corpus_bytes[j]):
                j += 1
            word = bytes(corpus_bytes[i:j])
            counter[(word, False)] += 1
            i = j
            continue
        i += 1
    return counter


def build_tokenizer(corpus_bytes, vocab_size=16_000):
    counter = scan_words(corpus_bytes)
    sorted_items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    word_list = [k for k, _ in sorted_items[:vocab_size]]
    tk = Tokenizer(word_tokens=word_list)
    return tk, counter


def load_corpus(paths=None):
    paths = paths or CORPUS_CANDIDATES
    chunks = []
    for p in paths:
        if p.is_file():
            b = p.read_bytes()
            chunks.append(b)
            print(f"  [corpus] {p} -> {len(b)} bytes")
    if not chunks:
        raise FileNotFoundError("No corpus found")
    return b"\n".join(chunks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", nargs="+", type=Path, default=None)
    parser.add_argument("--vocab-size", type=int, default=16_000)
    parser.add_argument("--demo", type=str, default="The cat sleeps peacefully on the warm mat.")
    args = parser.parse_args()

    print("=" * 70)
    print("WORD TOKENIZER — exact, space-aware (SentencePiece-style)")
    print("=" * 70)

    print(f"\n[1] Loading corpus...")
    t0 = time.time()
    corpus = load_corpus(args.corpus)
    print(f"    total: {len(corpus)} bytes  ({time.time()-t0:.1f}s)")

    print(f"\n[2] Building vocab (target {args.vocab_size} word tokens)...")
    t0 = time.time()
    tk, counter = build_tokenizer(corpus, vocab_size=args.vocab_size)
    print(f"    unique (word, prefix) combos: {len(counter)}")
    print(f"    vocab size total: {tk.vocab_size} (256 BYTE + {len(tk.punct_list)} PUNCT + {MAX_WS_RUN-1} WS_RUN + {tk.vocab_size - tk.WORD_BASE} WORD)")
    print(f"    ({time.time()-t0:.1f}s)")

    print(f"\n[3] Top 30 WORD tokens:")
    print(f"    {'ID':>5} {'count':>9}  prefix  text")
    sorted_items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:30]
    for i, ((word, has_prefix), c) in enumerate(sorted_items):
        tid = tk.WORD_BASE + i
        shown = word.decode("utf-8", errors="replace")
        if not shown.isprintable():
            shown = "<bin:" + word.hex() + ">"
        marker = "▁" if has_prefix else " "
        print(f"    {tid:>5} {c:>9}  {marker}       {shown}")

    # --- Round-trip sanity check ---
    print(f"\n[4] Round-trip sanity — encoding a sample from corpus...")
    sample = corpus[:500]
    ids = tk.encode(sample)
    back = tk.decode(ids)
    match = back == sample
    print(f"    sample {len(sample)} bytes -> {len(ids)} ids -> {len(back)} bytes  match={match}")
    if not match:
        # find first divergence
        for i in range(min(len(sample), len(back))):
            if sample[i] != back[i]:
                print(f"    FIRST DIFF at byte {i}: orig {sample[i]:#04x} vs decoded {back[i]:#04x}")
                print(f"    ...context orig: {sample[max(0,i-10):i+10]!r}")
                print(f"    ...context back: {back[max(0,i-10):i+10]!r}")
                break

    # --- Demo on user's sentence ---
    print(f"\n[5] Demo: {args.demo!r}")
    demo_bytes = args.demo.encode("utf-8")
    demo_ids = tk.encode(demo_bytes)
    demo_back = tk.decode(demo_ids)
    print(f"    ids: {demo_ids}")
    print(f"    format: _" + "_".join(str(i) for i in demo_ids) + "_")
    print(f"    decoded: {demo_back!r}")
    print(f"    exact: {demo_back == demo_bytes}")
    # token-by-token
    print(f"    token breakdown:")
    for tid in demo_ids:
        kind, payload = tk.id_to_token[tid]
        if kind == "WORD":
            word, has_prefix = payload
            marker = "▁" if has_prefix else ""
            print(f"      {tid:>5}  WORD     {marker}{word.decode('utf-8','replace')}")
        elif kind == "BYTE":
            print(f"      {tid:>5}  BYTE     {payload:#04x} ({chr(payload) if 32<=payload<127 else '.'})")
        elif kind == "PUNCT":
            print(f"      {tid:>5}  PUNCT    {chr(payload)}")
        elif kind == "WS_RUN":
            print(f"      {tid:>5}  WS_RUN   {payload} spaces")

    # --- Compression ratio ---
    print(f"\n[6] Compression ratio on full corpus:")
    t0 = time.time()
    all_ids = tk.encode(corpus)
    dt = time.time() - t0
    # Ceiling bits per token: log2(vocab_size) rounded up
    import math
    bits_per_token = math.ceil(math.log2(tk.vocab_size))
    token_bytes_ceil = len(all_ids) * bits_per_token / 8
    print(f"    corpus bytes: {len(corpus)}")
    print(f"    token count: {len(all_ids)}  ({dt:.1f}s)")
    print(f"    bits per token (fixed): {bits_per_token}")
    print(f"    tokens total size (fixed-width): {token_bytes_ceil:.0f} bytes ({100*token_bytes_ceil/len(corpus):.2f}% of raw)")
    print(f"    avg tokens per byte: {len(all_ids)/len(corpus):.3f}")

    # --- Save ---
    vocab_path = OUT_DIR / f"vocab_{args.vocab_size}.json"
    vocab_path.write_text(json.dumps(tk.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Saved vocab: {vocab_path}  ({vocab_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
