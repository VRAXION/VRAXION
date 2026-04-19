"""V1 word embedding scaffold — 32,294 vocab × 64 dim, random init.

This is the NEURAL channel infrastructure (not storage). We allocate a
random-init embedding table for the frozen V2 hybrid tokenizer, verify
it works end-to-end (text -> IDs -> [N, 64] tensor), and save the table
as both float32 and int8 quantized.

No training yet. Just scaffolding. Upper model will train the table.

Usage:
  python tools/diag_word_embedding_v1.py
  python tools/diag_word_embedding_v1.py --demo "The cat sleeps peacefully."
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).parent))
from diag_subword_tokenizer_exact import LexicalTokenizer
from diag_word_tokenizer_champion_freeze import load_tokenizer_from_json

CHAMPION_VOCAB = Path("output/word_tokenizer_champion/champion_vocab.json")
OUT_DIR = Path("output/word_embedding_v1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMBED_DIM = 64
SEED = 42


def allocate_random_table(vocab_size: int, dim: int, seed: int = SEED) -> torch.Tensor:
    """Xavier-init embedding table. Small values, zero mean."""
    g = torch.Generator().manual_seed(seed)
    std = 1.0 / (dim ** 0.5)  # common init scale
    return torch.empty(vocab_size, dim).normal_(mean=0.0, std=std, generator=g)


def table_stats(table: torch.Tensor, name: str) -> dict:
    t = table
    stats = {
        "name": name,
        "shape": list(t.shape),
        "param_count": t.numel(),
        "float32_mb": t.numel() * 4 / 1e6,
        "int8_mb": t.numel() / 1e6,
        "mean": float(t.mean()),
        "std": float(t.std()),
        "min": float(t.min()),
        "max": float(t.max()),
        "l2_norm_mean": float(torch.linalg.vector_norm(t, dim=1).mean()),
        "l2_norm_std": float(torch.linalg.vector_norm(t, dim=1).std()),
    }
    return stats


def print_stats(stats: dict) -> None:
    print(f"  [{stats['name']}]")
    print(f"    shape: {stats['shape']}  params: {stats['param_count']:,}")
    print(f"    memory: {stats['float32_mb']:.2f} MB (float32)  |  {stats['int8_mb']:.2f} MB (int8)")
    print(f"    value distribution: mean={stats['mean']:+.4f}  std={stats['std']:.4f}  "
          f"range=[{stats['min']:+.3f}, {stats['max']:+.3f}]")
    print(f"    per-row L2 norm: mean={stats['l2_norm_mean']:.3f}  std={stats['l2_norm_std']:.3f}")


def quantize_int8(table: torch.Tensor) -> tuple[np.ndarray, float]:
    """Symmetric per-tensor int8 quant. Returns (int8 array, scale)."""
    absmax = float(table.abs().max())
    scale = absmax / 127.0 if absmax > 0 else 1.0
    q = torch.round(table / scale).clamp(-127, 127).to(torch.int8).cpu().numpy()
    return q, scale


def dequantize_int8(q: np.ndarray, scale: float) -> torch.Tensor:
    return torch.tensor(q.astype(np.float32) * scale)


def demo_encode(tk: LexicalTokenizer, embedding: torch.nn.Embedding, text: str) -> None:
    print(f"\n  demo text: {text!r}")
    data = text.encode("utf-8")
    ids = tk.encode(data)
    id_tensor = torch.tensor(ids, dtype=torch.long)
    vectors = embedding(id_tensor)
    print(f"    input bytes: {len(data)}")
    print(f"    token IDs ({len(ids)}): {ids[:20]}{'...' if len(ids) > 20 else ''}")
    print(f"    embedding tensor shape: {list(vectors.shape)}")
    print(f"    first token (id={ids[0]}): vector[0:8] = {vectors[0, :8].tolist()}")
    print(f"    token breakdown:")
    for i, tid in enumerate(ids[:min(len(ids), 12)]):
        kind, payload = tk.id_to_token[tid]
        if kind == "LEARNED":
            sub, has_prefix = payload
            shown = ("▁" if has_prefix else "") + sub.decode("utf-8", "replace")
        elif kind == "BYTE":
            shown = f"byte 0x{payload:02x}"
        elif kind == "PUNCT":
            shown = f"punct {chr(payload)!r}"
        elif kind == "WS_RUN":
            shown = f"ws_run x{payload}"
        print(f"      ids[{i}]={tid:<6} kind={kind:<8} {shown}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=EMBED_DIM)
    parser.add_argument("--demo", type=str,
                        default="The cat sleeps peacefully on the warm mat.")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    print("=" * 70)
    print(f"WORD EMBEDDING V1 — SCAFFOLD  (dim={args.dim}, seed={args.seed})")
    print("=" * 70)

    print(f"\n[1] Loading champion tokenizer: {CHAMPION_VOCAB}")
    t0 = time.time()
    tk = load_tokenizer_from_json(CHAMPION_VOCAB)
    print(f"    vocab_size: {tk.vocab_size:,}  ({time.time()-t0:.1f}s)")

    print(f"\n[2] Allocating random-init embedding table")
    table = allocate_random_table(tk.vocab_size, args.dim, seed=args.seed)
    print_stats(table_stats(table, f"float32 random init"))

    print(f"\n[3] Wrapping as torch.nn.Embedding and running demo")
    embedding = torch.nn.Embedding.from_pretrained(table, freeze=True)
    demo_encode(tk, embedding, args.demo)

    print(f"\n[4] Int8 quantization (symmetric per-tensor)")
    q, scale = quantize_int8(table)
    print(f"    scale: {scale:.6e}")
    dequant = dequantize_int8(q, scale)
    err = (table - dequant).abs()
    print(f"    dequant error: mean={float(err.mean()):.6e}  max={float(err.max()):.6e}")

    print(f"\n[5] Saving artifacts to {OUT_DIR}")
    f32_path = OUT_DIR / f"embedding_table_f32_d{args.dim}.npy"
    np.save(f32_path, table.numpy())
    print(f"    {f32_path}  ({f32_path.stat().st_size:,} bytes)")

    i8_path = OUT_DIR / f"embedding_table_int8_d{args.dim}.npz"
    np.savez(i8_path, q=q, scale=np.float32(scale))
    print(f"    {i8_path}  ({i8_path.stat().st_size:,} bytes)")

    meta = {
        "dim": args.dim,
        "vocab_size": tk.vocab_size,
        "seed": args.seed,
        "init": "Xavier normal (std=1/sqrt(dim))",
        "param_count": tk.vocab_size * args.dim,
        "float32_bytes": tk.vocab_size * args.dim * 4,
        "int8_bytes": tk.vocab_size * args.dim,
        "int8_scale": scale,
        "trained": False,
        "note": "Random init scaffold. Upper model will train the table.",
        "source_vocab": str(CHAMPION_VOCAB).replace("\\", "/"),
    }
    meta_path = OUT_DIR / f"embedding_metadata_d{args.dim}.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"    {meta_path}")

    print("\n" + "=" * 70)
    print("SCAFFOLD VERDICT")
    print("=" * 70)
    print(f"  vocab:         {tk.vocab_size:,} tokens")
    print(f"  embedding dim: {args.dim}")
    print(f"  param count:   {tk.vocab_size * args.dim:,}")
    print(f"  memory f32:    {tk.vocab_size * args.dim * 4 / 1e6:.2f} MB")
    print(f"  memory int8:   {tk.vocab_size * args.dim / 1e6:.2f} MB")
    print(f"  end-to-end text -> [N, {args.dim}] tensor: OK")
    print(f"  training:      NOT YET (upper model dependency)")
    print("=" * 70)


if __name__ == "__main__":
    main()
