"""Nano brain v1 scaffold — 2-layer transformer, 64 dim, random init.

This is the forward-pass-only shape scaffold. Embedder + transformer + output
head in a single module. Tied embedder/output weights. No training yet.

Purpose: prove the end-to-end shape pipeline text -> IDs -> [N,64] ->
transformer -> [N, 32294] logits works. Random params, so logits are noise.

Architecture (nano):
  Embedder:    32,294 × 64  (tied with output head)
  Pos emb:     max_seq × 64  (learned)
  2 × TransformerBlock:
    LN -> MHA (4 heads, causal) -> residual
    LN -> FFN (64 -> 256 -> 64) -> residual
  Final LN
  Output: x @ embedder.weight.T  (tied, produces 32,294 logits)

Params (tied): ~2.2M total
  - embedder (tied):      2,066,816  (94%)
  - pos_emb (seq=256):       16,384
  - 2 × (MHA + FFN):        ~100,000
  - LayerNorms:              ~1,000

Usage:
  python tools/diag_nano_brain_v1.py
  python tools/diag_nano_brain_v1.py --demo "The cat sleeps peacefully."
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).parent))
from diag_subword_tokenizer_exact import LexicalTokenizer
from diag_word_tokenizer_champion_freeze import load_tokenizer_from_json

CHAMPION_VOCAB = Path("output/word_tokenizer_champion/champion_vocab.json")
OUT_DIR = Path("output/nano_brain_v1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

VOCAB_SIZE = 32294
DIM = 64
N_LAYERS = 2
N_HEADS = 4
FFN_MULT = 4
MAX_SEQ = 256
SEED = 42


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, hd)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self, dim: int, ffn_mult: int):
        super().__init__()
        h = dim * ffn_mult
        self.fc1 = nn.Linear(dim, h, bias=False)
        self.fc2 = nn.Linear(h, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, ffn_mult: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, ffn_mult)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class NanoBrain(nn.Module):
    def __init__(self, vocab: int = VOCAB_SIZE, dim: int = DIM,
                 n_layers: int = N_LAYERS, n_heads: int = N_HEADS,
                 ffn_mult: int = FFN_MULT, max_seq: int = MAX_SEQ):
        super().__init__()
        self.vocab = vocab
        self.dim = dim
        self.max_seq = max_seq
        self.tok_emb = nn.Embedding(vocab, dim)
        self.pos_emb = nn.Embedding(max_seq, dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads, ffn_mult) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(dim)
        # Output is tied: logits = x @ tok_emb.weight.T
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=1.0 / math.sqrt(m.in_features))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=1.0 / math.sqrt(m.embedding_dim))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        assert T <= self.max_seq, f"sequence length {T} exceeds max_seq {self.max_seq}"
        pos = torch.arange(T, device=ids.device)
        x = self.tok_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = x @ self.tok_emb.weight.T  # tied head
        return logits

    def param_breakdown(self) -> dict:
        d = {}
        d["tok_emb"] = self.tok_emb.weight.numel()
        d["pos_emb"] = self.pos_emb.weight.numel()
        d["blocks"] = sum(p.numel() for p in self.blocks.parameters())
        d["ln_f"] = sum(p.numel() for p in self.ln_f.parameters())
        d["total_untied"] = d["tok_emb"] + d["pos_emb"] + d["blocks"] + d["ln_f"] + d["tok_emb"]  # if separate head
        d["total_tied"] = d["tok_emb"] + d["pos_emb"] + d["blocks"] + d["ln_f"]
        return d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=DIM)
    parser.add_argument("--layers", type=int, default=N_LAYERS)
    parser.add_argument("--heads", type=int, default=N_HEADS)
    parser.add_argument("--max-seq", type=int, default=MAX_SEQ)
    parser.add_argument("--demo", type=str,
                        default="The cat sleeps peacefully on the warm mat.")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    print("=" * 70)
    print(f"NANO BRAIN V1 — SCAFFOLD  "
          f"(dim={args.dim} layers={args.layers} heads={args.heads} max_seq={args.max_seq})")
    print("=" * 70)
    torch.manual_seed(args.seed)

    print(f"\n[1] Loading champion tokenizer")
    tk = load_tokenizer_from_json(CHAMPION_VOCAB)
    print(f"    vocab_size: {tk.vocab_size:,}")

    print(f"\n[2] Building nano brain")
    model = NanoBrain(vocab=tk.vocab_size, dim=args.dim, n_layers=args.layers,
                      n_heads=args.heads, max_seq=args.max_seq)
    breakdown = model.param_breakdown()
    print(f"    param breakdown:")
    for k, v in breakdown.items():
        print(f"      {k:<15}: {v:>10,}")
    total = breakdown["total_tied"]
    print(f"    total (tied): {total:,}  ({total * 4 / 1e6:.2f} MB f32, {total / 1e6:.2f} MB int8)")
    ratio_emb = breakdown["tok_emb"] / total
    print(f"    embedder accounts for {100*ratio_emb:.1f}% of params (rest is attention/FFN/norms)")

    print(f"\n[3] Forward pass on demo")
    ids = tk.encode(args.demo.encode("utf-8"))
    id_tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(0)  # [1, N]
    print(f"    demo: {args.demo!r}")
    print(f"    tokens: {len(ids)}  ids[:10]={ids[:10]}")
    t0 = time.time()
    with torch.no_grad():
        logits = model(id_tensor)
    dt = time.time() - t0
    print(f"    logits shape: {list(logits.shape)}  ({dt*1000:.1f} ms)")
    # Top-1 prediction per position
    top1 = logits.argmax(dim=-1)[0].tolist()
    print(f"    top-1 next-token predictions (first 10):")
    for i, (inp_id, pred_id) in enumerate(zip(ids[:10], top1[:10])):
        inp_kind = tk.id_to_token[inp_id][0]
        pred_kind = tk.id_to_token[pred_id][0]
        pred_payload = tk.id_to_token[pred_id][1]
        if pred_kind == "LEARNED":
            pred_str = ("▁" if pred_payload[1] else "") + pred_payload[0].decode("utf-8", "replace")
        elif pred_kind == "BYTE":
            pred_str = f"byte 0x{pred_payload:02x}"
        else:
            pred_str = str(pred_payload)
        print(f"      pos {i:2d}: input id={inp_id:<6} ({inp_kind}) -> pred id={pred_id:<6} ({pred_kind} {pred_str})")

    print(f"\n    (Predictions are NOISE — model is random-init and untrained.)")

    print(f"\n[4] Batch shape test")
    batch_ids = id_tensor.expand(4, -1)
    with torch.no_grad():
        batch_logits = model(batch_ids)
    print(f"    batch [4, {len(ids)}, vocab] -> logits {list(batch_logits.shape)}")

    print(f"\n[5] Cross-entropy loss value (random init baseline)")
    # Predict position i+1 from position i
    shifted_ids = id_tensor[0, 1:]  # [N-1]
    shifted_logits = logits[0, :-1, :]  # [N-1, vocab]
    loss = F.cross_entropy(shifted_logits, shifted_ids)
    expected_random = math.log(tk.vocab_size)
    print(f"    current loss: {loss.item():.4f}")
    print(f"    random-uniform expected: {expected_random:.4f}  (log({tk.vocab_size}))")
    print(f"    perplexity: {math.exp(loss.item()):.1f}  (uniform expected: {tk.vocab_size})")

    print(f"\n[6] Saving blueprint metadata")
    meta = {
        "architecture": "nano-transformer (causal, tied embedder/head)",
        "vocab_size": tk.vocab_size,
        "dim": args.dim,
        "n_layers": args.layers,
        "n_heads": args.heads,
        "ffn_mult": FFN_MULT,
        "max_seq": args.max_seq,
        "param_breakdown": breakdown,
        "total_params_tied": breakdown["total_tied"],
        "memory_f32_mb": breakdown["total_tied"] * 4 / 1e6,
        "memory_int8_mb": breakdown["total_tied"] / 1e6,
        "trained": False,
        "init": "linear/emb: normal std=1/sqrt(fan_in); LN: ones/zeros (pytorch default)",
        "seed": args.seed,
        "champion_tokenizer_vocab": str(CHAMPION_VOCAB).replace("\\", "/"),
    }
    meta_path = OUT_DIR / f"nano_brain_metadata_d{args.dim}_L{args.layers}.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"    {meta_path}")

    print("\n" + "=" * 70)
    print("SCAFFOLD VERDICT")
    print("=" * 70)
    print(f"  total params (tied):       {total:,}")
    print(f"  memory:                    {total * 4 / 1e6:.2f} MB f32  /  {total / 1e6:.2f} MB int8")
    print(f"  text -> logits [B, N, V]:  OK")
    print(f"  init loss:                 {loss.item():.3f}  (expected random ~{expected_random:.3f})")
    print(f"  training:                  NOT YET (needs data pipeline + optimizer)")
    print("=" * 70)


if __name__ == "__main__":
    main()
