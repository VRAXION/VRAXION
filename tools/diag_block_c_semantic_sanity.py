"""Block C semantic sanity check.

Given a saved embedding_table_E*.npy, probe the geometry by listing the
top-K nearest neighbours for a curated set of anchor tokens. A healthy
embedding places morphologically / semantically related tokens close
together; an unhealthy one gives random-looking neighbours.

We also quantise the table to int4 (per-channel symmetric) and re-run the
same probe: robust tables keep the neighbourhoods stable.

Run:
    python3 tools/diag_block_c_semantic_sanity.py \
        --embedding output/block_c_next_token_embedder/embedding_table_E16.npy \
        --vocab output/word_tokenizer_champion/champion_vocab.json \
        --top-k 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# A curated set of anchor tokens to probe. Each entry is the human-readable
# label we'll look up in the vocab JSON.
DEFAULT_ANCHORS = [
    ("LEARNED", " the"),
    ("LEARNED", " a"),
    ("LEARNED", " is"),
    ("LEARNED", " and"),
    ("LEARNED", " cat"),
    ("LEARNED", " dog"),
    ("LEARNED", " run"),
    ("LEARNED", " water"),
    ("LEARNED", " man"),
    ("LEARNED", " woman"),
    ("LEARNED", " one"),
    ("LEARNED", " two"),
    ("LEARNED", " good"),
    ("LEARNED", " bad"),
    ("LEARNED", "ing"),
    ("LEARNED", "ed"),
    ("LEARNED", "ly"),
    ("PUNCT", "."),
    ("PUNCT", ","),
    ("BYTE", 0x20),   # space as a standalone byte
    ("BYTE", 0x0A),   # newline
]


def load_vocab(vocab_path: Path):
    data = json.loads(vocab_path.read_text())
    # Build id → label and label → id mappings
    id2label = {}
    label2id = {}
    for entry in data:
        tid = entry["id"]
        kind = entry["kind"]
        if kind == "BYTE":
            lbl = ("BYTE", int(entry["byte"]))
            pretty = f"BYTE(0x{entry['byte']:02x})"
        elif kind == "PUNCT":
            lbl = ("PUNCT", entry["char"])
            pretty = f"PUNCT({entry['char']!r})"
        elif kind == "WS_RUN":
            lbl = ("WS_RUN", int(entry.get("count", 0)))
            pretty = f"WS_RUN(n={entry.get('count')})"
        elif kind == "LEARNED":
            # Reconstruct the textual form (space prefix preserved)
            text = entry.get("text", "")
            if entry.get("space_prefix"):
                text = " " + text
            lbl = ("LEARNED", text)
            pretty = f"{text!r}"
        else:
            lbl = (kind, tid)
            pretty = f"{kind}({tid})"
        id2label[tid] = pretty
        label2id[lbl] = tid
    return id2label, label2id


def ptq_per_channel(W, bits=4):
    qmax = float(2 ** (bits - 1) - 1)
    amax = np.max(np.abs(W), axis=0)
    safe = np.where(amax == 0, np.float32(1.0), amax)
    scale = safe / qmax
    q = np.round(W / scale).clip(-qmax, qmax)
    return (q * scale).astype(np.float32)


def top_k_neighbors(emb, anchor_id, k, exclude_self=True):
    v = emb[anchor_id]
    diff = emb - v
    d2 = np.sum(diff * diff, axis=1)
    if exclude_self:
        d2[anchor_id] = np.inf
    idx = np.argpartition(d2, k)[:k]
    order = idx[np.argsort(d2[idx])]
    return order, np.sqrt(d2[order])


def probe(emb, label, anchors, id2label, label2id, k):
    print(f"\n== {label}  (shape {emb.shape}) ==")
    missing = 0
    total_anchors = 0
    neighbor_cross = {}  # anchor_id -> list of neighbor_ids (for cross comparison)
    for kind, key in anchors:
        aid = label2id.get((kind, key))
        if aid is None:
            print(f"  [skip] anchor not in vocab: {kind} {key!r}")
            missing += 1
            continue
        nbrs, dists = top_k_neighbors(emb, aid, k)
        neighbor_cross[aid] = list(nbrs)
        total_anchors += 1
        a_pretty = id2label[aid]
        print(f"  {a_pretty:>30}  (id={aid})")
        for i, (nid, d) in enumerate(zip(nbrs, dists)):
            print(f"      #{i+1}  d={d:.4f}  {id2label[nid]}  (id={nid})")
    print(f"  ({total_anchors} anchors probed, {missing} missing)")
    return neighbor_cross


def stability_report(nbr_float, nbr_int4, id2label):
    print("\n== float ↔ int4 neighbour stability ==")
    per_anchor = []
    for aid, float_nbrs in nbr_float.items():
        int4_nbrs = nbr_int4.get(aid, [])
        if not int4_nbrs:
            continue
        overlap = len(set(float_nbrs) & set(int4_nbrs))
        k = len(float_nbrs)
        per_anchor.append(overlap / k)
        if overlap < k:
            print(f"  {id2label[aid]:>30}  overlap {overlap}/{k}  "
                  f"int4-only: "
                  f"{[id2label[n] for n in int4_nbrs if n not in float_nbrs]}")
    if per_anchor:
        mean = float(np.mean(per_anchor))
        print(f"  mean top-K overlap: {mean*100:.1f}%  ({len(per_anchor)} anchors)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedding", type=Path, required=True,
                    help="Path to embedding_table_E*.npy")
    ap.add_argument("--vocab", type=Path, required=True,
                    help="Path to champion_vocab.json")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    emb_float = np.load(args.embedding).astype(np.float32)
    id2label, label2id = load_vocab(args.vocab)
    print(f"Loaded embedding: {emb_float.shape} from {args.embedding}")
    print(f"Vocab has {len(id2label):,} entries")

    # int4 variant
    emb_int4 = ptq_per_channel(emb_float, bits=4)
    collisions = emb_float.shape[0] - len(np.unique(np.round(emb_int4, 6), axis=0))
    print(f"int4 uniqueness: {emb_float.shape[0] - collisions}/{emb_float.shape[0]}  "
          f"({collisions} collisions)")

    nbr_f = probe(emb_float, "FLOAT embedding", DEFAULT_ANCHORS,
                  id2label, label2id, args.top_k)
    nbr_q = probe(emb_int4, "INT4 embedding", DEFAULT_ANCHORS,
                  id2label, label2id, args.top_k)
    stability_report(nbr_f, nbr_q, id2label)


if __name__ == "__main__":
    main()
