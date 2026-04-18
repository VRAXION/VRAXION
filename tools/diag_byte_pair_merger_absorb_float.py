"""L1 Merger — ABSORB FLOAT into existing categories.

User insight: always quantize a group/cell where at least one member is
currently float. Success => one float disappears. This pass tries to
absorb each still-float cell into an EXISTING codebook slot (lookup or
already-used int8 value), without growing any table.

Protocol:
  1. Load free_int8 result (with full state restore).
  2. For each float cell F (smallest-|w| first):
     Build candidate list:
       - All lookup codebook entries (16 for W1, 7 for W2)
       - All already-used int8 values in the same matrix
     Sort by |F.value - candidate.value|.
     Try each: commit, check lossless, accept first successful.
  3. Retrain every 30 accepts.
  4. Save.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).parent))
from diag_byte_pair_merger_free_int8 import (
    FreeInt8Merger, load_source_json, load_byte_pairs,
    metrics, DEVICE, LOSSLESS_FLOOR,
    RETRAIN_MAX_ITER, RETRAIN_PATIENCE,
)

RETRAIN_EVERY = 30


def restore_full_state(model, state):
    """Override __init__ defaults with saved values from JSON."""
    with torch.no_grad():
        model.alpha_free_W1.data.fill_(float(state["alpha_free_W1"]))
        model.alpha_free_W2.data.fill_(float(state["alpha_free_W2"]))
        model.W1_int8.copy_(torch.tensor(state["W1_int8"], dtype=torch.float32, device=DEVICE))
        model.W2_int8.copy_(torch.tensor(state["W2_int8"], dtype=torch.float32, device=DEVICE))
        model.W1_int8_mask.copy_(torch.tensor(state["W1_int8_mask"], dtype=torch.bool, device=DEVICE))
        model.W2_int8_mask.copy_(torch.tensor(state["W2_int8_mask"], dtype=torch.bool, device=DEVICE))


def pick_float_cell(model, skipped):
    with torch.no_grad():
        W1_abs = model.W1_float.abs().clone()
        W2_abs = model.W2_float.abs().clone()
        W1_abs[model.W1_frozen_mask | model.W1_int8_mask] = float("inf")
        W2_abs[model.W2_frozen_mask | model.W2_int8_mask] = float("inf")
        for (mat, i, j) in skipped:
            if mat == "W1":
                W1_abs[i, j] = float("inf")
            else:
                W2_abs[i, j] = float("inf")
        all_abs = torch.cat([W1_abs.flatten(), W2_abs.flatten()])
        free_count = (~torch.isinf(all_abs)).sum().item()
        if free_count == 0:
            return None
        val, flat_idx = torch.min(all_abs, dim=0)
    idx = flat_idx.item()
    n1 = model.W1_float.numel()
    if idx < n1:
        i = int(idx // model.W1_float.shape[1])
        j = int(idx % model.W1_float.shape[1])
        return ("W1", i, j)
    idx2 = idx - n1
    i = int(idx2 // model.W2_float.shape[1])
    j = int(idx2 % model.W2_float.shape[1])
    return ("W2", i, j)


def build_candidates(model, matrix, target_w):
    """Return list of (kind, key, effective_value, distance).
    kind in {'lookup', 'int8'}.
    For 'lookup', key is codebook index. For 'int8', key is int value.
    """
    candidates = []
    if matrix == "W1":
        cb = model.codebook_W1.data.cpu().numpy()
        alpha_f = model.alpha_free_W1.item()
        int8_used = set(model.W1_int8[model.W1_int8_mask].cpu().numpy().tolist())
    else:
        cb = model.codebook_W2.data.cpu().numpy()
        alpha_f = model.alpha_free_W2.item()
        int8_used = set(model.W2_int8[model.W2_int8_mask].cpu().numpy().tolist())

    # Lookup candidates
    for k, v in enumerate(cb):
        candidates.append(("lookup", k, float(v), abs(float(v) - target_w)))
    # Int8 candidates (only already-used values)
    for iv in int8_used:
        eff = iv * alpha_f
        candidates.append(("int8", int(iv), float(eff), abs(eff - target_w)))

    candidates.sort(key=lambda c: c[3])
    return candidates


def try_absorb(model, cell, cand, data):
    """Apply candidate, return (lossless, saved_state)."""
    mat, i, j = cell
    kind, key, eff, _ = cand
    if mat == "W1":
        orig_float = float(model.W1_float.data[i, j].item())
        orig_frozen_mask = bool(model.W1_frozen_mask[i, j].item())
        orig_int8_mask = bool(model.W1_int8_mask[i, j].item())
        orig_idx = int(model.W1_idx[i, j].item())
        orig_int8 = float(model.W1_int8[i, j].item())
        if kind == "lookup":
            model.W1_frozen_mask[i, j] = True
            model.W1_idx[i, j] = key
        else:  # int8
            model.W1_int8_mask[i, j] = True
            model.W1_int8[i, j] = float(key)
        model.W1_float.data[i, j] = 0.0
    else:
        orig_float = float(model.W2_float.data[i, j].item())
        orig_frozen_mask = bool(model.W2_frozen_mask[i, j].item())
        orig_int8_mask = bool(model.W2_int8_mask[i, j].item())
        orig_idx = int(model.W2_idx[i, j].item())
        orig_int8 = float(model.W2_int8[i, j].item())
        if kind == "lookup":
            model.W2_frozen_mask[i, j] = True
            model.W2_idx[i, j] = key
        else:
            model.W2_int8_mask[i, j] = True
            model.W2_int8[i, j] = float(key)
        model.W2_float.data[i, j] = 0.0

    ll, _ = metrics(model, data)
    saved = (mat, i, j, orig_float, orig_frozen_mask, orig_int8_mask, orig_idx, orig_int8)
    return ll, saved


def rollback(model, saved):
    mat, i, j, orig_float, orig_frozen, orig_int8m, orig_idx, orig_int8v = saved
    if mat == "W1":
        model.W1_float.data[i, j] = orig_float
        model.W1_frozen_mask[i, j] = orig_frozen
        model.W1_int8_mask[i, j] = orig_int8m
        model.W1_idx[i, j] = orig_idx
        model.W1_int8[i, j] = orig_int8v
    else:
        model.W2_float.data[i, j] = orig_float
        model.W2_frozen_mask[i, j] = orig_frozen
        model.W2_int8_mask[i, j] = orig_int8m
        model.W2_idx[i, j] = orig_idx
        model.W2_int8[i, j] = orig_int8v


def retrain(model, data, max_iter=RETRAIN_MAX_ITER, patience=RETRAIN_PATIENCE):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.LBFGS(params, lr=1.0, max_iter=20, history_size=50,
                            line_search_fn="strong_wolfe")
    best, no_imp = float("inf"), 0
    for _ in range(max_iter):
        def closure():
            opt.zero_grad()
            x_hat, _ = model(data)
            loss = F.mse_loss(x_hat, data)
            loss.backward()
            with torch.no_grad():
                if model.W1_float.grad is not None:
                    mask = model.W1_frozen_mask | model.W1_int8_mask
                    model.W1_float.grad[mask] = 0.0
                if model.W2_float.grad is not None:
                    mask = model.W2_frozen_mask | model.W2_int8_mask
                    model.W2_float.grad[mask] = 0.0
            return loss
        lv = opt.step(closure)
        lf = lv.item() if torch.is_tensor(lv) else lv
        if lf < best - 1e-9:
            best, no_imp = lf, 0
        else:
            no_imp += 1
        if no_imp >= patience:
            break
    return best


def save_checkpoint(path, model, skipped, n_processed, n_accepted, n_skipped):
    torch.save({
        "codebook_W1": model.codebook_W1.data.cpu(),
        "codebook_W2": model.codebook_W2.data.cpu(),
        "W1_idx": model.W1_idx.cpu(),
        "W2_idx": model.W2_idx.cpu(),
        "W1_frozen_mask": model.W1_frozen_mask.cpu(),
        "W2_frozen_mask": model.W2_frozen_mask.cpu(),
        "W1_int8_mask": model.W1_int8_mask.cpu(),
        "W2_int8_mask": model.W2_int8_mask.cpu(),
        "W1_int8": model.W1_int8.cpu(),
        "W2_int8": model.W2_int8.cpu(),
        "alpha_free_W1": model.alpha_free_W1.data.cpu(),
        "alpha_free_W2": model.alpha_free_W2.data.cpu(),
        "W1_float": model.W1_float.data.cpu(),
        "W2_float": model.W2_float.data.cpu(),
        "b1": model.b1.data.cpu(),
        "b2": model.b2.data.cpu(),
        "db1": model.db1.data.cpu(),
        "db2": model.db2.data.cpu(),
        "c19_c": model.c19.c_raw.data.cpu(),
        "c19_rho": model.c19.rho_raw.data.cpu(),
        "skipped": [list(c) for c in skipped],
        "n_processed": n_processed,
        "n_accepted": n_accepted,
        "n_skipped": n_skipped,
    }, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    telem_path = out_dir / "telemetry.jsonl"

    print("=== ABSORB FLOAT INTO EXISTING CATEGORIES ===", flush=True)
    state = load_source_json(args.source)
    model = FreeInt8Merger(state, DEVICE).to(DEVICE)
    restore_full_state(model, state)

    data = load_byte_pairs().to(DEVICE)
    ll0, pd0 = metrics(model, data)
    free_w1, free_w2 = model.still_free_count()
    print(f"Baseline: lossless={ll0:.4f}%, still_free={free_w1+free_w2}", flush=True)
    print(f"Codebook W1: {len(model.codebook_W1)} entries", flush=True)
    print(f"Codebook W2: {len(model.codebook_W2)} entries", flush=True)
    used_int8_W1 = len(set(model.W1_int8[model.W1_int8_mask].cpu().numpy().tolist()))
    used_int8_W2 = len(set(model.W2_int8[model.W2_int8_mask].cpu().numpy().tolist()))
    print(f"Used int8 W1: {used_int8_W1} distinct values", flush=True)
    print(f"Used int8 W2: {used_int8_W2} distinct values", flush=True)

    skipped = set()
    n_processed = 0
    n_accepted = 0
    n_skipped = 0
    n_lookup = 0
    n_int8 = 0
    t0 = time.time()

    while True:
        cell = pick_float_cell(model, skipped)
        if cell is None:
            print("\nNo more float cells.", flush=True)
            break
        n_processed += 1

        mat, i, j = cell
        if mat == "W1":
            w = model.W1_float.data[i, j].item()
        else:
            w = model.W2_float.data[i, j].item()

        candidates = build_candidates(model, mat, w)
        # Only try the closest N candidates (saves time on far ones)
        candidates = candidates[:8]

        accepted = False
        best_ll = -1.0
        used_kind = None
        used_key = None
        n_tries = 0
        for cand in candidates:
            n_tries += 1
            ll, saved = try_absorb(model, cell, cand, data)
            if ll > best_ll:
                best_ll = ll
            if ll >= LOSSLESS_FLOOR:
                accepted = True
                used_kind = cand[0]
                used_key = cand[1]
                break
            else:
                rollback(model, saved)

        if accepted:
            n_accepted += 1
            if used_kind == "lookup":
                n_lookup += 1
            else:
                n_int8 += 1
        else:
            skipped.add(cell)
            n_skipped += 1

        free_w1, free_w2 = model.still_free_count()

        row = {
            "cell": list(cell), "w": w,
            "accepted": accepted, "used_kind": used_kind, "used_key": used_key,
            "n_tries": n_tries, "best_ll": best_ll,
            "still_free": free_w1 + free_w2,
        }
        with open(telem_path, "a") as f:
            f.write(json.dumps(row) + "\n")

        if n_processed % 50 == 0 or not accepted:
            ll, _ = metrics(model, data)
            status = f"ACCEPT({used_kind}={used_key})" if accepted else f"SKIP ll={best_ll:.2f}%"
            print(
                f"  #{n_processed} [{status}]: tries={n_tries}, "
                f"free={free_w1+free_w2}, acc={n_accepted} (look={n_lookup}, "
                f"i8={n_int8}), skip={n_skipped}, ll={ll:.4f}%",
                flush=True,
            )

        if n_accepted > 0 and n_accepted % RETRAIN_EVERY == 0 and accepted:
            print(f"  [retrain @ {n_accepted} accepts]", flush=True)
            retrain(model, data)
            save_checkpoint(out_dir / "checkpoint.pt", model, skipped,
                            n_processed, n_accepted, n_skipped)

        if n_processed % 100 == 0:
            save_checkpoint(out_dir / "checkpoint.pt", model, skipped,
                            n_processed, n_accepted, n_skipped)

    # Final retrain
    print("\n--- FINAL RETRAIN ---", flush=True)
    retrain(model, data, max_iter=100, patience=30)
    ll, pd = metrics(model, data)

    t_total = time.time() - t0
    free_w1, free_w2 = model.still_free_count()
    n_int8_W1 = model.W1_int8_mask.sum().item()
    n_int8_W2 = model.W2_int8_mask.sum().item()
    lookup_frozen = (model.W1_frozen_mask.sum().item() +
                     model.W2_frozen_mask.sum().item())

    print(f"\n=== DONE ===")
    print(f"  Processed: {n_processed}")
    print(f"  Accepted: {n_accepted} (lookup: {n_lookup}, int8: {n_int8})")
    print(f"  Skipped: {n_skipped}")
    print(f"  Still float: {free_w1+free_w2}")
    print(f"  Lossless: {ll:.4f}%, per_dim: {pd:.4f}%")
    print(f"  Time: {t_total:.0f}s")

    # Deploy size estimate
    cb1_sz = len(model.codebook_W1)
    cb2_sz = len(model.codebook_W2)
    bits_W1 = max(1, int(np.ceil(np.log2(max(cb1_sz, 2)))))
    bits_W2 = max(1, int(np.ceil(np.log2(max(cb2_sz, 2)))))
    idx_bytes_W1 = int(np.ceil(model.W1_idx.numel() * bits_W1 / 8))
    idx_bytes_W2 = int(np.ceil(model.W2_idx.numel() * bits_W2 / 8))
    cb_bytes = (cb1_sz + cb2_sz) * 4
    int8_bytes = n_int8_W1 + n_int8_W2
    free_float_bytes = (free_w1 + free_w2) * 4
    mask_bytes = (model.W1_frozen_mask.numel() + model.W2_frozen_mask.numel()) // 8
    int8_mask_bytes = (model.W1_int8_mask.numel() + model.W2_int8_mask.numel()) // 8
    misc_bytes = (model.b1.numel() + model.b2.numel() + model.db1.numel() +
                  model.db2.numel() + 2 * model.H + 2) * 4
    total_bytes = (idx_bytes_W1 + idx_bytes_W2 + cb_bytes +
                   int8_bytes + free_float_bytes +
                   mask_bytes + int8_mask_bytes + misc_bytes)
    print(f"\n  Deploy: {total_bytes} B ({total_bytes/1024:.2f} KB)")
    print(f"    (lookup {bits_W1}+{bits_W2}bit idx: {idx_bytes_W1+idx_bytes_W2}, "
          f"cb: {cb_bytes}, int8: {int8_bytes}, "
          f"float: {free_float_bytes}, mask: {mask_bytes+int8_mask_bytes}, "
          f"bias+C19: {misc_bytes})")

    artifact = {
        "architecture": "absorb-float into existing categories",
        "source": args.source,
        "H": model.H, "in_dim": model.in_dim, "out_dim": model.out_dim,
        "total_cells": model.total_cells(),
        "lookup_frozen": lookup_frozen,
        "int8_frozen": n_int8_W1 + n_int8_W2,
        "still_free": free_w1 + free_w2,
        "codebook_W1": model.codebook_W1.data.cpu().numpy().tolist(),
        "codebook_W2": model.codebook_W2.data.cpu().numpy().tolist(),
        "alpha_free_W1": float(model.alpha_free_W1.item()),
        "alpha_free_W2": float(model.alpha_free_W2.item()),
        "lossless": ll, "per_dim": pd,
        "n_processed": n_processed, "n_accepted": n_accepted,
        "n_skipped": n_skipped,
        "n_lookup": n_lookup, "n_int8": n_int8,
        "time_s": t_total,
        "deploy_bytes_est": total_bytes,
        "W1_idx": model.W1_idx.cpu().numpy().astype(int).tolist(),
        "W2_idx": model.W2_idx.cpu().numpy().astype(int).tolist(),
        "W1_frozen_mask": model.W1_frozen_mask.cpu().numpy().astype(int).tolist(),
        "W2_frozen_mask": model.W2_frozen_mask.cpu().numpy().astype(int).tolist(),
        "W1_int8_mask": model.W1_int8_mask.cpu().numpy().astype(int).tolist(),
        "W2_int8_mask": model.W2_int8_mask.cpu().numpy().astype(int).tolist(),
        "W1_int8": model.W1_int8.cpu().numpy().astype(int).tolist(),
        "W2_int8": model.W2_int8.cpu().numpy().astype(int).tolist(),
        "W1_float": model.W1_float.data.cpu().numpy().tolist(),
        "W2_float": model.W2_float.data.cpu().numpy().tolist(),
        "b1": model.b1.data.cpu().numpy().tolist(),
        "b2": model.b2.data.cpu().numpy().tolist(),
        "db1": model.db1.data.cpu().numpy().tolist(),
        "db2": model.db2.data.cpu().numpy().tolist(),
        "c19_c": model.c19.c_raw.data.cpu().numpy().tolist(),
        "c19_rho": model.c19.rho_raw.data.cpu().numpy().tolist(),
    }
    with open(out_dir / "final_model.json", "w") as f:
        json.dump(artifact, f)
    print(f"  Saved: {out_dir / 'final_model.json'}")


if __name__ == "__main__":
    main()
