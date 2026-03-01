"""CPU toy benchmark: local pointer read/write vs local/global dot-product variants.

Purpose:
  Compare memory read/write addressing strategies on non-trivial synthetic tasks,
  without touching the main training pipeline.

Variants:
  1) local_vshape      : current-style pointer local weighted window (read+write)
  2) local_dot         : pointer-local read with content reweight, local write
  3) global_dot_soft   : full-ring content softmax read+write (pointer effectively unused)
  4) global_dot_top2   : full-ring content top-2 read+write
  5) global_dot_top1   : full-ring content top-1 read+write

Tasks:
  - delayed_xor
  - chained_xor
  - associative_recall

Usage:
  python tests/bench_read_write_modes_cpu.py
  python tests/bench_read_write_modes_cpu.py --steps 800 --device cpu
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent.parent
for subdir in ("model", "training"):
    p = str(ROOT / subdir)
    if p not in sys.path:
        sys.path.insert(0, p)

from instnct import (  # type: ignore[import-not-found]
    INSTNCT,
    _c19_activation,
    _rho_from_raw,
    _C_from_raw,
    func_softread_tns,
    func_softwrit_tns,
    func_movepntr_tns,
)
from train import func_maskloss_mse, func_accuracy_bin  # type: ignore[import-not-found]


SEED = 42
EVAL_SEED = 9999
BATCH = 4
STEPS = 800
EVAL_EVERY = 100
LR = 1e-3


def _bytes_to_bits(data_np, mask_np, batch_size, seq_len, device):
    data_all = np.zeros((batch_size, seq_len + 1), dtype=np.uint8)
    mask_all = np.zeros((batch_size, seq_len + 1), dtype=np.uint8)
    for i in range(batch_size):
        data_all[i, : len(data_np[i])] = data_np[i][: seq_len + 1]
        mask_all[i, : len(mask_np[i])] = mask_np[i][: seq_len + 1]
    flat = np.unpackbits(data_all.reshape(-1))
    bits = flat.reshape(batch_size, seq_len + 1, 8).astype(np.float32)
    x = torch.from_numpy(bits[:, :seq_len].copy()).to(device)
    y = torch.from_numpy(bits[:, 1 : seq_len + 1].copy()).to(device)
    sup = mask_all[:, 1 : seq_len + 1].astype(np.float32)
    mask = torch.from_numpy(sup).unsqueeze(-1).to(device)
    return x, y, mask


def make_delayed_xor_batch(batch_size, seq_len, device, rng, delay=8):
    unit = delay + 3
    n_units = (seq_len + 1) // unit + 2
    data, masks = [], []
    for _ in range(batch_size):
        d, m = [], []
        for _ in range(n_units):
            a, b = rng.randint(0, 256), rng.randint(0, 256)
            d.append(a)
            m.append(0)
            d.append(b)
            m.append(0)
            for _ in range(delay):
                d.append(rng.randint(0, 256))
                m.append(0)
            d.append(a ^ b)
            m.append(1)
        data.append(np.array(d[: seq_len + 1], dtype=np.uint8))
        masks.append(np.array(m[: seq_len + 1], dtype=np.uint8))
    return _bytes_to_bits(data, masks, batch_size, seq_len, device)


def make_chained_xor_batch(batch_size, seq_len, device, rng):
    data, masks = [], []
    for _ in range(batch_size):
        d, m = [], []
        while len(d) < seq_len + 1:
            a, b = rng.randint(0, 256), rng.randint(0, 256)
            acc = a ^ b
            d.extend([a, b, acc])
            m.extend([0, 0, 1])
            for _ in range(2):
                c = rng.randint(0, 256)
                acc ^= c
                d.extend([c, acc])
                m.extend([0, 1])
            d.append(rng.randint(0, 256))
            m.append(0)
        data.append(np.array(d[: seq_len + 1], dtype=np.uint8))
        masks.append(np.array(m[: seq_len + 1], dtype=np.uint8))
    return _bytes_to_bits(data, masks, batch_size, seq_len, device)


def make_associative_recall_batch(batch_size, seq_len, device, rng, n_pairs=4):
    """[k1,v1,...,kn,vn,q,ans] with mask only on ans positions."""
    data, masks = [], []
    for _ in range(batch_size):
        d, m = [], []
        while len(d) < seq_len + 1:
            keys = rng.choice(256, size=n_pairs, replace=False).astype(np.uint8)
            vals = rng.randint(0, 256, size=n_pairs, dtype=np.uint8)
            q_idx = rng.randint(0, n_pairs)
            q = int(keys[q_idx])
            ans = int(vals[q_idx])
            for i in range(n_pairs):
                d.extend([int(keys[i]), int(vals[i])])
                m.extend([0, 0])
            d.append(q)
            m.append(0)
            d.append(ans)
            m.append(1)
            d.append(rng.randint(0, 256))
            m.append(0)
        data.append(np.array(d[: seq_len + 1], dtype=np.uint8))
        masks.append(np.array(m[: seq_len + 1], dtype=np.uint8))
    return _bytes_to_bits(data, masks, batch_size, seq_len, device)


TASKS = {
    "delayed_xor": {"fn": make_delayed_xor_batch, "seq": 44, "label": "Delayed XOR"},
    "chained_xor": {"fn": make_chained_xor_batch, "seq": 48, "label": "Chained XOR"},
    "assoc_recall": {"fn": make_associative_recall_batch, "seq": 54, "label": "Associative Recall"},
}


class RingModeModel(INSTNCT):
    """INSTNCT variant with pluggable read/write addressing modes for CPU toy benches."""

    def __init__(
        self,
        *,
        read_mode="local_vshape",
        write_mode="local_vshape",
        topk=2,
        mix_lambda=0.35,
        dot_temp=6.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.read_mode = read_mode
        self.write_mode = write_mode
        self.topk = topk
        self.mix_lambda = mix_lambda
        self.dot_temp = dot_temp
        self.read_q = nn.ModuleList([nn.Linear(self.hidden_dim, self.slot_dim) for _ in range(self.N)])
        self.write_q = nn.ModuleList([nn.Linear(self.hidden_dim, self.slot_dim) for _ in range(self.N)])

    @staticmethod
    def _all_indices(batch_size, slots, device):
        return torch.arange(slots, device=device).unsqueeze(0).expand(batch_size, -1)

    @staticmethod
    def _norm(x):
        return F.normalize(x, dim=-1, eps=1e-6)

    def _global_logits(self, q_tns, ring_tns):
        # q: (B,D), ring: (B,M,D) -> (B,M)
        qn = self._norm(q_tns)
        kn = self._norm(ring_tns)
        return (kn * qn.unsqueeze(1)).sum(-1) * self.dot_temp

    def _topk_sparse(self, logits, k):
        vals, idx = logits.topk(k, dim=1)
        w = torch.softmax(vals, dim=1)
        return idx, w

    def _process_chunk(self, x_chunk, ring_tns, ptr_tns, hidden_tns, S_flt, probs_lst, offsets_long, expert_weights):
        M, slot_dim, N = self.M, self.slot_dim, self.N
        ptr_tns = ptr_tns.clone()
        hidden_lst = [hidden_tns[i] for i in range(N)]
        C = x_chunk.shape[1]
        outs_lst = []

        for t in range(C):
            input_vec_tns = self.inp(x_chunk[:, t])

            for i in range(N):
                B = input_vec_tns.shape[0]
                center = ptr_tns[i].long().clamp(0, M - 1)
                local_idx = (center.unsqueeze(1) + offsets_long) % M
                local_w = expert_weights[i].unsqueeze(0).expand(B, -1)

                # --- READ addressing ---
                if self.read_mode == "local_vshape":
                    read_idx = local_idx
                    read_w = local_w
                elif self.read_mode == "local_dot":
                    exp_local = local_idx.unsqueeze(-1).expand(-1, -1, slot_dim)
                    neigh = ring_tns.gather(1, exp_local)
                    q = self.read_q[i](hidden_lst[i])  # (B,D)
                    content = (self._norm(neigh) * self._norm(q).unsqueeze(1)).sum(-1) * self.dot_temp
                    loc_logits = torch.log(local_w + 1e-8)
                    logits = (1.0 - self.mix_lambda) * loc_logits + self.mix_lambda * content
                    read_idx = local_idx
                    read_w = torch.softmax(logits, dim=1)
                else:
                    q = self.read_q[i](hidden_lst[i])
                    logits = self._global_logits(q, ring_tns)
                    if self.read_mode == "global_dot_soft":
                        read_idx = self._all_indices(B, M, x_chunk.device)
                        read_w = torch.softmax(logits, dim=1)
                    elif self.read_mode == "global_dot_top2":
                        read_idx, read_w = self._topk_sparse(logits, 2)
                    elif self.read_mode == "global_dot_top1":
                        read_idx, read_w = self._topk_sparse(logits, 1)
                    else:
                        raise ValueError(f"Unknown read_mode: {self.read_mode}")

                read_vec_tns, _ = func_softread_tns(ring_tns, read_idx, read_w, slot_dim)

                # --- Hidden update ---
                theta_tns = (ptr_tns[i] / M) * (2 * math.pi)
                phase_tns = (
                    torch.cos(theta_tns).unsqueeze(-1) * self.phase_cos
                    + torch.sin(theta_tns).unsqueeze(-1) * self.phase_sin
                )
                hidden_lst[i] = _c19_activation(
                    input_vec_tns
                    + S_flt * self.read_proj[i](read_vec_tns)
                    + phase_tns
                    + hidden_lst[i],
                    rho=_rho_from_raw(self.c19_rho_hidden),
                    C=_C_from_raw(self.c19_C_hidden),
                )

                # --- WRITE addressing ---
                if self.write_proj is not None:
                    write_vec = self.write_proj[i](hidden_lst[i])
                else:
                    write_vec = hidden_lst[i]
                if self._expert_conf is not None:
                    write_vec = self._expert_conf[i].item() * write_vec

                if self.write_mode == "local_vshape":
                    write_idx = local_idx
                    write_w = local_w
                else:
                    wq = self.write_q[i](hidden_lst[i])
                    w_logits = self._global_logits(wq, ring_tns)
                    B = w_logits.shape[0]
                    if self.write_mode == "global_dot_soft":
                        write_idx = self._all_indices(B, M, x_chunk.device)
                        write_w = torch.softmax(w_logits, dim=1)
                    elif self.write_mode == "global_dot_top2":
                        write_idx, write_w = self._topk_sparse(w_logits, 2)
                    elif self.write_mode == "global_dot_top1":
                        write_idx, write_w = self._topk_sparse(w_logits, 1)
                    else:
                        raise ValueError(f"Unknown write_mode: {self.write_mode}")

                write_expanded = write_idx.unsqueeze(-1).expand(-1, -1, slot_dim)
                ring_tns = func_softwrit_tns(ring_tns, write_vec, write_expanded, write_w)

                ptr_tns[i] = func_movepntr_tns(ptr_tns[i], self.dests[i], probs_lst[i % len(probs_lst)], M)

            mean_hidden = torch.stack(hidden_lst).mean(0)
            outs_lst.append(self.out(mean_hidden))

        hidden_tns = torch.stack(hidden_lst)
        outs_tns = torch.stack(outs_lst, dim=1)
        return ring_tns, ptr_tns, hidden_tns, outs_tns


VARIANTS = {
    "local_vshape": {"read_mode": "local_vshape", "write_mode": "local_vshape"},
    "local_dot": {"read_mode": "local_dot", "write_mode": "local_vshape"},
    "global_dot_soft": {"read_mode": "global_dot_soft", "write_mode": "global_dot_soft"},
    "global_dot_top2": {"read_mode": "global_dot_top2", "write_mode": "global_dot_top2"},
    "global_dot_top1": {"read_mode": "global_dot_top1", "write_mode": "global_dot_top1"},
}


def run_one(task_info, variant_name, variant_cfg, steps=STEPS, batch=BATCH, device="cpu"):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng_train = np.random.RandomState(SEED)
    rng_eval = np.random.RandomState(EVAL_SEED)

    model = RingModeModel(
        M=64,
        embed_dim=64,
        N=2,
        R=1,
        embed_mode=False,
        read_mode=variant_cfg["read_mode"],
        write_mode=variant_cfg["write_mode"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    ex, ey, emask = task_info["fn"](8, task_info["seq"], device, rng_eval)
    best_eval = 0.0
    t0 = time.perf_counter()

    for step in range(1, steps + 1):
        x, y, mask = task_info["fn"](batch, task_info["seq"], device, rng_train)
        pred, _ = model(x)
        _, loss = func_maskloss_mse(pred, y, mask)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()

        if step % EVAL_EVERY == 0 or step == steps:
            with torch.no_grad():
                epred, _ = model(ex)
                _, acc = func_accuracy_bin(epred, ey, emask)
            best_eval = max(best_eval, float(acc))

    wall = time.perf_counter() - t0
    return best_eval, wall, n_params


def main():
    ap = argparse.ArgumentParser(description="CPU benchmark: local/global dot-product read/write variants.")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    print("=" * 88)
    print("  BENCH: POINTER-LOCAL VS GLOBAL DOT-PRODUCT ADDRESSING (CPU TOY, NON-TRIVIAL TASKS)")
    print("=" * 88)
    print(f"  Steps: {args.steps} | Batch: {args.batch} | Device: {args.device} | Seed: {SEED}")
    print()

    all_results = []
    for task_name, task_info in TASKS.items():
        print(f"[TASK] {task_info['label']} (seq={task_info['seq']})")
        for v_name, v_cfg in VARIANTS.items():
            print(f"  {v_name:16s} ... ", end="", flush=True)
            acc, wall, n_params = run_one(task_info, v_name, v_cfg, args.steps, args.batch, args.device)
            print(f"{acc*100:6.2f}%  ({wall:6.1f}s, {n_params:,} params)")
            all_results.append(
                {
                    "task": task_name,
                    "variant": v_name,
                    "acc": acc,
                    "wall": wall,
                    "params": n_params,
                }
            )
        print()

    print("=" * 88)
    print("SUMMARY (mean acc across tasks)")
    print("-" * 88)
    for v_name in VARIANTS:
        rows = [r for r in all_results if r["variant"] == v_name]
        mean_acc = sum(r["acc"] for r in rows) / len(rows)
        mean_wall = sum(r["wall"] for r in rows) / len(rows)
        print(f"{v_name:16s}  mean_acc={mean_acc*100:6.2f}%  mean_wall={mean_wall:6.1f}s")
    print("=" * 88)


if __name__ == "__main__":
    main()

