from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class EdgeListState:
    n: int
    rows: list[int]
    cols: list[int]
    vals: list[int]
    pos: dict[int, int]

    @classmethod
    def from_mask(cls, mask: torch.Tensor) -> "EdgeListState":
        arr = mask.detach().cpu().numpy()
        rows_np, cols_np = np.nonzero(arr)
        vals_np = arr[rows_np, cols_np].astype(np.int8, copy=False)
        rows = rows_np.astype(np.int32, copy=False).tolist()
        cols = cols_np.astype(np.int32, copy=False).tolist()
        vals = vals_np.astype(np.int8, copy=False).tolist()
        pos = {cls._key_static(mask.shape[0], row, col): idx for idx, (row, col) in enumerate(zip(rows, cols))}
        state = cls(mask.shape[0], rows, cols, vals, pos)
        state.validate()
        return state

    @staticmethod
    def _key_static(n: int, row: int, col: int) -> int:
        return row * n + col

    def key(self, row: int, col: int) -> int:
        return self._key_static(self.n, row, col)

    def __len__(self) -> int:
        return len(self.rows)

    def density(self) -> float:
        total = self.n * self.n - self.n
        return len(self.rows) / float(total) if total else 0.0

    def validate(self) -> None:
        if not (len(self.rows) == len(self.cols) == len(self.vals) == len(self.pos)):
            raise AssertionError("edge-list arrays out of sync")
        seen = set()
        for idx, (row, col, val) in enumerate(zip(self.rows, self.cols, self.vals)):
            if row == col:
                raise AssertionError("diagonal edge found in edge list")
            if val not in (-1, 1):
                raise AssertionError(f"invalid edge sign {val}")
            key = self.key(row, col)
            if key in seen:
                raise AssertionError("duplicate edge in edge list")
            seen.add(key)
            if self.pos.get(key) != idx:
                raise AssertionError("edge index map mismatch")

    def has_edge(self, row: int, col: int) -> bool:
        return self.key(row, col) in self.pos

    def get_value(self, row: int, col: int) -> int:
        return self.vals[self.pos[self.key(row, col)]]

    def set_value(self, row: int, col: int, value: int) -> None:
        self.vals[self.pos[self.key(row, col)]] = int(value)

    def add_edge(self, row: int, col: int, value: int) -> None:
        if row == col:
            raise AssertionError("diagonal edge add attempted")
        key = self.key(row, col)
        if key in self.pos:
            raise AssertionError("duplicate edge add attempted")
        idx = len(self.rows)
        self.rows.append(int(row))
        self.cols.append(int(col))
        self.vals.append(int(value))
        self.pos[key] = idx

    def remove_edge(self, row: int, col: int) -> tuple[int, int, int]:
        return self.remove_at(self.pos[self.key(row, col)])

    def remove_at(self, idx: int) -> tuple[int, int, int]:
        row = self.rows[idx]
        col = self.cols[idx]
        value = self.vals[idx]
        removed_key = self.key(row, col)
        last = len(self.rows) - 1
        if idx != last:
            last_row = self.rows[last]
            last_col = self.cols[last]
            last_val = self.vals[last]
            self.rows[idx] = last_row
            self.cols[idx] = last_col
            self.vals[idx] = last_val
            self.pos[self.key(last_row, last_col)] = idx
        self.rows.pop()
        self.cols.pop()
        self.vals.pop()
        del self.pos[removed_key]
        return row, col, value

    def build_dense_cpu(self) -> np.ndarray:
        dense = np.zeros((self.n, self.n), dtype=np.int8)
        if self.rows:
            dense[np.array(self.rows, dtype=np.int64), np.array(self.cols, dtype=np.int64)] = np.array(
                self.vals,
                dtype=np.int8,
            )
        return dense

    def rebuild_mask(self, mask: torch.Tensor) -> None:
        mask.zero_()
        if not self.rows:
            return
        rows = torch.tensor(self.rows, dtype=torch.long, device=mask.device)
        cols = torch.tensor(self.cols, dtype=torch.long, device=mask.device)
        vals = torch.tensor(self.vals, dtype=torch.int8, device=mask.device)
        mask[rows, cols] = vals

    def sample_dead_edge(self, rng: random.Random) -> tuple[int, int] | None:
        max_edges = self.n * self.n - self.n
        if len(self.rows) >= max_edges:
            return None
        probes = max(32, min(512, self.n))
        for _ in range(probes):
            row = rng.randrange(self.n)
            col = rng.randrange(self.n)
            if row != col and not self.has_edge(row, col):
                return row, col
        row_start = rng.randrange(self.n)
        col_start = rng.randrange(self.n)
        for row_off in range(self.n):
            row = (row_start + row_off) % self.n
            for col_off in range(self.n):
                col = (col_start + col_off) % self.n
                if row != col and not self.has_edge(row, col):
                    return row, col
        return None

    def sample_dead_dst(self, row: int, rng: random.Random, forbidden_dst: int | None = None) -> int | None:
        probes = max(32, min(512, self.n))
        for _ in range(probes):
            col = rng.randrange(self.n)
            if col != row and col != forbidden_dst and not self.has_edge(row, col):
                return col
        col_start = rng.randrange(self.n)
        for off in range(self.n):
            col = (col_start + off) % self.n
            if col != row and col != forbidden_dst and not self.has_edge(row, col):
                return col
        return None


def validate_mask_matches_state(mask: torch.Tensor, state: EdgeListState) -> None:
    dense = state.build_dense_cpu()
    if not np.array_equal(mask.detach().cpu().numpy(), dense):
        raise AssertionError("dense mask does not match edge-list state")


def edge_add_connection(state: EdgeListState, rng: random.Random, undo: list[tuple]) -> bool:
    target = state.sample_dead_edge(rng)
    if target is None:
        return False
    row, col = target
    value = 1 if rng.random() > 0.5 else -1
    state.add_edge(row, col, value)
    undo.append(("delete_edge", row, col))
    return True


def edge_flip_connection(state: EdgeListState, rng: random.Random, undo: list[tuple]) -> bool:
    if not state.rows:
        return False
    idx = rng.randrange(len(state.rows))
    row = state.rows[idx]
    col = state.cols[idx]
    old = state.vals[idx]
    state.vals[idx] = -old
    undo.append(("set_value", row, col, old))
    return True


def edge_remove_connection(state: EdgeListState, rng: random.Random, undo: list[tuple]) -> bool:
    if not state.rows:
        return False
    idx = rng.randrange(len(state.rows))
    row, col, value = state.remove_at(idx)
    undo.append(("restore_edge", row, col, value))
    return True


def edge_rewire_connection(state: EdgeListState, rng: random.Random, undo: list[tuple]) -> bool:
    if not state.rows:
        return False
    probes = min(32, len(state.rows))
    for _ in range(probes):
        idx = rng.randrange(len(state.rows))
        row = state.rows[idx]
        old_col = state.cols[idx]
        value = state.vals[idx]
        new_col = state.sample_dead_dst(row, rng, forbidden_dst=old_col)
        if new_col is None:
            continue
        state.remove_at(idx)
        state.add_edge(row, new_col, value)
        undo.append(("rewire", row, old_col, new_col, value))
        return True
    return False


def rollback_edge_ops(state: EdgeListState, undo: list[tuple]) -> None:
    for op in reversed(undo):
        kind = op[0]
        if kind == "delete_edge":
            _kind, row, col = op
            if state.has_edge(row, col):
                state.remove_edge(row, col)
        elif kind == "restore_edge":
            _kind, row, col, value = op
            if not state.has_edge(row, col):
                state.add_edge(row, col, value)
        elif kind == "set_value":
            _kind, row, col, old = op
            state.set_value(row, col, old)
        elif kind == "rewire":
            _kind, row, old_col, new_col, value = op
            if state.has_edge(row, new_col):
                state.remove_edge(row, new_col)
            if not state.has_edge(row, old_col):
                state.add_edge(row, old_col, value)
        else:
            raise AssertionError(f"unknown edge rollback op {kind}")

