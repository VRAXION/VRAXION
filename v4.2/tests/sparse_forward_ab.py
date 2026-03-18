"""
Sparse Edge-List Forward — A/B Benchmark
==========================================
Compares dense matmul (`act @ mask`) vs sparse edge-list scatter forward.

The SWG mask is ~4% dense with ternary values {-0.6, 0, +0.6}.
Instead of a full N×N matmul, we can scatter over only the live edges:
    out[dst] += act[src] * sign

Approaches tested:
  A) Dense baseline: `act @ mask` (NumPy BLAS)
  B) Sparse edge-list: vectorized np.add.at scatter
  C) Sparse edge-list: sorted-source gather (contiguous reads)
  D) Sparse CSR via scipy (existing approach, for reference)
  E) Sparse edge-list in C via ctypes (hand-rolled loop)

For each approach we measure:
  - Correctness: outputs match dense baseline
  - Throughput: ms per forward_batch call at various V sizes
  - Scaling: how ms grows with V
"""

import sys, os, time, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.graph import SelfWiringGraph

# ── Approach A: Dense baseline ──────────────────────────────────────

def forward_batch_dense(net, ticks=8):
    """Original dense forward. Baseline."""
    V, N = net.V, net.N
    charges = np.zeros((V, N), dtype=np.float32)
    acts = np.zeros((V, N), dtype=np.float32)
    retain = float(net.retention)
    for t in range(ticks):
        if t == 0:
            acts[:, :V] = np.eye(V, dtype=np.float32)
        raw = acts @ net.mask
        np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        charges += raw
        charges *= retain
        acts = np.maximum(charges - net.THRESHOLD, 0.0)
        charges = np.clip(charges, -1.0, 1.0)
    return charges[:, net.out_start:net.out_start + V]


# ── Approach B: np.add.at scatter ───────────────────────────────────

def build_edge_arrays(net):
    """Convert alive list to sorted numpy arrays for vectorized ops."""
    if not net.alive:
        return (np.array([], dtype=np.int32),
                np.array([], dtype=np.int32),
                np.array([], dtype=np.float32))
    edges = np.array(net.alive, dtype=np.int32)
    src = edges[:, 0]
    dst = edges[:, 1]
    vals = net.mask[src, dst].astype(np.float32)
    # Sort by src for cache-friendly access on the act vector
    order = np.argsort(src)
    return src[order], dst[order], vals[order]


def forward_batch_scatter(net, ticks=8, edge_cache=None):
    """Sparse scatter forward using np.add.at.
    O(edges × V × ticks) vs O(N² × V × ticks) for dense."""
    V, N = net.V, net.N
    if edge_cache is not None:
        src, dst, vals = edge_cache
    else:
        src, dst, vals = build_edge_arrays(net)

    E = len(src)
    if E == 0:
        return np.zeros((V, V), dtype=np.float32)

    charges = np.zeros((V, N), dtype=np.float32)
    acts = np.zeros((V, N), dtype=np.float32)
    retain = float(net.retention)

    for t in range(ticks):
        if t == 0:
            acts[:, :V] = np.eye(V, dtype=np.float32)
        # Scatter: raw[b, dst[e]] += acts[b, src[e]] * vals[e]
        # acts[:, src] shape: (V, E), vals shape: (E,)
        contrib = acts[:, src] * vals[np.newaxis, :]  # (V, E)
        raw = np.zeros((V, N), dtype=np.float32)
        np.add.at(raw, (slice(None), dst), contrib)

        charges += raw
        charges *= retain
        acts = np.maximum(charges - net.THRESHOLD, 0.0)
        charges = np.clip(charges, -1.0, 1.0)

    return charges[:, net.out_start:net.out_start + V]


# ── Approach C: Gather-based (column scatter via bincount) ──────────

def forward_batch_gather(net, ticks=8, edge_cache=None):
    """Sparse forward using gather + manual bincount-style accumulation.
    Groups edges by destination for efficient write patterns."""
    V, N = net.V, net.N
    if edge_cache is not None:
        src, dst, vals = edge_cache
    else:
        src, dst, vals = build_edge_arrays(net)

    E = len(src)
    if E == 0:
        return np.zeros((V, V), dtype=np.float32)

    # Pre-sort by dst for write-friendly grouping
    order = np.argsort(dst)
    src_d = src[order]
    dst_d = dst[order]
    vals_d = vals[order]

    charges = np.zeros((V, N), dtype=np.float32)
    acts = np.zeros((V, N), dtype=np.float32)
    retain = float(net.retention)

    for t in range(ticks):
        if t == 0:
            acts[:, :V] = np.eye(V, dtype=np.float32)
        # Gather from src, multiply, scatter to dst
        contrib = acts[:, src_d] * vals_d[np.newaxis, :]  # (V, E)
        raw = np.zeros((V, N), dtype=np.float32)
        np.add.at(raw, (slice(None), dst_d), contrib)

        charges += raw
        charges *= retain
        acts = np.maximum(charges - net.THRESHOLD, 0.0)
        charges = np.clip(charges, -1.0, 1.0)

    return charges[:, net.out_start:net.out_start + V]


# ── Approach D: scipy CSR ───────────────────────────────────────────

def forward_batch_csr(net, ticks=8, csr_cache=None):
    """Sparse forward using scipy CSR matmul."""
    from scipy import sparse as sp
    V, N = net.V, net.N
    mask_csr = csr_cache if csr_cache is not None else sp.csr_matrix(net.mask)

    charges = np.zeros((V, N), dtype=np.float32)
    acts = np.zeros((V, N), dtype=np.float32)
    retain = float(net.retention)

    for t in range(ticks):
        if t == 0:
            acts[:, :V] = np.eye(V, dtype=np.float32)
        raw = acts @ mask_csr
        if sp.issparse(raw):
            raw = raw.toarray()
        else:
            raw = np.asarray(raw)
        np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        charges += raw
        charges *= retain
        acts = np.maximum(charges - net.THRESHOLD, 0.0)
        charges = np.clip(charges, -1.0, 1.0)

    return charges[:, net.out_start:net.out_start + V]


# ── Approach E: C edge-list scatter ─────────────────────────────────

C_SRC = r"""
#include <string.h>
#include <stdlib.h>

/*
 * Fast CSR builder: scan dense mask, output CSR arrays.
 * O(N²) scan but in C it's just memcpy-speed sequential reads.
 */
void build_csr_from_mask(
    const int N,
    const float *mask,     /* N x N dense mask */
    int *row_ptr,          /* N+1 output */
    int *col_idx,          /* E output (caller allocates worst-case N*N) */
    float *csr_vals,       /* E output */
    signed char *signs,    /* E output */
    int *out_nnz           /* actual number of nonzeros */
)
{
    int nnz = 0;
    row_ptr[0] = 0;
    for (int i = 0; i < N; i++) {
        const float *row = mask + i * N;
        for (int j = 0; j < N; j++) {
            float v = row[j];
            if (v != 0.0f) {
                col_idx[nnz] = j;
                csr_vals[nnz] = v;
                signs[nnz] = v > 0.0f ? 1 : -1;
                nnz++;
            }
        }
        row_ptr[i + 1] = nnz;
    }
    *out_nnz = nnz;
}

/*
 * V1: Naive edge scatter (batch-inner) — baseline C implementation.
 */
void edge_scatter_forward(
    const int V, const int N, const int E, const int ticks,
    const int *src, const int *dst, const float *vals,
    const float retain, const float threshold,
    const int out_start,
    float *charges,   /* V x N, zeroed by caller */
    float *acts,      /* V x N, zeroed by caller */
    float *out        /* V x V, output */
)
{
    memset(charges, 0, (size_t)V * N * sizeof(float));
    memset(acts,    0, (size_t)V * N * sizeof(float));

    for (int t = 0; t < ticks; t++) {
        if (t == 0) {
            for (int i = 0; i < V; i++)
                acts[i * N + i] = 1.0f;
        }

        for (int e = 0; e < E; e++) {
            int s = src[e];
            int d = dst[e];
            float v = vals[e];
            for (int b = 0; b < V; b++) {
                charges[b * N + d] += acts[b * N + s] * v;
            }
        }

        for (int b = 0; b < V; b++) {
            float *ch = charges + b * N;
            float *ac = acts + b * N;
            for (int j = 0; j < N; j++) {
                ch[j] *= retain;
                float a = ch[j] - threshold;
                ac[j] = a > 0.0f ? a : 0.0f;
                if (ch[j] > 1.0f) ch[j] = 1.0f;
                if (ch[j] < -1.0f) ch[j] = -1.0f;
            }
        }
    }

    for (int b = 0; b < V; b++)
        for (int j = 0; j < V; j++)
            out[b * V + j] = charges[b * N + out_start + j];
}

/*
 * V2: Batch-outer edge scatter — better cache locality.
 *     For each batch row, iterate all edges.
 *     acts[b] row stays in L1/L2 cache throughout.
 */
void edge_scatter_v2(
    const int V, const int N, const int E, const int ticks,
    const int *src, const int *dst, const float *vals,
    const float retain, const float threshold,
    const int out_start,
    float *charges,
    float *acts,
    float *out
)
{
    memset(charges, 0, (size_t)V * N * sizeof(float));
    memset(acts,    0, (size_t)V * N * sizeof(float));

    for (int t = 0; t < ticks; t++) {
        if (t == 0) {
            for (int i = 0; i < V; i++)
                acts[i * N + i] = 1.0f;
        }

        /* Batch-outer: each batch row's acts stays hot in cache */
        for (int b = 0; b < V; b++) {
            const float *act_row = acts + b * N;
            float *ch_row = charges + b * N;
            for (int e = 0; e < E; e++) {
                ch_row[dst[e]] += act_row[src[e]] * vals[e];
            }
        }

        for (int b = 0; b < V; b++) {
            float *ch = charges + b * N;
            float *ac = acts + b * N;
            for (int j = 0; j < N; j++) {
                ch[j] *= retain;
                float a = ch[j] - threshold;
                ac[j] = a > 0.0f ? a : 0.0f;
                if (ch[j] > 1.0f) ch[j] = 1.0f;
                if (ch[j] < -1.0f) ch[j] = -1.0f;
            }
        }
    }

    for (int b = 0; b < V; b++)
        for (int j = 0; j < V; j++)
            out[b * V + j] = charges[b * N + out_start + j];
}

/*
 * V3: CSR-style C forward — row-major compressed sparse.
 *     Edges grouped by source row for sequential reads.
 *     row_ptr[i]..row_ptr[i+1] = edges from node i.
 */
void csr_forward(
    const int V, const int N, const int n_rows,
    const int *row_ptr, const int *col_idx, const float *csr_vals,
    const int ticks,
    const float retain, const float threshold,
    const int out_start,
    float *charges,
    float *acts,
    float *out
)
{
    memset(charges, 0, (size_t)V * N * sizeof(float));
    memset(acts,    0, (size_t)V * N * sizeof(float));

    for (int t = 0; t < ticks; t++) {
        if (t == 0) {
            for (int i = 0; i < V; i++)
                acts[i * N + i] = 1.0f;
        }

        /* For each batch row, do CSR SpMV: out[b] = acts[b] @ M */
        for (int b = 0; b < V; b++) {
            const float *act_row = acts + b * N;
            float *ch_row = charges + b * N;
            /* For each source node i with nonzero edges */
            for (int i = 0; i < n_rows; i++) {
                float a = act_row[i];
                if (a == 0.0f) continue;  /* skip dead neurons — huge win */
                int start = row_ptr[i];
                int end   = row_ptr[i + 1];
                for (int p = start; p < end; p++) {
                    ch_row[col_idx[p]] += a * csr_vals[p];
                }
            }
        }

        for (int b = 0; b < V; b++) {
            float *ch = charges + b * N;
            float *ac = acts + b * N;
            for (int j = 0; j < N; j++) {
                ch[j] *= retain;
                float a = ch[j] - threshold;
                ac[j] = a > 0.0f ? a : 0.0f;
                if (ch[j] > 1.0f) ch[j] = 1.0f;
                if (ch[j] < -1.0f) ch[j] = -1.0f;
            }
        }
    }

    for (int b = 0; b < V; b++)
        for (int j = 0; j < V; j++)
            out[b * V + j] = charges[b * N + out_start + j];
}

/*
 * V4: Ternary sign-only CSR — no multiply, just add/subtract.
 *     Since vals are always ±DRIVE, store sign as int8 {-1,+1}.
 *     Avoids float multiply entirely: ch[col] += or -= a * DRIVE
 *     Actually uses one multiply by DRIVE per source (not per edge).
 */
void ternary_csr_forward(
    const int V, const int N, const int n_rows,
    const int *row_ptr, const int *col_idx, const signed char *signs,
    const float drive,
    const int ticks,
    const float retain, const float threshold,
    const int out_start,
    float *charges,
    float *acts,
    float *out
)
{
    memset(charges, 0, (size_t)V * N * sizeof(float));
    memset(acts,    0, (size_t)V * N * sizeof(float));

    for (int t = 0; t < ticks; t++) {
        if (t == 0) {
            for (int i = 0; i < V; i++)
                acts[i * N + i] = 1.0f;
        }

        for (int b = 0; b < V; b++) {
            const float *act_row = acts + b * N;
            float *ch_row = charges + b * N;
            for (int i = 0; i < n_rows; i++) {
                float a = act_row[i];
                if (a == 0.0f) continue;
                float a_drive = a * drive;  /* one multiply per source */
                int start = row_ptr[i];
                int end   = row_ptr[i + 1];
                for (int p = start; p < end; p++) {
                    if (signs[p] > 0)
                        ch_row[col_idx[p]] += a_drive;
                    else
                        ch_row[col_idx[p]] -= a_drive;
                }
            }
        }

        for (int b = 0; b < V; b++) {
            float *ch = charges + b * N;
            float *ac = acts + b * N;
            for (int j = 0; j < N; j++) {
                ch[j] *= retain;
                float a = ch[j] - threshold;
                ac[j] = a > 0.0f ? a : 0.0f;
                if (ch[j] > 1.0f) ch[j] = 1.0f;
                if (ch[j] < -1.0f) ch[j] = -1.0f;
            }
        }
    }

    for (int b = 0; b < V; b++)
        for (int j = 0; j < V; j++)
            out[b * V + j] = charges[b * N + out_start + j];
}
"""

_c_lib = None

def _get_c_lib():
    global _c_lib
    if _c_lib is not None:
        return _c_lib
    import ctypes, tempfile, subprocess
    src_path = os.path.join(tempfile.gettempdir(), "edge_scatter.c")
    lib_path = os.path.join(tempfile.gettempdir(), "edge_scatter.so")
    with open(src_path, 'w') as f:
        f.write(C_SRC)
    r = subprocess.run(
        ["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", lib_path, src_path],
        capture_output=True, text=True
    )
    if r.returncode != 0:
        print(f"C compile failed: {r.stderr}")
        return None
    _c_lib = ctypes.CDLL(lib_path)
    return _c_lib


def _c_call(fn_name, *args):
    """Helper to call a C function by name."""
    import ctypes
    lib = _get_c_lib()
    if lib is None:
        return None
    fn = getattr(lib, fn_name, None)
    if fn is None:
        return None
    fn(*args)
    return True


def forward_batch_c_edge(net, ticks=8, edge_cache=None):
    """V1: Naive C edge-scatter (edge-outer, batch-inner)."""
    import ctypes
    lib = _get_c_lib()
    if lib is None:
        return forward_batch_scatter(net, ticks, edge_cache)

    V, N = net.V, net.N
    if edge_cache is not None:
        src, dst, vals = edge_cache
    else:
        src, dst, vals = build_edge_arrays(net)

    E = len(src)
    if E == 0:
        return np.zeros((V, V), dtype=np.float32)

    src_c = np.ascontiguousarray(src, dtype=np.int32)
    dst_c = np.ascontiguousarray(dst, dtype=np.int32)
    vals_c = np.ascontiguousarray(vals, dtype=np.float32)
    charges = np.zeros((V, N), dtype=np.float32)
    acts = np.zeros((V, N), dtype=np.float32)
    out = np.zeros((V, V), dtype=np.float32)

    lib.edge_scatter_forward(
        ctypes.c_int(V), ctypes.c_int(N), ctypes.c_int(E), ctypes.c_int(ticks),
        src_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        dst_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        vals_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_float(float(net.retention)),
        ctypes.c_float(float(net.THRESHOLD)),
        ctypes.c_int(net.out_start),
        charges.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        acts.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    return out


def forward_batch_c_v2(net, ticks=8, edge_cache=None):
    """V2: Batch-outer C edge-scatter — better cache locality."""
    import ctypes
    lib = _get_c_lib()
    if lib is None:
        return forward_batch_scatter(net, ticks, edge_cache)

    V, N = net.V, net.N
    if edge_cache is not None:
        src, dst, vals = edge_cache
    else:
        src, dst, vals = build_edge_arrays(net)

    E = len(src)
    if E == 0:
        return np.zeros((V, V), dtype=np.float32)

    src_c = np.ascontiguousarray(src, dtype=np.int32)
    dst_c = np.ascontiguousarray(dst, dtype=np.int32)
    vals_c = np.ascontiguousarray(vals, dtype=np.float32)
    charges = np.zeros((V, N), dtype=np.float32)
    acts = np.zeros((V, N), dtype=np.float32)
    out = np.zeros((V, V), dtype=np.float32)

    lib.edge_scatter_v2(
        ctypes.c_int(V), ctypes.c_int(N), ctypes.c_int(E), ctypes.c_int(ticks),
        src_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        dst_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        vals_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_float(float(net.retention)),
        ctypes.c_float(float(net.THRESHOLD)),
        ctypes.c_int(net.out_start),
        charges.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        acts.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    return out


def build_csr_arrays(net):
    """Build CSR arrays from net mask — uses C scanner if available."""
    import ctypes
    lib = _get_c_lib()
    N = net.N

    if lib and hasattr(lib, 'build_csr_from_mask'):
        # Fast C path: scan dense mask → CSR
        max_nnz = len(net.alive) + 16  # small slack
        row_ptr = np.zeros(N + 1, dtype=np.int32)
        col_idx = np.zeros(max_nnz, dtype=np.int32)
        csr_vals = np.zeros(max_nnz, dtype=np.float32)
        signs = np.zeros(max_nnz, dtype=np.int8)
        out_nnz = ctypes.c_int(0)

        mask_c = np.ascontiguousarray(net.mask, dtype=np.float32)
        lib.build_csr_from_mask(
            ctypes.c_int(N),
            mask_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            row_ptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            col_idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            csr_vals.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            signs.ctypes.data_as(ctypes.POINTER(ctypes.c_byte)),
            ctypes.byref(out_nnz),
        )
        nnz = out_nnz.value
        return (row_ptr, col_idx[:nnz].copy(), csr_vals[:nnz].copy(), signs[:nnz].copy())

    # Fallback: numpy path
    if not net.alive:
        empty_ptr = np.zeros(N + 1, dtype=np.int32)
        return (empty_ptr,
                np.array([], dtype=np.int32),
                np.array([], dtype=np.float32),
                np.array([], dtype=np.int8))

    edges = np.array(net.alive, dtype=np.int32)
    src = edges[:, 0]
    dst = edges[:, 1]
    order = np.lexsort((dst, src))
    src = src[order]
    dst = dst[order]
    vals = net.mask[src, dst].astype(np.float32)
    signs_arr = np.where(vals > 0, np.int8(1), np.int8(-1))
    counts = np.bincount(src, minlength=N)
    row_ptr = np.zeros(N + 1, dtype=np.int32)
    np.cumsum(counts, out=row_ptr[1:])
    return (row_ptr,
            np.ascontiguousarray(dst),
            np.ascontiguousarray(vals),
            np.ascontiguousarray(signs_arr))


def forward_batch_c_csr(net, ticks=8, csr_cache=None):
    """V3: CSR C forward — row-compressed, skips zero activations."""
    import ctypes
    lib = _get_c_lib()
    if lib is None:
        return forward_batch_scatter(net, ticks)

    V, N = net.V, net.N
    if csr_cache is not None:
        row_ptr, col_idx, csr_vals, _ = csr_cache
    else:
        row_ptr, col_idx, csr_vals, _ = build_csr_arrays(net)

    charges = np.zeros((V, N), dtype=np.float32)
    acts = np.zeros((V, N), dtype=np.float32)
    out = np.zeros((V, V), dtype=np.float32)

    lib.csr_forward(
        ctypes.c_int(V), ctypes.c_int(N), ctypes.c_int(N),
        row_ptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        col_idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        csr_vals.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(ticks),
        ctypes.c_float(float(net.retention)),
        ctypes.c_float(float(net.THRESHOLD)),
        ctypes.c_int(net.out_start),
        charges.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        acts.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    return out


def forward_batch_c_ternary(net, ticks=8, csr_cache=None):
    """V4: Ternary CSR — signs only, no float multiply per edge."""
    import ctypes
    lib = _get_c_lib()
    if lib is None:
        return forward_batch_scatter(net, ticks)

    V, N = net.V, net.N
    if csr_cache is not None:
        row_ptr, col_idx, _, signs = csr_cache
    else:
        row_ptr, col_idx, _, signs = build_csr_arrays(net)

    charges = np.zeros((V, N), dtype=np.float32)
    acts = np.zeros((V, N), dtype=np.float32)
    out = np.zeros((V, V), dtype=np.float32)

    lib.ternary_csr_forward(
        ctypes.c_int(V), ctypes.c_int(N), ctypes.c_int(N),
        row_ptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        col_idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        signs.ctypes.data_as(ctypes.POINTER(ctypes.c_byte)),
        ctypes.c_float(float(net.DRIVE)),
        ctypes.c_int(ticks),
        ctypes.c_float(float(net.retention)),
        ctypes.c_float(float(net.THRESHOLD)),
        ctypes.c_int(net.out_start),
        charges.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        acts.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    return out


# ── Correctness check ──────────────────────────────────────────────

def check_correctness(net, ticks=8):
    """Verify all sparse approaches match the dense baseline."""
    from scipy import sparse as sp
    baseline = forward_batch_dense(net, ticks)
    edge_cache = build_edge_arrays(net)
    csr_cache_sp = sp.csr_matrix(net.mask)
    csr_cache_c = build_csr_arrays(net)

    approaches = {
        'scatter':   lambda: forward_batch_scatter(net, ticks, edge_cache),
        'gather':    lambda: forward_batch_gather(net, ticks, edge_cache),
        'csr':       lambda: forward_batch_csr(net, ticks, csr_cache_sp),
        'c_edge_v1': lambda: forward_batch_c_edge(net, ticks, edge_cache),
        'c_edge_v2': lambda: forward_batch_c_v2(net, ticks, edge_cache),
        'c_csr':     lambda: forward_batch_c_csr(net, ticks, csr_cache_c),
        'c_ternary': lambda: forward_batch_c_ternary(net, ticks, csr_cache_c),
    }

    ok = True
    for name, fn in approaches.items():
        result = fn()
        maxdiff = np.max(np.abs(result - baseline))
        status = "OK" if maxdiff < 1e-4 else f"FAIL (maxdiff={maxdiff:.6f})"
        if maxdiff >= 1e-4:
            ok = False
        print(f"  {name:12s}: maxdiff={maxdiff:.2e}  {status}")
    return ok


# ── Benchmark ──────────────────────────────────────────────────────

def bench_one(fn, warmup=2, repeats=5):
    """Time a forward call. Returns median ms."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return np.median(times)


def run_benchmark(vocab_list, ticks=8, repeats=5, seed=42):
    from scipy import sparse as sp

    print("=" * 90)
    print("SPARSE EDGE-LIST FORWARD — A/B BENCHMARK (V1–V4 + baselines)")
    print("=" * 90)
    print(f"Ticks={ticks}, Repeats={repeats}, Seed={seed}")
    print()

    # Compile C lib upfront
    print("Compiling C sparse library (4 kernels: edge_v1, edge_v2, csr, ternary_csr)...")
    lib = _get_c_lib()
    if lib:
        print("  C library compiled OK")
    else:
        print("  C library FAILED — will skip C approaches")
    print()

    # Correctness check at V=32
    print("Correctness check (V=32):")
    np.random.seed(seed)
    net32 = SelfWiringGraph(32)
    if not check_correctness(net32, ticks):
        print("CORRECTNESS FAILURE — aborting")
        return
    print()

    # Benchmark
    results = []
    all_names = ['dense', 'csr_scipy', 'c_edge_v1', 'c_edge_v2', 'c_csr', 'c_ternary']
    header = f"{'V':>5} {'N':>5} {'E':>6} {'d%':>5} | " + " ".join(f"{n:>10}" for n in all_names) + " | best     vs_dense"
    print(header)
    print("-" * len(header))

    for V in vocab_list:
        np.random.seed(seed)
        net = SelfWiringGraph(V)
        N = net.N
        E = net.count_connections()
        density = E / (N * N) * 100

        edge_cache = build_edge_arrays(net)
        csr_cache_sp = sp.csr_matrix(net.mask)
        csr_cache_c = build_csr_arrays(net)

        approaches = [
            ('dense',      lambda: forward_batch_dense(net, ticks)),
            ('csr_scipy',  lambda: forward_batch_csr(net, ticks, csr_cache_sp)),
        ]
        if lib:
            approaches += [
                ('c_edge_v1',  lambda: forward_batch_c_edge(net, ticks, edge_cache)),
                ('c_edge_v2',  lambda: forward_batch_c_v2(net, ticks, edge_cache)),
                ('c_csr',      lambda: forward_batch_c_csr(net, ticks, csr_cache_c)),
                ('c_ternary',  lambda: forward_batch_c_ternary(net, ticks, csr_cache_c)),
            ]

        times = {}
        for name, fn in approaches:
            times[name] = bench_one(fn, warmup=2, repeats=repeats)

        sparse_names = [k for k in times if k != 'dense']
        best_sparse = min(times[k] for k in sparse_names)
        best_name = min(sparse_names, key=lambda k: times[k])
        speedup = times['dense'] / best_sparse if best_sparse > 0 else 0

        parts = [f"{V:5d} {N:5d} {E:6d} {density:4.1f}% |"]
        for n in all_names:
            if n in times:
                parts.append(f"{times[n]:9.2f}ms")
            else:
                parts.append(f"{'—':>10s}")
        parts.append(f"| {best_name:<10s} {speedup:.2f}x")
        print(" ".join(parts))
        sys.stdout.flush()

        results.append({
            'V': V, 'N': N, 'E': E, 'density': density,
            'times': times, 'best': best_name, 'speedup': speedup,
        })

    # Scaling analysis
    print()
    print("Scaling analysis (ms ratios between consecutive V sizes):")
    key_names = ['dense', 'c_csr', 'c_ternary']
    for i in range(1, len(results)):
        r0, r1 = results[i-1], results[i]
        v_ratio = r1['V'] / r0['V']
        parts = [f"  V={r0['V']:>3d}→{r1['V']:<3d}:"]
        for name in key_names:
            if name in r0['times'] and name in r1['times'] and r0['times'][name] > 0:
                ms_ratio = r1['times'][name] / r0['times'][name]
                parts.append(f"{name}={ms_ratio:.2f}x")
        parts.append(f"(V²={v_ratio**2:.2f})")
        print(" ".join(parts))

    print()
    print("Summary:")
    print("  dense:      NumPy BLAS matmul — O(N²×V), highly optimized but touches all zeros")
    print("  csr_scipy:  scipy CSR sparse — good but Python overhead + conversion cost")
    print("  c_edge_v1:  C edge-scatter, edge-outer loop — poor cache (random batch access)")
    print("  c_edge_v2:  C edge-scatter, batch-outer loop — better cache (row stays hot)")
    print("  c_csr:      C CSR, skips zero activations — best for sparse activations")
    print("  c_ternary:  C CSR + ternary signs — eliminates per-edge float multiply")

    return results


# ── Main ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sparse Edge-List Forward A/B Benchmark")
    parser.add_argument("--vocab", type=str, default="16,32,64,96,128,192,256",
                        help="Comma-separated vocab sizes")
    parser.add_argument("--ticks", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    vocab_list = [int(v) for v in args.vocab.split(",")]
    run_benchmark(vocab_list, ticks=args.ticks, repeats=args.repeats, seed=args.seed)
