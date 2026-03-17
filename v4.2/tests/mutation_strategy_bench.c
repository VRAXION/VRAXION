/*
 * Mutation Strategy Benchmark — Single-Op Sequential Patterns
 * ============================================================
 * Tests remove-heavy mutation patterns WITHOUT flip.
 * Each op is individually eval'd and accepted/rejected.
 *
 * Key idea: more removes than adds → network self-discovers optimal sparsity.
 * Each "cycle" is a fixed pattern like R,R,A (2 removes, 1 add).
 * Each op in the cycle: mutate(1 op) → forward → eval → accept/reject.
 *
 * Based on graph_v3.c (sparse float forward, MT19937 RNG).
 *
 * Compile: gcc -O3 -o mutation_strategy_bench mutation_strategy_bench.c -lm
 * Run:     ./mutation_strategy_bench [vocab] [seed] [budget]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

#define NV_RATIO    3
#define DENSITY_PCT 4
#define TICKS       8
#define DRIVE       0.6f
#define THRESHOLD   0.5f

typedef struct {
    int V, N, out_start;
    float *mask;
    float *charges, *acts, *raw;
    int *alive_r, *alive_c;
    float *alive_s;
    int alive_n, alive_cap;
    int loss_pct;
    /* Single-op undo (only 1 op at a time) */
    char undo_op;
    int undo_r, undo_c, undo_idx;
    float undo_val;
    uint32_t mt[624];
    int mt_idx;
} Net;

/* ─── MT19937 RNG (from graph_v3.c) ─── */
static void mt_twist(Net *n) {
    for (int i = 0; i < 624; i++) {
        uint32_t y = (n->mt[i] & 0x80000000u) | (n->mt[(i+1) % 624] & 0x7fffffffu);
        n->mt[i] = n->mt[(i + 397) % 624] ^ (y >> 1);
        if (y & 1) n->mt[i] ^= 0x9908b0dfu;
    }
    n->mt_idx = 0;
}

static uint32_t mt_rand(Net *n) {
    if (n->mt_idx >= 624) mt_twist(n);
    uint32_t y = n->mt[n->mt_idx++];
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= y >> 18;
    return y;
}

static double mt_rand_float(Net *n) {
    uint32_t a = mt_rand(n) >> 5;
    uint32_t b = mt_rand(n) >> 6;
    return ((double)a * 67108864.0 + (double)b) / 9007199254740992.0;
}

static int ri(Net *n, int lo, int hi) {
    uint32_t range = (uint32_t)(hi - lo + 1);
    uint32_t limit = (UINT32_MAX / range) * range;
    uint32_t r;
    do { r = mt_rand(n); } while (r >= limit);
    return lo + (int)(r % range);
}

static uint32_t np_interval(Net *n, uint32_t max) {
    if (max == 0) return 0;
    uint32_t mask = max;
    mask |= mask >> 1; mask |= mask >> 2; mask |= mask >> 4;
    mask |= mask >> 8; mask |= mask >> 16;
    uint32_t value;
    do { value = mt_rand(n) & mask; } while (value > max);
    return value;
}

/* ─── Init (same as graph_v3.c) ─── */
int net_init(Net *n, int vocab, uint32_t seed) {
    memset(n, 0, sizeof(Net));
    n->V = vocab;
    n->N = vocab * NV_RATIO;
    n->out_start = (n->N >= 2 * vocab) ? n->N - vocab : 0;
    n->mt[0] = seed;
    for (int i = 1; i < 624; i++)
        n->mt[i] = 1812433253u * (n->mt[i-1] ^ (n->mt[i-1] >> 30)) + (uint32_t)i;
    n->mt_idx = 624;
    n->loss_pct = 15;

    int NN = n->N * n->N;
    int VN = n->V * n->N;
    n->alive_cap = NN / 4 + 256;
    n->mask    = (float *)calloc(NN, sizeof(float));
    n->charges = (float *)calloc(VN, sizeof(float));
    n->acts    = (float *)calloc(VN, sizeof(float));
    n->raw     = (float *)calloc(VN, sizeof(float));
    n->alive_r = (int *)malloc(n->alive_cap * sizeof(int));
    n->alive_c = (int *)malloc(n->alive_cap * sizeof(int));
    n->alive_s = (float *)malloc(n->alive_cap * sizeof(float));
    if (!n->mask || !n->charges || !n->acts || !n->raw ||
        !n->alive_r || !n->alive_c || !n->alive_s) return -1;

    int N = n->N;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            if (i == j) { mt_rand_float(n); continue; }
            double r = mt_rand_float(n);
            double d = DENSITY_PCT / 100.0;
            float val = 0;
            if (r < d / 2) val = -DRIVE;
            else if (r > 1.0 - d / 2) val = DRIVE;
            if (val != 0) {
                n->mask[i * N + j] = val;
                n->alive_r[n->alive_n] = i;
                n->alive_c[n->alive_n] = j;
                n->alive_s[n->alive_n] = val;
                n->alive_n++;
            }
        }
    return 0;
}

void net_free(Net *n) {
    free(n->mask); free(n->charges); free(n->acts); free(n->raw);
    free(n->alive_r); free(n->alive_c); free(n->alive_s);
}

/* ─── Sparse forward (same as graph_v3.c) ─── */
void forward(Net *n) {
    int V = n->V, N = n->N, VN = V * N;
    float retain = (100 - n->loss_pct) * 0.01f;
    memset(n->charges, 0, VN * sizeof(float));
    memset(n->acts, 0, VN * sizeof(float));
    for (int t = 0; t < TICKS; t++) {
        if (t == 0)
            for (int i = 0; i < V; i++)
                n->acts[i * N + i] = 1.0f;
        memset(n->raw, 0, VN * sizeof(float));
        for (int e = 0; e < n->alive_n; e++) {
            int src = n->alive_r[e], dst = n->alive_c[e];
            float sign = n->alive_s[e];
            for (int row = 0; row < V; row++)
                n->raw[row * N + dst] += n->acts[row * N + src] * sign;
        }
        for (int i = 0; i < VN; i++) {
            n->charges[i] += n->raw[i];
            n->charges[i] *= retain;
            n->acts[i] = n->charges[i] > THRESHOLD ? n->charges[i] - THRESHOLD : 0.0f;
            if (n->charges[i] > 1.0f) n->charges[i] = 1.0f;
            if (n->charges[i] < -1.0f) n->charges[i] = -1.0f;
        }
    }
}

/* ─── Evaluate ─── */
float evaluate(Net *n, const int *targets) {
    int V = n->V, N = n->N;
    float acc = 0, tp = 0;
    for (int i = 0; i < V; i++) {
        float mx = -1e30f;
        for (int j = 0; j < V; j++) {
            float v = n->charges[i * N + n->out_start + j];
            if (v > mx) mx = v;
        }
        float sum = 0;
        for (int j = 0; j < V; j++)
            sum += expf(n->charges[i * N + n->out_start + j] - mx);
        int pred = 0;
        float pred_v = n->charges[i * N + n->out_start];
        for (int j = 1; j < V; j++) {
            float v = n->charges[i * N + n->out_start + j];
            if (v > pred_v) { pred_v = v; pred = j; }
        }
        if (pred == targets[i]) acc += 1.0f;
        tp += expf(n->charges[i * N + n->out_start + targets[i]] - mx) / sum;
    }
    return 0.5f * acc / V + 0.5f * tp / V;
}

/* ─── Single-op mutations (with inline undo) ─── */

/* Returns 1 if op was applied, 0 if skipped */
int do_remove(Net *n) {
    if (!n->alive_n) return 0;
    int idx = ri(n, 0, n->alive_n - 1);
    int r = n->alive_r[idx], c = n->alive_c[idx];
    float old = n->alive_s[idx];
    n->mask[r * n->N + c] = 0;
    /* swap-to-end */
    n->alive_n--;
    n->alive_r[idx] = n->alive_r[n->alive_n];
    n->alive_c[idx] = n->alive_c[n->alive_n];
    n->alive_s[idx] = n->alive_s[n->alive_n];
    /* save undo */
    n->undo_op = 'R';
    n->undo_r = r; n->undo_c = c; n->undo_val = old;
    return 1;
}

int do_add(Net *n) {
    int r = ri(n, 0, n->N - 1), c = ri(n, 0, n->N - 1);
    if (r == c || n->mask[r * n->N + c] != 0 || n->alive_n >= n->alive_cap)
        return 0;
    float val = ri(n, 0, 1) ? DRIVE : -DRIVE;
    n->mask[r * n->N + c] = val;
    n->alive_r[n->alive_n] = r;
    n->alive_c[n->alive_n] = c;
    n->alive_s[n->alive_n] = val;
    n->alive_n++;
    n->undo_op = 'A';
    n->undo_r = r; n->undo_c = c;
    return 1;
}

int do_rewire(Net *n) {
    if (!n->alive_n) return 0;
    int idx = ri(n, 0, n->alive_n - 1);
    int r = n->alive_r[idx], c = n->alive_c[idx];
    int nc = ri(n, 0, n->N - 1);
    if (nc == r || nc == c || n->mask[r * n->N + nc] != 0) return 0;
    float val = n->mask[r * n->N + c];
    n->mask[r * n->N + c] = 0;
    n->mask[r * n->N + nc] = val;
    n->alive_c[idx] = nc;
    n->undo_op = 'W';
    n->undo_r = r; n->undo_c = c; n->undo_idx = idx;
    n->undo_val = (float)nc;
    return 1;
}

void undo_last(Net *n) {
    switch (n->undo_op) {
        case 'R': {
            int r = n->undo_r, c = n->undo_c;
            float val = n->undo_val;
            n->mask[r * n->N + c] = val;
            n->alive_r[n->alive_n] = r;
            n->alive_c[n->alive_n] = c;
            n->alive_s[n->alive_n] = val;
            n->alive_n++;
            break;
        }
        case 'A': {
            int r = n->undo_r, c = n->undo_c;
            n->mask[r * n->N + c] = 0;
            n->alive_n--;
            break;
        }
        case 'W': {
            int r = n->undo_r, c = n->undo_c;
            int idx = n->undo_idx;
            int nc = (int)n->undo_val;
            float val = n->mask[r * n->N + nc];
            n->mask[r * n->N + nc] = 0;
            n->mask[r * n->N + c] = val;
            n->alive_c[idx] = c;
            break;
        }
    }
}

/* ─── Mutation patterns ─── */
/* Each pattern is a string of ops: R=remove, A=add, W=rewire */

typedef struct {
    const char *name;
    const char *pattern;  /* e.g., "RRA" = remove, remove, add */
} Strategy;

static Strategy strategies[] = {
    {"baseline_RA",   "RA"},        /* 1:1 balanced, no flip */
    {"R2_A1",         "RRA"},       /* 2:1 mild shrink */
    {"R3_A1",         "RRRA"},      /* 3:1 aggressive shrink */
    {"R4_A1",         "RRRRA"},     /* 4:1 ultra shrink */
    {"R3_A2",         "RRRAA"},     /* 3:2 slow shrink */
    {"add_first_AR",  "AR"},        /* 1:1 add-first */
    {"interleaved",   "RAR"},       /* 2:1 interleaved */
    {"rewire_heavy",  "RWA"},       /* remove, rewire, add */
    {"pure_remove",   "RRR"},       /* remove-only */
    {"pure_rewire",   "WWW"},       /* rewire-only (constant conns) */
    {NULL, NULL}
};

/* ─── Train with pattern ─── */
typedef struct {
    float best_score;
    int final_conns;
    int min_conns;
    int max_conns;
    int total_ops;
    double elapsed_sec;
} Result;

Result train_pattern(Net *n, const int *targets, const char *pattern, int budget) {
    Result res = {0};
    res.min_conns = n->alive_n;
    res.max_conns = n->alive_n;

    forward(n);
    float score = evaluate(n, targets);
    res.best_score = score;
    int stale = 0;
    int plen = (int)strlen(pattern);
    int pi = 0;  /* pattern index */

    clock_t t0 = clock();

    for (int att = 0; att < budget; att++) {
        char op = pattern[pi];
        pi = (pi + 1) % plen;

        int old_loss = n->loss_pct;
        /* Loss drift — 1/5 chance */
        if (ri(n, 1, 5) == 1) {
            n->loss_pct += ri(n, -3, 3);
            if (n->loss_pct < 1) n->loss_pct = 1;
            if (n->loss_pct > 50) n->loss_pct = 50;
        }

        int applied = 0;
        switch (op) {
            case 'R': applied = do_remove(n); break;
            case 'A': applied = do_add(n); break;
            case 'W': applied = do_rewire(n); break;
        }

        if (!applied) {
            n->loss_pct = old_loss;
            continue;
        }

        forward(n);
        float s = evaluate(n, targets);
        res.total_ops++;

        if (s > score) {
            score = s;
            if (s > res.best_score) res.best_score = s;
            stale = 0;
        } else {
            undo_last(n);
            n->loss_pct = old_loss;
            stale++;
        }

        if (n->alive_n < res.min_conns) res.min_conns = n->alive_n;
        if (n->alive_n > res.max_conns) res.max_conns = n->alive_n;

        if (res.best_score >= 0.99f || stale >= 6000) break;
    }

    res.elapsed_sec = (double)(clock() - t0) / CLOCKS_PER_SEC;
    res.final_conns = n->alive_n;
    return res;
}

int main(int argc, char **argv) {
    int vocab = (argc > 1) ? atoi(argv[1]) : 64;
    uint32_t base_seed = (argc > 2) ? (uint32_t)atoi(argv[2]) : 42;
    int budget = (argc > 3) ? atoi(argv[3]) : 8000;

    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  MUTATION STRATEGY BENCHMARK — Single-Op Sequential Patterns   ║\n");
    printf("╠══════════════════════════════════════════════════════════════════╣\n");
    printf("║  V=%d  N=%d  Budget=%d  No flip. Each op → eval → accept/reject ║\n",
           vocab, vocab * NV_RATIO, budget);
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");

    /* Generate targets once (using a temporary net for RNG) */
    Net tmp;
    net_init(&tmp, vocab, base_seed);
    int *targets = (int *)malloc(vocab * sizeof(int));
    for (int i = 0; i < vocab; i++) targets[i] = i;
    for (int i = vocab - 1; i > 0; i--) {
        int j = (int)np_interval(&tmp, (uint32_t)i);
        int t = targets[i]; targets[i] = targets[j]; targets[j] = t;
    }
    int init_conns = tmp.alive_n;
    net_free(&tmp);

    printf("%-16s %6s %6s %6s %6s %6s %8s %7s\n",
           "Strategy", "Score%", "Final", "Min", "Max", "Ops", "Time(s)", "ms/op");
    printf("─────────────────────────────────────────────────────────────────────────\n");

    for (int s = 0; strategies[s].name; s++) {
        Net n;
        net_init(&n, vocab, base_seed);
        /* consume same RNG for targets (keep init identical) */
        for (int i = vocab - 1; i > 0; i--)
            np_interval(&n, (uint32_t)i);

        Result r = train_pattern(&n, targets, strategies[s].pattern, budget);

        double ms_per_op = (r.total_ops > 0) ?
            r.elapsed_sec / r.total_ops * 1000.0 : 0;

        printf("%-16s %5.1f%% %6d %6d %6d %6d %7.2fs %6.2fms",
               strategies[s].name, r.best_score * 100,
               r.final_conns, r.min_conns, r.max_conns,
               r.total_ops, r.elapsed_sec, ms_per_op);

        /* Sparsity change indicator */
        int delta = r.final_conns - init_conns;
        if (delta < -100)
            printf("  ▼▼ SPARSE");
        else if (delta < -20)
            printf("  ▼ shrunk");
        else if (delta > 100)
            printf("  ▲▲ DENSE");
        else if (delta > 20)
            printf("  ▲ grew");
        else
            printf("  ≈ stable");
        printf("\n");

        net_free(&n);
    }

    printf("\n─────────────────────────────────────────────────────────────────────────\n");
    printf("Init conns: %d (%.1f%% density)\n", init_conns,
           100.0 * init_conns / (vocab * NV_RATIO * vocab * NV_RATIO));
    printf("\nPattern legend: R=remove, A=add, W=rewire (no flip!)\n");
    printf("Each op individually: mutate(1) → forward → eval → accept/reject\n");

    free(targets);
    return 0;
}
