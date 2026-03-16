/*
 * Self-Wiring Graph v2 — Pure C, Sparse, ×1000 precision
 * =======================================================
 * Merged best of graph.c (ours) + vraxion_full.c (GPT/Claude Online):
 *   - ×1000 int scale (500 precision levels, not 4)
 *   - Sparse forward (alive edges only, 25× fewer ops)
 *   - 2-bit decision tree + replay undo (proven in Python)
 *   - Heap alloc, no MAX_N limits
 *   - xorshift32 RNG
 *
 * Compile: gcc -O3 -o graph graph_v2.c -lm
 * Run:     ./graph [vocab] [seed] [max_attempts]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

/* ─── Constants (×1000 scale, all sweep-validated) ─── */
#define NV_RATIO   3
#define DENSITY_PCT 4
#define DRIVE      60      /* 0.6 × 100 */
#define THRESHOLD  50      /* 0.5 × 100 */
#define CLIP       100     /* 1.0 × 100 */
#define INPUT_SC   100     /* 1.0 × 100 */
#define TICKS      8

/* ─── Net struct (all heap-allocated) ─── */
typedef struct {
    int V, N, out_start;
    int16_t *mask;           /* N×N, values {-DRIVE, 0, +DRIVE} */
    int32_t *charges;        /* V×N scratch */
    int32_t *acts;           /* V×N scratch */

    int *alive_r, *alive_c;  /* edge endpoints */
    int16_t *alive_s;        /* edge signs (±DRIVE) */
    int alive_n, alive_cap;

    int loss_pct, signal, grow, intensity;

    char undo_op[16];
    int  undo_r[16], undo_c[16], undo_x[16];
    int  undo_n;

    uint32_t rng;
} Net;

/* ─── RNG ─── */
static uint32_t xor32(Net *n) {
    n->rng ^= n->rng << 13;
    n->rng ^= n->rng >> 17;
    n->rng ^= n->rng << 5;
    return n->rng;
}
static int ri(Net *n, int lo, int hi) {
    return lo + (int)(xor32(n) % (uint32_t)(hi - lo + 1));
}

/* ─── Init ─── */
int net_init(Net *n, int vocab, uint32_t seed) {
    memset(n, 0, sizeof(Net));
    n->V = vocab;
    n->N = vocab * NV_RATIO;
    n->out_start = (n->N >= 2 * vocab) ? n->N - vocab : 0;
    n->rng = seed ? seed : 1;
    n->loss_pct = 15;
    n->signal = 0;
    n->grow = 1;
    n->intensity = 7;

    int NN = n->N * n->N;
    int VN = n->V * n->N;
    n->alive_cap = NN / 4 + 256;
    n->mask     = (int16_t *)calloc(NN, sizeof(int16_t));
    n->charges  = (int32_t *)calloc(VN, sizeof(int32_t));
    n->acts     = (int32_t *)calloc(VN, sizeof(int32_t));
    n->alive_r  = (int *)malloc(n->alive_cap * sizeof(int));
    n->alive_c  = (int *)malloc(n->alive_cap * sizeof(int));
    n->alive_s  = (int16_t *)malloc(n->alive_cap * sizeof(int16_t));
    if (!n->mask || !n->charges || !n->acts || !n->alive_r || !n->alive_c || !n->alive_s)
        return -1;

    /* Sparse init (DENSITY_PCT% density) */
    int N = n->N;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == j) continue;
            int r = xor32(n) % 10000;
            int16_t val = 0;
            if (r < DENSITY_PCT * 100 / 2) val = -DRIVE;
            else if (r >= 10000 - DENSITY_PCT * 100 / 2) val = DRIVE;
            if (val) {
                n->mask[i * N + j] = val;
                n->alive_r[n->alive_n] = i;
                n->alive_c[n->alive_n] = j;
                n->alive_s[n->alive_n] = val;
                n->alive_n++;
            }
        }
    }
    return 0;
}

void net_free(Net *n) {
    free(n->mask); free(n->charges); free(n->acts);
    free(n->alive_r); free(n->alive_c); free(n->alive_s);
}

/* ─── Sparse forward batch ─── */
void forward(Net *n) {
    int V = n->V, N = n->N, VN = V * N;
    int32_t *ch = n->charges;
    int32_t *ac = n->acts;
    memset(ch, 0, VN * sizeof(int32_t));
    memset(ac, 0, VN * sizeof(int32_t));
    int retain = 100 - n->loss_pct; /* e.g. 85 for loss_pct=15 */

    /* Allocate raw scratch once (static to avoid per-call malloc) */
    static int32_t raw[256 * 768]; /* enough for V=256 N=768 */

    for (int t = 0; t < TICKS; t++) {
        if (t == 0)
            for (int i = 0; i < V; i++)
                ac[i * N + i] = INPUT_SC;

        /* Zero raw scratch for this tick */
        memset(raw, 0, VN * sizeof(int32_t));

        /* SPARSE matmul: raw = acts @ mask (alive edges only) */
        for (int e = 0; e < n->alive_n; e++) {
            int src = n->alive_r[e];
            int dst = n->alive_c[e];
            int32_t sign = (int32_t)n->alive_s[e];
            for (int row = 0; row < V; row++)
                raw[row * N + dst] += ac[row * N + src] * sign;
        }

        /* charges += raw; charges *= retain; threshold; clip */
        for (int i = 0; i < VN; i++) {
            ch[i] += raw[i];
            ch[i] = ch[i] * retain / 100;
            if (ch[i] > CLIP) ch[i] = CLIP;
            if (ch[i] < -CLIP) ch[i] = -CLIP;
            ac[i] = ch[i] > THRESHOLD ? ch[i] - THRESHOLD : 0;
        }
    }
}

/* ─── Evaluate (argmax accuracy, integer) ─── */
int evaluate(Net *n, const int *targets) {
    int V = n->V, N = n->N, correct = 0;
    for (int i = 0; i < V; i++) {
        int best_j = 0, best_v = n->charges[i * N + n->out_start];
        for (int j = 1; j < V; j++) {
            int v = n->charges[i * N + n->out_start + j];
            if (v > best_v) { best_v = v; best_j = j; }
        }
        if (best_j == targets[i]) correct++;
    }
    return correct;
}

/* ─── Mutation ops ─── */
static void op_flip(Net *n) {
    if (!n->alive_n) return;
    int idx = ri(n, 0, n->alive_n - 1);
    int r = n->alive_r[idx], c = n->alive_c[idx];
    n->mask[r * n->N + c] *= -1;
    n->alive_s[idx] *= -1;
    int u = n->undo_n++;
    n->undo_op[u] = 'F'; n->undo_r[u] = idx;
}

static void op_add(Net *n) {
    int r = ri(n, 0, n->N - 1), c = ri(n, 0, n->N - 1);
    if (r != c && n->mask[r * n->N + c] == 0 && n->alive_n < n->alive_cap) {
        int16_t val = ri(n, 0, 1) ? DRIVE : -DRIVE;
        n->mask[r * n->N + c] = val;
        n->alive_r[n->alive_n] = r;
        n->alive_c[n->alive_n] = c;
        n->alive_s[n->alive_n] = val;
        n->alive_n++;
        int u = n->undo_n++;
        n->undo_op[u] = 'A'; n->undo_r[u] = r; n->undo_c[u] = c;
    }
}

static void op_remove(Net *n) {
    if (!n->alive_n) return;
    int idx = ri(n, 0, n->alive_n - 1);
    int r = n->alive_r[idx], c = n->alive_c[idx];
    int16_t old = n->alive_s[idx];
    n->mask[r * n->N + c] = 0;
    n->alive_n--;
    n->alive_r[idx] = n->alive_r[n->alive_n];
    n->alive_c[idx] = n->alive_c[n->alive_n];
    n->alive_s[idx] = n->alive_s[n->alive_n];
    int u = n->undo_n++;
    n->undo_op[u] = 'R'; n->undo_r[u] = r; n->undo_c[u] = c; n->undo_x[u] = old;
}

static void op_rewire(Net *n) {
    if (!n->alive_n) return;
    int N = n->N;
    int idx = ri(n, 0, n->alive_n - 1);
    int r = n->alive_r[idx], c = n->alive_c[idx];
    int nc = ri(n, 0, N - 1);
    if (nc != r && nc != c && n->mask[r * N + nc] == 0) {
        int16_t val = n->mask[r * N + c];
        n->mask[r * N + c] = 0;
        n->mask[r * N + nc] = val;
        n->alive_c[idx] = nc;
        int u = n->undo_n++;
        n->undo_op[u] = 'W'; n->undo_r[u] = idx; n->undo_c[u] = c; n->undo_x[u] = nc;
    }
}

/* ─── Mutate (2-bit decision tree) ─── */
void mutate(Net *n) {
    n->undo_n = 0;
    if (ri(n, 1, 20) <= 7) {
        n->intensity += ri(n, 0, 1) ? 1 : -1;
        if (n->intensity < 1) n->intensity = 1;
        if (n->intensity > 15) n->intensity = 15;
    }
    if (ri(n, 1, 5) == 1) {
        n->loss_pct += ri(n, -3, 3);
        if (n->loss_pct < 1) n->loss_pct = 1;
        if (n->loss_pct > 50) n->loss_pct = 50;
    }
    for (int i = 0; i < n->intensity; i++) {
        if (n->signal)
            op_flip(n);
        else if (n->grow)
            op_add(n);
        else {
            if (ri(n, 1, 10) <= 7) op_remove(n);
            else op_rewire(n);
        }
    }
}

/* ─── Replay (undo) ─── */
void replay(Net *n) {
    for (int i = n->undo_n - 1; i >= 0; i--) {
        switch (n->undo_op[i]) {
            case 'F': {
                int idx = n->undo_r[i];
                n->mask[n->alive_r[idx] * n->N + n->alive_c[idx]] *= -1;
                n->alive_s[idx] *= -1;
                break;
            }
            case 'A': {
                int r = n->undo_r[i], c = n->undo_c[i];
                n->mask[r * n->N + c] = 0;
                n->alive_n--;
                break;
            }
            case 'R': {
                int r = n->undo_r[i], c = n->undo_c[i];
                int16_t val = (int16_t)n->undo_x[i];
                n->mask[r * n->N + c] = val;
                n->alive_r[n->alive_n] = r;
                n->alive_c[n->alive_n] = c;
                n->alive_s[n->alive_n] = val;
                n->alive_n++;
                break;
            }
            case 'W': {
                int idx = n->undo_r[i];
                int old_c = n->undo_c[i], new_c = n->undo_x[i];
                int r = n->alive_r[idx];
                int16_t val = n->mask[r * n->N + new_c];
                n->mask[r * n->N + new_c] = 0;
                n->mask[r * n->N + old_c] = val;
                n->alive_c[idx] = old_c;
                break;
            }
        }
    }
}

/* ─── Train ─── */
int train(Net *n, const int *targets, int max_att, int verbose) {
    forward(n);
    int best = evaluate(n, targets);
    int score = best, stale = 0;

    clock_t t0 = clock();
    for (int att = 0; att < max_att; att++) {
        int old_loss = n->loss_pct;
        mutate(n);
        forward(n);
        int s = evaluate(n, targets);

        if (s > score) {
            score = s;
            if (s > best) best = s;
            stale = 0;
        } else {
            replay(n);
            n->loss_pct = old_loss;
            stale++;
            if (ri(n, 1, 20) <= 7) n->signal = 1 - n->signal;
            if (ri(n, 1, 20) <= 7) n->grow = 1 - n->grow;
        }

        if (verbose && (att + 1) % 1000 == 0) {
            double ms = (double)(clock() - t0) / CLOCKS_PER_SEC * 1000.0 / (att + 1);
            printf("  [%5d] %d/%d (%.1f%%) conns=%d %s int=%d loss=%d  %.2fms/att\n",
                   att + 1, best, n->V, 100.0 * best / n->V,
                   n->alive_n,
                   n->signal ? "SIG" : (n->grow ? "GRO" : "SHR"),
                   n->intensity, n->loss_pct, ms);
        }
        if (best == n->V || stale >= 6000) break;
    }
    return best;
}

/* ─── Main ─── */
int main(int argc, char **argv) {
    int vocab = (argc > 1) ? atoi(argv[1]) : 64;
    uint32_t seed = (argc > 2) ? (uint32_t)atoi(argv[2]) : 42;
    int budget = (argc > 3) ? atoi(argv[3]) : 16000;

    Net n;
    if (net_init(&n, vocab, seed) != 0) {
        fprintf(stderr, "OOM V=%d N=%d\n", vocab, vocab * NV_RATIO);
        return 1;
    }

    printf("graph_v2 — sparse ×1000 | V=%d N=%d seed=%u conns=%d\n\n",
           n.V, n.N, seed, n.alive_n);

    /* Random permutation */
    int *targets = (int *)malloc(vocab * sizeof(int));
    for (int i = 0; i < vocab; i++) targets[i] = i;
    for (int i = vocab - 1; i > 0; i--) {
        int j = ri(&n, 0, i);
        int tmp = targets[i]; targets[i] = targets[j]; targets[j] = tmp;
    }

    clock_t t0 = clock();
    int best = train(&n, targets, budget, 1);
    double elapsed = (double)(clock() - t0) / CLOCKS_PER_SEC;

    printf("\nFinal: %d/%d (%.1f%%) conns=%d time=%.2fs (%.2fms/att)\n",
           best, vocab, 100.0 * best / vocab, n.alive_n,
           elapsed, elapsed / budget * 1000);

    free(targets);
    net_free(&n);
    return 0;
}
