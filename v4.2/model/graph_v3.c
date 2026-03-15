/*
 * Self-Wiring Graph v3 — Sparse Float, Zero BLAS
 * ================================================
 * Float32 forward (precision) + sparse alive-only matmul (speed)
 * + int mutation path (fast). Best of all worlds.
 *
 * Compile: gcc -O3 -o graph_v3 graph_v3.c -lm
 * Run:     ./graph_v3 [vocab] [seed] [budget]
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

#ifndef INIT_LOSS_PCT
#define INIT_LOSS_PCT 15  /* same as Python default */
#endif

#ifndef RNG_SEED_MODE
#define RNG_SEED_MODE 1  /* hash seed for better distribution */
#endif

/* Float constants — same as Python graph.py */
#define DRIVE       0.6f
#define THRESHOLD   0.5f

typedef struct {
    int V, N, out_start;
    float *mask;              /* N×N, {-0.6, 0, +0.6} baked drive */
    float *charges;           /* V×N scratch */
    float *acts;              /* V×N scratch */
    float *raw;               /* V×N scratch */

    int *alive_r, *alive_c;
    float *alive_s;           /* edge sign with baked drive: ±0.6 */
    int alive_n, alive_cap;

    int loss_pct, signal, grow, intensity;

    char undo_op[16];
    int  undo_r[16], undo_c[16];
    float undo_x[16];        /* old sign for R, nc for W (stored as float for R) */
    int  undo_wi[16];        /* alive idx for W */
    int  undo_n;

    uint32_t rng;
} Net;

static uint32_t xor32(Net *n) {
    n->rng ^= n->rng << 13;
    n->rng ^= n->rng >> 17;
    n->rng ^= n->rng << 5;
    return n->rng;
}

static uint32_t init_rng_state(uint32_t seed) {
#if RNG_SEED_MODE == 0
    return seed ? seed : 1u;
#else
    uint32_t x = seed + 0x9E3779B9u;
    x ^= x >> 16;
    x *= 0x7FEB352Du;
    x ^= x >> 15;
    x *= 0x846CA68Bu;
    x ^= x >> 16;
    return x ? x : 1u;
#endif
}

static int ri(Net *n, int lo, int hi) {
    uint32_t range = (uint32_t)(hi - lo + 1);
    uint32_t limit = (UINT32_MAX / range) * range;
    uint32_t r;
    do { r = xor32(n); } while (r >= limit);
    return lo + (int)(r % range);
}

int net_init(Net *n, int vocab, uint32_t seed) {
    memset(n, 0, sizeof(Net));
    n->V = vocab;
    n->N = vocab * NV_RATIO;
    n->out_start = (n->N >= 2 * vocab) ? n->N - vocab : 0;
    n->rng = init_rng_state(seed);
    n->loss_pct = INIT_LOSS_PCT;
    n->signal = 0;
    n->grow = 1;
    n->intensity = 7;

    int NN = n->N * n->N;
    int VN = n->V * n->N;
    n->alive_cap = NN / 4 + 256;
    n->mask     = (float *)calloc(NN, sizeof(float));
    n->charges  = (float *)calloc(VN, sizeof(float));
    n->acts     = (float *)calloc(VN, sizeof(float));
    n->raw      = (float *)calloc(VN, sizeof(float));
    n->alive_r  = (int *)malloc(n->alive_cap * sizeof(int));
    n->alive_c  = (int *)malloc(n->alive_cap * sizeof(int));
    n->alive_s  = (float *)malloc(n->alive_cap * sizeof(float));
    if (!n->mask || !n->charges || !n->acts || !n->raw ||
        !n->alive_r || !n->alive_c || !n->alive_s) return -1;

    int N = n->N;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            if (i == j) continue;
            int r = ri(n, 0, 9999);
            float val = 0;
            if (r < DENSITY_PCT * 100 / 2) val = -DRIVE;
            else if (r >= 10000 - DENSITY_PCT * 100 / 2) val = DRIVE;
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

/* ─── SPARSE FLOAT forward ─── */
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

        /* Sparse matmul: only alive edges */
        for (int e = 0; e < n->alive_n; e++) {
            int src = n->alive_r[e], dst = n->alive_c[e];
            float sign = n->alive_s[e];
            for (int row = 0; row < V; row++)
                n->raw[row * N + dst] += n->acts[row * N + src] * sign;
        }

        for (int i = 0; i < VN; i++) {
            n->charges[i] += n->raw[i];
            n->charges[i] *= retain;
            if (n->charges[i] > 1.0f) n->charges[i] = 1.0f;
            if (n->charges[i] < -1.0f) n->charges[i] = -1.0f;
            n->acts[i] = n->charges[i] > THRESHOLD ? n->charges[i] - THRESHOLD : 0.0f;
        }
    }
}

/* ─── Evaluate (softmax scoring, same as Python) ─── */
float evaluate(Net *n, const int *targets) {
    int V = n->V, N = n->N;
    float acc = 0, tp = 0;
    for (int i = 0; i < V; i++) {
        /* Softmax row i */
        float mx = -1e30f;
        for (int j = 0; j < V; j++) {
            float v = n->charges[i * N + n->out_start + j];
            if (v > mx) mx = v;
        }
        float sum = 0;
        for (int j = 0; j < V; j++)
            sum += expf(n->charges[i * N + n->out_start + j] - mx);
        /* Argmax */
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

/* ─── Mutations (int logic, float mask values) ─── */
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
        float val = ri(n, 0, 1) ? DRIVE : -DRIVE;
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
    float old = n->alive_s[idx];
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
    int idx = ri(n, 0, n->alive_n - 1);
    int r = n->alive_r[idx], c = n->alive_c[idx];
    int nc = ri(n, 0, n->N - 1);
    if (nc != r && nc != c && n->mask[r * n->N + nc] == 0) {
        float val = n->mask[r * n->N + c];
        n->mask[r * n->N + c] = 0;
        n->mask[r * n->N + nc] = val;
        n->alive_c[idx] = nc;
        int u = n->undo_n++;
        n->undo_op[u] = 'W'; n->undo_wi[u] = idx; n->undo_r[u] = r; n->undo_c[u] = c; n->undo_x[u] = (float)nc;
    }
}

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
                float val = n->undo_x[i];
                n->mask[r * n->N + c] = val;
                n->alive_r[n->alive_n] = r;
                n->alive_c[n->alive_n] = c;
                n->alive_s[n->alive_n] = val;
                n->alive_n++;
                break;
            }
            case 'W': {
                int idx = n->undo_wi[i];
                int r = n->undo_r[i], old_c = n->undo_c[i];
                int new_c = (int)n->undo_x[i];
                float val = n->mask[r * n->N + new_c];
                n->mask[r * n->N + new_c] = 0;
                n->mask[r * n->N + old_c] = val;
                n->alive_c[idx] = old_c;
                break;
            }
        }
    }
}

float train(Net *n, const int *targets, int max_att, int verbose) {
    forward(n);
    float best = evaluate(n, targets);
    float score = best;
    int stale = 0;

    clock_t t0 = clock();
    for (int att = 0; att < max_att; att++) {
        int old_loss = n->loss_pct;
        mutate(n);
        forward(n);
        float s = evaluate(n, targets);

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
            printf("  [%5d] score=%.1f%% conns=%d %s int=%d loss=%d  %.2fms/att\n",
                   att + 1, best * 100,
                   n->alive_n,
                   n->signal ? "SIG" : (n->grow ? "GRO" : "SHR"),
                   n->intensity, n->loss_pct, ms);
        }
        if (best >= 0.99f || stale >= 6000) break;
    }
    return best;
}

int main(int argc, char **argv) {
    int vocab = (argc > 1) ? atoi(argv[1]) : 64;
    uint32_t seed = (argc > 2) ? (uint32_t)atoi(argv[2]) : 42;
    int budget = (argc > 3) ? atoi(argv[3]) : 16000;

    Net n;
    if (net_init(&n, vocab, seed) != 0) {
        fprintf(stderr, "OOM\n"); return 1;
    }

    printf("graph_v3 — sparse float | V=%d N=%d seed=%u conns=%d\n\n",
           n.V, n.N, seed, n.alive_n);

    int *targets = (int *)malloc(vocab * sizeof(int));
    for (int i = 0; i < vocab; i++) targets[i] = i;
    for (int i = vocab - 1; i > 0; i--) {
        int j = ri(&n, 0, i);
        int tmp = targets[i]; targets[i] = targets[j]; targets[j] = tmp;
    }

    clock_t t0 = clock();
    float best = train(&n, targets, budget, 1);
    double elapsed = (double)(clock() - t0) / CLOCKS_PER_SEC;

    printf("\nFinal: %.1f%% conns=%d time=%.2fs (%.2fms/att)\n",
           best * 100, n.alive_n, elapsed, elapsed / budget * 1000);

    free(targets);
    net_free(&n);
    return 0;
}
