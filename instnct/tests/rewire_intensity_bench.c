/*
 * Rewire Intensity Benchmark
 * ===========================
 * Pure rewire strategy with variable intensity:
 *   N rewires → 1 forward → 1 eval → accept/reject ALL N at once
 *
 * Tests: does batching multiple rewires before eval help or hurt?
 * Tradeoff: more rewires per eval = fewer evals (faster per attempt)
 *           but coarser search (accept/reject is all-or-nothing).
 *
 * Compile: gcc -O3 -o rewire_intensity_bench rewire_intensity_bench.c -lm
 * Run:     ./rewire_intensity_bench [vocab] [seed] [budget]
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
#define MAX_UNDO    64

typedef struct {
    int V, N, out_start;
    float *mask;
    float *charges, *acts, *raw;
    int *alive_r, *alive_c;
    float *alive_s;
    int alive_n, alive_cap;
    int loss_pct;
    /* Multi-op undo for rewire batch */
    int undo_idx[MAX_UNDO];
    int undo_old_c[MAX_UNDO];
    int undo_n;
    uint32_t mt[624];
    int mt_idx;
} Net;

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
    y ^= y >> 11; y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u; y ^= y >> 18;
    return y;
}

static double mt_rand_float(Net *n) {
    uint32_t a = mt_rand(n) >> 5, b = mt_rand(n) >> 6;
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

int net_init(Net *n, int vocab, uint32_t seed) {
    memset(n, 0, sizeof(Net));
    n->V = vocab; n->N = vocab * NV_RATIO;
    n->out_start = (n->N >= 2 * vocab) ? n->N - vocab : 0;
    n->mt[0] = seed;
    for (int i = 1; i < 624; i++)
        n->mt[i] = 1812433253u * (n->mt[i-1] ^ (n->mt[i-1] >> 30)) + (uint32_t)i;
    n->mt_idx = 624;
    n->loss_pct = 15;
    int NN = n->N * n->N, VN = n->V * n->N;
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

/* Batch N rewires, then eval once. Returns number actually applied. */
int batch_rewire(Net *n, int count) {
    n->undo_n = 0;
    int applied = 0;
    for (int i = 0; i < count && n->undo_n < MAX_UNDO; i++) {
        if (!n->alive_n) break;
        int idx = ri(n, 0, n->alive_n - 1);
        int r = n->alive_r[idx], c = n->alive_c[idx];
        int nc = ri(n, 0, n->N - 1);
        if (nc == r || nc == c || n->mask[r * n->N + nc] != 0) continue;
        float val = n->mask[r * n->N + c];
        n->mask[r * n->N + c] = 0;
        n->mask[r * n->N + nc] = val;
        n->alive_c[idx] = nc;
        n->alive_s[idx] = val;  /* sign unchanged */
        /* save undo */
        n->undo_idx[n->undo_n] = idx;
        n->undo_old_c[n->undo_n] = c;
        n->undo_n++;
        applied++;
    }
    return applied;
}

void undo_rewires(Net *n) {
    for (int i = n->undo_n - 1; i >= 0; i--) {
        int idx = n->undo_idx[i];
        int r = n->alive_r[idx];
        int cur_c = n->alive_c[idx];
        int old_c = n->undo_old_c[i];
        float val = n->mask[r * n->N + cur_c];
        n->mask[r * n->N + cur_c] = 0;
        n->mask[r * n->N + old_c] = val;
        n->alive_c[idx] = old_c;
    }
}

typedef struct {
    float best_score;
    int total_evals;
    int accepted;
    double elapsed_sec;
} Result;

Result train_rewire(Net *n, const int *targets, int intensity, int budget) {
    Result res = {0};
    forward(n);
    float score = evaluate(n, targets);
    res.best_score = score;
    int stale = 0;

    clock_t t0 = clock();

    for (int att = 0; att < budget; att++) {
        int old_loss = n->loss_pct;
        if (ri(n, 1, 5) == 1) {
            n->loss_pct += ri(n, -3, 3);
            if (n->loss_pct < 1) n->loss_pct = 1;
            if (n->loss_pct > 50) n->loss_pct = 50;
        }

        int applied = batch_rewire(n, intensity);
        if (applied == 0) { n->loss_pct = old_loss; continue; }

        forward(n);
        float s = evaluate(n, targets);
        res.total_evals++;

        if (s > score) {
            score = s;
            if (s > res.best_score) res.best_score = s;
            stale = 0;
            res.accepted++;
        } else {
            undo_rewires(n);
            n->loss_pct = old_loss;
            stale++;
        }

        if (res.best_score >= 0.99f || stale >= 6000) break;
    }

    res.elapsed_sec = (double)(clock() - t0) / CLOCKS_PER_SEC;
    return res;
}

int main(int argc, char **argv) {
    int vocab = (argc > 1) ? atoi(argv[1]) : 64;
    uint32_t base_seed = (argc > 2) ? (uint32_t)atoi(argv[2]) : 42;
    int budget = (argc > 3) ? atoi(argv[3]) : 8000;

    int intensities[] = {1, 2, 3, 5, 7, 10, 15, 20, 30, 50};
    int n_int = sizeof(intensities) / sizeof(intensities[0]);

    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║  REWIRE INTENSITY BENCHMARK — Pure Rewire, Batch Eval   ║\n");
    printf("╠═══════════════════════════════════════════════════════════╣\n");
    printf("║  V=%d  N=%d  Budget=%d attempts per intensity            ║\n",
           vocab, vocab * NV_RATIO, budget);
    printf("║  Pattern: rewire(N) → forward → eval → accept/reject    ║\n");
    printf("╚═══════════════════════════════════════════════════════════╝\n\n");

    /* Generate targets */
    Net tmp;
    net_init(&tmp, vocab, base_seed);
    int *targets = (int *)malloc(vocab * sizeof(int));
    for (int i = 0; i < vocab; i++) targets[i] = i;
    for (int i = vocab - 1; i > 0; i--) {
        int j = (int)np_interval(&tmp, (uint32_t)i);
        int t = targets[i]; targets[i] = targets[j]; targets[j] = t;
    }
    net_free(&tmp);

    printf("%-6s %6s %6s %6s %8s %7s %7s  %s\n",
           "N_rew", "Score%", "Evals", "Accpt", "Time(s)", "ms/eval", "ms/rew", "Bar");
    printf("──────────────────────────────────────────────────────────────────────\n");

    float best_overall = 0;
    int best_intensity = 0;

    for (int i = 0; i < n_int; i++) {
        int intensity = intensities[i];
        Net n;
        net_init(&n, vocab, base_seed);
        for (int k = vocab - 1; k > 0; k--)
            np_interval(&n, (uint32_t)k);

        Result r = train_rewire(&n, targets, intensity, budget);

        double ms_eval = (r.total_evals > 0) ?
            r.elapsed_sec / r.total_evals * 1000.0 : 0;
        double ms_rew = ms_eval / intensity;
        float accept_rate = (r.total_evals > 0) ?
            100.0f * r.accepted / r.total_evals : 0;

        /* Visual bar */
        int bar_len = (int)(r.best_score * 60);
        if (bar_len > 60) bar_len = 60;
        char bar[61];
        for (int b = 0; b < bar_len; b++) bar[b] = '#';
        bar[bar_len] = '\0';

        printf("%-6d %5.1f%% %6d %5.1f%% %7.2fs %6.2fms %6.3fms  %s",
               intensity, r.best_score * 100,
               r.total_evals, accept_rate,
               r.elapsed_sec, ms_eval, ms_rew, bar);

        if (r.best_score > best_overall) {
            best_overall = r.best_score;
            best_intensity = intensity;
            printf(" ← BEST");
        }
        printf("\n");

        net_free(&n);
    }

    printf("──────────────────────────────────────────────────────────────────────\n");
    printf("\nBest: intensity=%d → %.1f%%\n", best_intensity, best_overall * 100);
    printf("\nTradeoff: higher intensity = fewer evals (faster) but coarser search\n");
    printf("Sweet spot: where score/time is maximized\n");

    free(targets);
    return 0;
}
