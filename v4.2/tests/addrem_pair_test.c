/*
 * A/B test: drive-only vs add+rem pair (joint vs sequential eval)
 *
 * Control:    single signed drive, +N=add, -N=remove
 * Pair-joint: add_n(0-15) + rem_n(0-15) executed together, 1 eval, full revert
 * Pair-seq:   add_n executed+eval+revert, then rem_n executed+eval+revert
 *
 * gcc -O3 -o addrem_pair_test addrem_pair_test.c -lm
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
#define DRIVE_VAL   0.6f
#define THRESHOLD   0.5f
#define MAX_UNDO    40

typedef struct {
    int V, N, out_start;
    float *mask;
    float *charges, *acts, *raw;
    int *alive_r, *alive_c;
    float *alive_s;
    int alive_n, alive_cap;
    int loss_pct;

    char  undo_op[MAX_UNDO];
    int   undo_r[MAX_UNDO], undo_c[MAX_UNDO];
    float undo_x[MAX_UNDO];
    int   undo_wi[MAX_UNDO];
    int   undo_n;

    uint32_t mt[624];
    int mt_idx;
} Net;

/* --- RNG --- */
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

/* --- Net --- */
static int net_init(Net *n, int vocab, uint32_t seed) {
    memset(n, 0, sizeof(Net));
    n->V = vocab;
    n->N = vocab * NV_RATIO;
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
            if (r < d / 2) val = -DRIVE_VAL;
            else if (r > 1.0 - d / 2) val = DRIVE_VAL;
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

static void net_free(Net *n) {
    free(n->mask); free(n->charges); free(n->acts); free(n->raw);
    free(n->alive_r); free(n->alive_c); free(n->alive_s);
}

/* Copy full net state (for identical starting point per strategy) */
static int net_copy(Net *dst, const Net *src) {
    memcpy(dst, src, sizeof(Net));
    int NN = src->N * src->N, VN = src->V * src->N;
    dst->mask    = (float *)malloc(NN * sizeof(float));
    dst->charges = (float *)calloc(VN, sizeof(float));
    dst->acts    = (float *)calloc(VN, sizeof(float));
    dst->raw     = (float *)calloc(VN, sizeof(float));
    dst->alive_r = (int *)malloc(dst->alive_cap * sizeof(int));
    dst->alive_c = (int *)malloc(dst->alive_cap * sizeof(int));
    dst->alive_s = (float *)malloc(dst->alive_cap * sizeof(float));
    if (!dst->mask || !dst->charges || !dst->acts || !dst->raw ||
        !dst->alive_r || !dst->alive_c || !dst->alive_s) return -1;
    memcpy(dst->mask, src->mask, NN * sizeof(float));
    memcpy(dst->alive_r, src->alive_r, src->alive_n * sizeof(int));
    memcpy(dst->alive_c, src->alive_c, src->alive_n * sizeof(int));
    memcpy(dst->alive_s, src->alive_s, src->alive_n * sizeof(float));
    return 0;
}

/* --- Forward + Eval --- */
static void forward(Net *n) {
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

static float eval_score(Net *n, const int *targets) {
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

/* --- Mutation ops --- */
static void op_add(Net *n) {
    int r = ri(n, 0, n->N - 1), c = ri(n, 0, n->N - 1);
    if (r != c && n->mask[r * n->N + c] == 0 && n->alive_n < n->alive_cap && n->undo_n < MAX_UNDO) {
        float val = ri(n, 0, 1) ? DRIVE_VAL : -DRIVE_VAL;
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
    if (!n->alive_n || n->undo_n >= MAX_UNDO) return;
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

static void replay(Net *n) {
    for (int i = n->undo_n - 1; i >= 0; i--) {
        switch (n->undo_op[i]) {
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
        }
    }
    n->undo_n = 0;
}

/* Partial replay: undo from undo_n down to 'from' */
static void replay_from(Net *n, int from) {
    for (int i = n->undo_n - 1; i >= from; i--) {
        switch (n->undo_op[i]) {
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
        }
    }
    n->undo_n = from;
}

/* --- Strategy 1: single signed drive --- */
static float train_drive(Net *n, const int *targets, int budget,
                         int *out_conns, int *out_drive, int *trace, int *trace_n) {
    forward(n);
    float score = eval_score(n, targets), best = score;
    int stale = 0, drive = 1;
    *trace_n = 0;
    trace[(*trace_n)++] = drive;

    for (int att = 0; att < budget; att++) {
        int old_loss = n->loss_pct, old_drive = drive;
        if (ri(n, 1, 5) == 1)
            n->loss_pct = fmax(1, fmin(50, n->loss_pct + ri(n, -3, 3)));
        if (ri(n, 1, 20) <= 7)
            drive = fmax(-15, fmin(15, drive + (ri(n, 0, 1) ? 1 : -1)));

        n->undo_n = 0;
        if (drive > 0)
            for (int i = 0; i < drive; i++) op_add(n);
        else if (drive < 0)
            for (int i = 0; i < -drive; i++) op_remove(n);

        forward(n);
        float s = eval_score(n, targets);
        if (s > score) { score = s; if (s > best) best = s; stale = 0; }
        else { replay(n); n->loss_pct = old_loss; drive = old_drive; stale++; }

        if ((att + 1) % 500 == 0 && *trace_n < 64) trace[(*trace_n)++] = drive;
        if (best >= 0.99f || stale >= 6000) break;
    }
    *out_conns = n->alive_n;
    *out_drive = drive;
    return best;
}

/* --- Strategy 2: pair-joint (add_n + rem_n, single eval) --- */
static float train_pair_joint(Net *n, const int *targets, int budget,
                              int *out_conns, int *out_add, int *out_rem,
                              int *trace_a, int *trace_r, int *trace_n) {
    forward(n);
    float score = eval_score(n, targets), best = score;
    int stale = 0, add_n = 1, rem_n = 0;
    *trace_n = 0;
    trace_a[(*trace_n)] = add_n; trace_r[(*trace_n)++] = rem_n;

    for (int att = 0; att < budget; att++) {
        int old_loss = n->loss_pct, old_add = add_n, old_rem = rem_n;
        if (ri(n, 1, 5) == 1)
            n->loss_pct = fmax(1, fmin(50, n->loss_pct + ri(n, -3, 3)));
        if (ri(n, 1, 20) <= 7)
            add_n = fmax(0, fmin(15, add_n + (ri(n, 0, 1) ? 1 : -1)));
        if (ri(n, 1, 20) <= 7)
            rem_n = fmax(0, fmin(15, rem_n + (ri(n, 0, 1) ? 1 : -1)));

        n->undo_n = 0;
        for (int i = 0; i < add_n; i++) op_add(n);
        for (int i = 0; i < rem_n; i++) op_remove(n);

        forward(n);
        float s = eval_score(n, targets);
        if (s > score) { score = s; if (s > best) best = s; stale = 0; }
        else { replay(n); n->loss_pct = old_loss; add_n = old_add; rem_n = old_rem; stale++; }

        if ((att + 1) % 500 == 0 && *trace_n < 64) {
            trace_a[*trace_n] = add_n; trace_r[(*trace_n)++] = rem_n;
        }
        if (best >= 0.99f || stale >= 6000) break;
    }
    *out_conns = n->alive_n;
    *out_add = add_n; *out_rem = rem_n;
    return best;
}

/* --- Strategy 3: pair-sequential (add_n eval, then rem_n eval, independent) --- */
static float train_pair_seq(Net *n, const int *targets, int budget,
                            int *out_conns, int *out_add, int *out_rem,
                            int *trace_a, int *trace_r, int *trace_n) {
    forward(n);
    float score = eval_score(n, targets), best = score;
    int stale = 0, add_n = 1, rem_n = 0;
    *trace_n = 0;
    trace_a[(*trace_n)] = add_n; trace_r[(*trace_n)++] = rem_n;

    for (int att = 0; att < budget; att++) {
        int old_loss = n->loss_pct, old_add = add_n, old_rem = rem_n;
        int any_ok = 0;

        if (ri(n, 1, 5) == 1)
            n->loss_pct = fmax(1, fmin(50, n->loss_pct + ri(n, -3, 3)));
        if (ri(n, 1, 20) <= 7)
            add_n = fmax(0, fmin(15, add_n + (ri(n, 0, 1) ? 1 : -1)));
        if (ri(n, 1, 20) <= 7)
            rem_n = fmax(0, fmin(15, rem_n + (ri(n, 0, 1) ? 1 : -1)));

        /* Phase A: adds */
        n->undo_n = 0;
        for (int i = 0; i < add_n; i++) op_add(n);

        forward(n);
        float s_after_add = eval_score(n, targets);
        if (s_after_add > score) {
            score = s_after_add; if (s_after_add > best) best = s_after_add;
            any_ok = 1;
        } else {
            replay_from(n, 0);
            add_n = old_add;
        }

        /* Phase B: removes */
        int rem_undo_start = n->undo_n;
        for (int i = 0; i < rem_n; i++) op_remove(n);

        forward(n);
        float s_after_rem = eval_score(n, targets);
        if (s_after_rem > score) {
            score = s_after_rem; if (s_after_rem > best) best = s_after_rem;
            any_ok = 1;
        } else {
            replay_from(n, rem_undo_start);
            rem_n = old_rem;
        }

        if (any_ok) { stale = 0; }
        else { n->loss_pct = old_loss; stale++; }

        if ((att + 1) % 500 == 0 && *trace_n < 64) {
            trace_a[*trace_n] = add_n; trace_r[(*trace_n)++] = rem_n;
        }
        if (best >= 0.99f || stale >= 6000) break;
    }
    *out_conns = n->alive_n;
    *out_add = add_n; *out_rem = rem_n;
    return best;
}

/* --- Targets (numpy-compatible shuffle) --- */
static void make_targets(Net *n, int *targets, int V) {
    for (int i = 0; i < V; i++) targets[i] = i;
    for (int i = V - 1; i > 0; i--) {
        int j = ri(n, 0, i);
        int tmp = targets[i]; targets[i] = targets[j]; targets[j] = tmp;
    }
}

int main(void) {
    int seeds[] = {0, 1, 2, 10, 42};
    int n_seeds = 5;
    int V = 64, budget = 16000;

    printf("A/B/C: drive vs pair-joint vs pair-seq | V=%d budget=%d\n", V, budget);
    printf("%6s  %9s  %9s  %9s  %8s %8s  %8s %8s  %6s %6s %6s\n",
           "seed", "drive", "pJoint", "pSeq", "d-j", "d-s",
           "jFinal", "sFinal", "dConn", "jConn", "sConn");
    printf("------------------------------------------"
           "------------------------------------------"
           "----------------------------\n");

    float d_total = 0, j_total = 0, s_total = 0;

    for (int si = 0; si < n_seeds; si++) {
        uint32_t seed = seeds[si];

        /* Build template net */
        Net tmpl;
        net_init(&tmpl, V, seed);
        int targets[64];
        make_targets(&tmpl, targets, V);

        /* Save RNG state after init+targets (all 3 start from same point) */
        /* Strategy 1: drive */
        Net n1; net_copy(&n1, &tmpl);
        memcpy(n1.mt, tmpl.mt, sizeof(tmpl.mt));
        n1.mt_idx = tmpl.mt_idx;
        int d_conns, d_drive, d_trace[64], d_tn;
        float d_score = train_drive(&n1, targets, budget, &d_conns, &d_drive, d_trace, &d_tn);

        /* Strategy 2: pair-joint */
        Net n2; net_copy(&n2, &tmpl);
        memcpy(n2.mt, tmpl.mt, sizeof(tmpl.mt));
        n2.mt_idx = tmpl.mt_idx;
        int j_conns, j_add, j_rem, ja_trace[64], jr_trace[64], j_tn;
        float j_score = train_pair_joint(&n2, targets, budget, &j_conns, &j_add, &j_rem,
                                         ja_trace, jr_trace, &j_tn);

        /* Strategy 3: pair-seq */
        Net n3; net_copy(&n3, &tmpl);
        memcpy(n3.mt, tmpl.mt, sizeof(tmpl.mt));
        n3.mt_idx = tmpl.mt_idx;
        int s_conns, s_add, s_rem, sa_trace[64], sr_trace[64], s_tn;
        float s_score = train_pair_seq(&n3, targets, budget, &s_conns, &s_add, &s_rem,
                                       sa_trace, sr_trace, &s_tn);

        d_total += d_score; j_total += j_score; s_total += s_score;

        printf("%6d  %8.1f%%  %8.1f%%  %8.1f%%  %+7.1f%% %+7.1f%%  %+d/%d    %+d/%d   %6d %6d %6d\n",
               seeds[si],
               d_score * 100, j_score * 100, s_score * 100,
               (j_score - d_score) * 100, (s_score - d_score) * 100,
               j_add, j_rem, s_add, s_rem,
               d_conns, j_conns, s_conns);

        /* Print trajectories */
        printf("       drv: [");
        for (int i = 0; i < d_tn; i++) printf("%s%+d", i ? " " : "", d_trace[i]);
        printf("]\n");
        printf("       jnt: [");
        for (int i = 0; i < j_tn; i++) printf("%s+%d/-%d", i ? " " : "", ja_trace[i], jr_trace[i]);
        printf("]\n");
        printf("       seq: [");
        for (int i = 0; i < s_tn; i++) printf("%s+%d/-%d", i ? " " : "", sa_trace[i], sr_trace[i]);
        printf("]\n");

        fflush(stdout);
        net_free(&n1); net_free(&n2); net_free(&n3); net_free(&tmpl);
    }

    printf("------------------------------------------"
           "------------------------------------------"
           "----------------------------\n");
    printf("%6s  %8.1f%%  %8.1f%%  %8.1f%%  %+7.1f%% %+7.1f%%\n",
           "avg",
           d_total / n_seeds * 100,
           j_total / n_seeds * 100,
           s_total / n_seeds * 100,
           (j_total - d_total) / n_seeds * 100,
           (s_total - d_total) / n_seeds * 100);

    return 0;
}
