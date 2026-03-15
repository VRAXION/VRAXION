/*
 * Self-Wiring Graph Network — Pure C, Zero Dependencies
 * ======================================================
 * Complete port of graph.py. Standalone executable.
 * Int-only arithmetic (×10 scale). Runs on any hardware.
 *
 * Compile: gcc -O2 -o graph graph.c -lm
 * Run:     ./graph [vocab] [seed] [max_attempts]
 *          ./graph 64 42 16000
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ─── Constants (all sweep-validated) ─── */
#define NV_RATIO   3
#define DENSITY    4       /* percent */
#define DRIVE      6       /* 0.6 × 10 */
#define THRESHOLD  5       /* 0.5 × 10 */
#define CLIP       10      /* 1.0 × 10 */
#define TICKS      8
#define INPUT_SCALE 10     /* 1.0 × 10 */
#define MAX_N      2048
#define MAX_EDGES  (MAX_N * MAX_N / 4)

/* ─── RNG: xorshift32, fast + deterministic ─── */
static unsigned rng;
static unsigned xor32(void) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return rng;
}
static int randint(int lo, int hi) {
    return lo + (int)(xor32() % (unsigned)(hi - lo + 1));
}

/* ─── Graph struct ─── */
typedef struct {
    int V, N, out_start;
    signed char mask[MAX_N * MAX_N];   /* {-DRIVE, 0, +DRIVE} = {-6, 0, +6} */
    int charge[MAX_N];                 /* persistent, ×10 scale */

    /* Alive edge list */
    int alive_r[MAX_EDGES];
    int alive_c[MAX_EDGES];
    int alive_n;

    /* Co-evolved params */
    int loss_pct;    /* 1-50 */
    int signal;      /* 0 or 1 */
    int grow;        /* 0 or 1 */
    int intensity;   /* 1-15 */

    /* Undo log */
    char undo_op[16];
    int  undo_r[16];
    int  undo_c[16];
    int  undo_x[16]; /* old_sign for R, new_col for W */
    int  undo_n;
} Graph;

/* ─── Init ─── */
void graph_init(Graph *g, int vocab, unsigned seed) {
    rng = seed;
    g->V = vocab;
    g->N = vocab * NV_RATIO;
    g->out_start = (g->N >= 2 * vocab) ? g->N - vocab : 0;

    /* Zero everything */
    memset(g->mask, 0, g->N * g->N);
    memset(g->charge, 0, g->N * sizeof(int));
    g->alive_n = 0;
    g->loss_pct = 15;
    g->signal = 0;
    g->grow = 1;
    g->intensity = 7;

    /* Random ternary mask (4% density) */
    int N = g->N;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == j) continue;
            int r = xor32() % 10000;
            if (r < DENSITY * 100 / 2) {
                g->mask[i * N + j] = -DRIVE;
                g->alive_r[g->alive_n] = i;
                g->alive_c[g->alive_n] = j;
                g->alive_n++;
            } else if (r >= 10000 - DENSITY * 100 / 2) {
                g->mask[i * N + j] = DRIVE;
                g->alive_r[g->alive_n] = i;
                g->alive_c[g->alive_n] = j;
                g->alive_n++;
            }
        }
    }
}

/* ─── Forward batch (int-only matmul) ─── */
void graph_forward_batch(Graph *g, int *out_charges) {
    int V = g->V, N = g->N;
    int retain = 100 - g->loss_pct;
    int charges[MAX_N * MAX_N]; /* V×N */
    int acts[MAX_N * MAX_N];    /* V×N */
    memset(charges, 0, V * N * sizeof(int));
    memset(acts, 0, V * N * sizeof(int));

    for (int t = 0; t < TICKS; t++) {
        if (t == 0) {
            for (int i = 0; i < V; i++)
                acts[i * N + i] = INPUT_SCALE;
        }
        /* Matmul: acts @ mask */
        for (int i = 0; i < V; i++) {
            for (int j = 0; j < N; j++) {
                int raw = 0;
                for (int k = 0; k < N; k++)
                    raw += acts[i * N + k] * (int)g->mask[k * N + j];
                charges[i * N + j] += raw;
            }
        }
        /* Leak + threshold + clip */
        for (int idx = 0; idx < V * N; idx++) {
            charges[idx] = charges[idx] * retain / 100;
            acts[idx] = (charges[idx] > THRESHOLD) ? charges[idx] - THRESHOLD : 0;
            if (charges[idx] > CLIP) charges[idx] = CLIP;
            if (charges[idx] < -CLIP) charges[idx] = -CLIP;
        }
    }
    /* Copy output region */
    for (int i = 0; i < V; i++)
        for (int j = 0; j < V; j++)
            out_charges[i * V + j] = charges[i * N + g->out_start + j];
}

/* ─── Evaluate (softmax + accuracy) ─── */
float graph_evaluate(Graph *g, const int *targets) {
    int V = g->V;
    int logits[MAX_N * MAX_N]; /* V×V */
    graph_forward_batch(g, logits);

    float acc = 0, tp = 0;
    for (int i = 0; i < V; i++) {
        /* Softmax row i */
        float mx = -1e30f;
        for (int j = 0; j < V; j++) {
            float v = (float)logits[i * V + j];
            if (v > mx) mx = v;
        }
        float sum = 0;
        float probs[MAX_N];
        for (int j = 0; j < V; j++) {
            probs[j] = expf((float)logits[i * V + j] - mx);
            sum += probs[j];
        }
        for (int j = 0; j < V; j++) probs[j] /= sum;
        /* Argmax */
        int pred = 0;
        for (int j = 1; j < V; j++)
            if (probs[j] > probs[pred]) pred = j;
        if (pred == targets[i]) acc += 1.0f;
        tp += probs[targets[i]];
    }
    return 0.5f * acc / V + 0.5f * tp / V;
}

/* ─── Mutation ops ─── */
static void op_add(Graph *g) {
    int N = g->N;
    int r = randint(0, N - 1), c = randint(0, N - 1);
    if (r != c && g->mask[r * N + c] == 0) {
        g->mask[r * N + c] = randint(0, 1) ? DRIVE : -DRIVE;
        g->alive_r[g->alive_n] = r;
        g->alive_c[g->alive_n] = c;
        g->alive_n++;
        int u = g->undo_n++;
        g->undo_op[u] = 'A'; g->undo_r[u] = r; g->undo_c[u] = c;
    }
}

static void op_flip(Graph *g) {
    if (g->alive_n == 0) return;
    int idx = randint(0, g->alive_n - 1);
    int r = g->alive_r[idx], c = g->alive_c[idx];
    g->mask[r * g->N + c] *= -1;
    int u = g->undo_n++;
    g->undo_op[u] = 'F'; g->undo_r[u] = r; g->undo_c[u] = c;
}

static void op_remove(Graph *g) {
    if (g->alive_n == 0) return;
    int idx = randint(0, g->alive_n - 1);
    int r = g->alive_r[idx], c = g->alive_c[idx];
    int old = g->mask[r * g->N + c];
    g->mask[r * g->N + c] = 0;
    /* Swap-to-end */
    g->alive_n--;
    g->alive_r[idx] = g->alive_r[g->alive_n];
    g->alive_c[idx] = g->alive_c[g->alive_n];
    int u = g->undo_n++;
    g->undo_op[u] = 'R'; g->undo_r[u] = r; g->undo_c[u] = c; g->undo_x[u] = old;
}

static void op_rewire(Graph *g) {
    if (g->alive_n == 0) return;
    int N = g->N;
    int idx = randint(0, g->alive_n - 1);
    int r = g->alive_r[idx], c = g->alive_c[idx];
    int nc = randint(0, N - 1);
    if (nc != r && nc != c && g->mask[r * N + nc] == 0) {
        int old = g->mask[r * N + c];
        g->mask[r * N + c] = 0;
        g->mask[r * N + nc] = old;
        g->alive_c[idx] = nc;
        int u = g->undo_n++;
        g->undo_op[u] = 'W'; g->undo_r[u] = r; g->undo_c[u] = c; g->undo_x[u] = nc;
    }
}

/* ─── Mutate (2-bit decision tree) ─── */
void graph_mutate(Graph *g) {
    g->undo_n = 0;
    /* Intensity drift — 7/20 */
    if (randint(1, 20) <= 7) {
        int d = randint(0, 1) ? 1 : -1;
        g->intensity += d;
        if (g->intensity < 1) g->intensity = 1;
        if (g->intensity > 15) g->intensity = 15;
    }
    /* Loss step — 1/5 */
    if (randint(1, 5) == 1) {
        g->loss_pct += randint(-3, 3);
        if (g->loss_pct < 1) g->loss_pct = 1;
        if (g->loss_pct > 50) g->loss_pct = 50;
    }
    /* Mask mutations */
    for (int i = 0; i < g->intensity; i++) {
        if (g->signal) {
            op_flip(g);
        } else {
            if (g->grow) {
                op_add(g);
            } else {
                if (randint(1, 10) <= 7)
                    op_remove(g);
                else
                    op_rewire(g);
            }
        }
    }
}

/* ─── Replay (undo) ─── */
void graph_replay(Graph *g) {
    int N = g->N;
    int rebuild = 0;
    for (int i = g->undo_n - 1; i >= 0; i--) {
        int r = g->undo_r[i], c = g->undo_c[i];
        switch (g->undo_op[i]) {
            case 'F':
                g->mask[r * N + c] *= -1;
                break;
            case 'A':
                g->mask[r * N + c] = 0;
                g->alive_n--;  /* was appended last */
                rebuild = 1;
                break;
            case 'R':
                g->mask[r * N + c] = (signed char)g->undo_x[i];
                g->alive_r[g->alive_n] = r;
                g->alive_c[g->alive_n] = c;
                g->alive_n++;
                rebuild = 1;
                break;
            case 'W': {
                int nc = g->undo_x[i];
                signed char val = g->mask[r * N + nc];
                g->mask[r * N + nc] = 0;
                g->mask[r * N + c] = val;
                rebuild = 1;
                break;
            }
        }
    }
    /* Rebuild alive list if structural ops happened */
    if (rebuild) {
        g->alive_n = 0;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                if (g->mask[i * N + j] != 0) {
                    g->alive_r[g->alive_n] = i;
                    g->alive_c[g->alive_n] = j;
                    g->alive_n++;
                }
    }
}

/* ─── Train ─── */
float graph_train(Graph *g, const int *targets, int max_att, int stale_limit, int verbose) {
    float score = graph_evaluate(g, targets);
    float best = score;
    int stale = 0;

    for (int att = 0; att < max_att; att++) {
        int old_loss = g->loss_pct;
        graph_mutate(g);
        float new_score = graph_evaluate(g, targets);

        if (new_score > score) {
            score = new_score;
            if (new_score > best) best = new_score;
            stale = 0;
        } else {
            graph_replay(g);
            g->loss_pct = old_loss;
            stale++;
            /* Flip strategy on reject — 7/20 */
            if (randint(1, 20) <= 7) g->signal = 1 - g->signal;
            if (randint(1, 20) <= 7) g->grow = 1 - g->grow;
        }

        if (verbose && (att + 1) % 1000 == 0) {
            const char *mode = g->signal ? "SIGNAL" : (g->grow ? "GROW" : "SHRINK");
            printf("  [%5d] Score: %5.1f%% | Conns: %4d | %s int=%d | Loss: %d%%\n",
                   att + 1, best * 100, g->alive_n, mode, g->intensity, g->loss_pct);
        }
        if (best >= 0.99f || stale >= stale_limit) break;
    }
    return best;
}

/* ─── Main ─── */
int main(int argc, char **argv) {
    int vocab = (argc > 1) ? atoi(argv[1]) : 64;
    unsigned seed = (argc > 2) ? (unsigned)atoi(argv[2]) : 42;
    int max_att = (argc > 3) ? atoi(argv[3]) : 16000;

    printf("Self-Wiring Graph — Pure C\n");
    printf("V=%d N=%d seed=%u attempts=%d\n", vocab, vocab * NV_RATIO, seed, max_att);

    Graph *g = (Graph *)calloc(1, sizeof(Graph));
    if (!g) { fprintf(stderr, "OOM\n"); return 1; }

    graph_init(g, vocab, seed);
    printf("Init: %d connections\n\n", g->alive_n);

    /* Random permutation target */
    int targets[MAX_N];
    for (int i = 0; i < vocab; i++) targets[i] = i;
    for (int i = vocab - 1; i > 0; i--) {
        int j = randint(0, i);
        int tmp = targets[i]; targets[i] = targets[j]; targets[j] = tmp;
    }

    float best = graph_train(g, targets, max_att, 6000, 1);
    printf("\nFinal: %.1f%% | Conns: %d\n", best * 100, g->alive_n);

    free(g);
    return 0;
}
