/* VRAXION v4.2 - portable single-file runtime, integer-only hot path.
   Default build: gcc -std=c11 -O3 -Wall -Wextra -Wpedantic -Wconversion -o vraxion vraxion_full.c
   Reference build: gcc -std=c11 -O3 -DVRAXION_REFERENCE=1 -o vraxion_ref vraxion_full.c */

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TICKS 8
#define THRESH 50   /* 0.5 x 100 */
#define RETAIN 85   /* 85/100 = 0.85 */
#define DRIVE 6     /* 6/10 = 0.6 */
#define CLIP 100    /* +/-1.0 x 100 */

typedef enum {
    MUT_MIXED = 0,
    MUT_FLIP_ONLY = 1,
    MUT_ADD_ONLY = 2,
    MUT_REMOVE_ONLY = 3
} MutMode;

typedef enum {
    UNDO_NONE = 0,
    UNDO_FLIP = 1,
    UNDO_ADD = 2,
    UNDO_REMOVE = 3
} UndoOp;

typedef struct {
    UndoOp op;
    int32_t idx;
    int32_t old_n;
    int32_t r;
    int32_t c;
    signed char sign;
    int32_t moved_r;
    int32_t moved_c;
    signed char moved_s;
} Undo;

typedef struct {
    signed char *mask;
    int32_t *charges;
    int32_t *acts;
    int32_t *prev_acts;
    int32_t *raw;
    int32_t *alive_r;
    int32_t *alive_c;
    signed char *alive_s;
    int32_t *src_head;
    int32_t *src_next;
    int32_t alive_n;
    int32_t alive_cap;
    int32_t N;
    int32_t V;
} Net;

typedef struct {
    int ok;
    int32_t best;
    int32_t alive_n;
    uint64_t mask_hash;
    uint64_t perm_hash;
} TrainResult;

static int alloc_array(void **dst, size_t count, size_t elem_size) {
    if (count != 0U && elem_size > (SIZE_MAX / count)) {
        *dst = NULL;
        return 0;
    }
    *dst = calloc(count, elem_size);
    return *dst != NULL;
}

static int parse_i32(const char *text, int32_t *out) {
    char *end = NULL;
    long value;
    errno = 0;
    value = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        return 0;
    }
    if (value < (long)INT32_MIN || value > (long)INT32_MAX) {
        return 0;
    }
    *out = (int32_t)value;
    return 1;
}

static uint64_t fnv1a64_bytes(const void *data, size_t len) {
    const unsigned char *bytes = (const unsigned char *)data;
    uint64_t hash = UINT64_C(1469598103934665603);
    size_t i;
    for (i = 0U; i < len; ++i) {
        hash ^= (uint64_t)bytes[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static void net_free(Net *net) {
    free(net->mask);
    free(net->charges);
    free(net->acts);
    free(net->prev_acts);
    free(net->raw);
    free(net->alive_r);
    free(net->alive_c);
    free(net->alive_s);
    free(net->src_head);
    free(net->src_next);
    memset(net, 0, sizeof(*net));
}

static int net_init(Net *net, int32_t V, int32_t N) {
    size_t nn = (size_t)N * (size_t)N;
    size_t vn = (size_t)V * (size_t)N;

    memset(net, 0, sizeof(*net));
    if (nn > (size_t)INT32_MAX) {
        return 0;
    }
    net->V = V;
    net->N = N;
    net->alive_cap = (int32_t)nn;

    if (!alloc_array((void **)&net->mask, nn, sizeof(*net->mask)) ||
        !alloc_array((void **)&net->charges, vn, sizeof(*net->charges)) ||
        !alloc_array((void **)&net->acts, vn, sizeof(*net->acts)) ||
        !alloc_array((void **)&net->prev_acts, (size_t)N, sizeof(*net->prev_acts)) ||
        !alloc_array((void **)&net->raw, (size_t)N, sizeof(*net->raw)) ||
        !alloc_array((void **)&net->alive_r, nn, sizeof(*net->alive_r)) ||
        !alloc_array((void **)&net->alive_c, nn, sizeof(*net->alive_c)) ||
        !alloc_array((void **)&net->alive_s, nn, sizeof(*net->alive_s)) ||
        !alloc_array((void **)&net->src_head, (size_t)N, sizeof(*net->src_head)) ||
        !alloc_array((void **)&net->src_next, nn, sizeof(*net->src_next))) {
        net_free(net);
        return 0;
    }
    return 1;
}

static void append_edge(Net *net, int32_t r, int32_t c, signed char sign) {
    int32_t idx = net->alive_n;
    net->mask[(size_t)r * (size_t)net->N + (size_t)c] = sign;
    net->alive_r[idx] = r;
    net->alive_c[idx] = c;
    net->alive_s[idx] = sign;
    net->alive_n = idx + 1;
}

#if !defined(VRAXION_REFERENCE)
static void build_src_index(Net *net) {
    int32_t i;
    for (i = 0; i < net->N; ++i) {
        net->src_head[i] = -1;
    }
    for (i = 0; i < net->alive_n; ++i) {
        int32_t src = net->alive_r[i];
        net->src_next[i] = net->src_head[src];
        net->src_head[src] = i;
    }
}
#endif

#if defined(VRAXION_REFERENCE)
static void forward_dense(Net *net) {
    int32_t V = net->V;
    int32_t N = net->N;
    int32_t t;
    memset(net->charges, 0, (size_t)V * (size_t)N * sizeof(*net->charges));
    memset(net->acts, 0, (size_t)V * (size_t)N * sizeof(*net->acts));
    for (t = 0; t < TICKS; ++t) {
        int32_t i;
        if (t == 0) {
            for (i = 0; i < V; ++i) {
                net->acts[(size_t)i * (size_t)N + (size_t)i] = 100;
            }
        }
        for (i = 0; i < V; ++i) {
            int32_t j;
            for (j = 0; j < N; ++j) {
                int32_t k;
                int32_t raw = 0;
                for (k = 0; k < N; ++k) {
                    raw += net->acts[(size_t)i * (size_t)N + (size_t)k] *
                           (int32_t)net->mask[(size_t)k * (size_t)N + (size_t)j];
                }
                {
                    int32_t ch = net->charges[(size_t)i * (size_t)N + (size_t)j] * RETAIN / 100 +
                                 raw * DRIVE / 10;
                    if (ch > CLIP * 10) {
                        ch = CLIP * 10;
                    } else if (ch < -CLIP * 10) {
                        ch = -CLIP * 10;
                    }
                    net->charges[(size_t)i * (size_t)N + (size_t)j] = ch;
                    net->acts[(size_t)i * (size_t)N + (size_t)j] = (ch > THRESH) ? (ch - THRESH) : 0;
                }
            }
        }
    }
}
#endif

#if !defined(VRAXION_REFERENCE)
static void forward_sparse(Net *net) {
    int32_t V = net->V;
    int32_t N = net->N;
    int32_t t;

    memset(net->charges, 0, (size_t)V * (size_t)N * sizeof(*net->charges));
    memset(net->acts, 0, (size_t)V * (size_t)N * sizeof(*net->acts));
    build_src_index(net);

    for (t = 0; t < TICKS; ++t) {
        int32_t row;
        if (t == 0) {
            for (row = 0; row < V; ++row) {
                net->acts[(size_t)row * (size_t)N + (size_t)row] = 100;
            }
        }
        for (row = 0; row < V; ++row) {
            int32_t *row_acts = net->acts + (size_t)row * (size_t)N;
            int32_t *row_charges = net->charges + (size_t)row * (size_t)N;
            int32_t idx;
            int32_t dst;

            memcpy(net->prev_acts, row_acts, (size_t)N * sizeof(*net->prev_acts));
            memset(net->raw, 0, (size_t)N * sizeof(*net->raw));

            for (idx = 0; idx < net->alive_n; ++idx) {
                net->raw[net->alive_c[idx]] += net->prev_acts[net->alive_r[idx]] *
                                               (int32_t)net->alive_s[idx];
            }

            for (dst = 0; dst < N; ++dst) {
                int32_t ch = row_charges[dst] * RETAIN / 100 + net->raw[dst] * DRIVE / 10;
                int32_t new_act;
                int32_t delta;
                if (ch > CLIP * 10) {
                    ch = CLIP * 10;
                } else if (ch < -CLIP * 10) {
                    ch = -CLIP * 10;
                }
                row_charges[dst] = ch;
                new_act = (ch > THRESH) ? (ch - THRESH) : 0;
                delta = new_act - net->prev_acts[dst];
                row_acts[dst] = new_act;
                if (delta != 0) {
                    for (idx = net->src_head[dst]; idx != -1; idx = net->src_next[idx]) {
                        int32_t out = net->alive_c[idx];
                        if (out > dst) {
                            net->raw[out] += delta * (int32_t)net->alive_s[idx];
                        }
                    }
                }
            }
        }
    }
}
#endif

static void forward_net(Net *net) {
#if defined(VRAXION_REFERENCE)
    forward_dense(net);
#else
    forward_sparse(net);
#endif
}

static int32_t eval_net(const Net *net, const int32_t *perm) {
    int32_t V = net->V;
    int32_t N = net->N;
    int32_t correct = 0;
    int32_t i;
    for (i = 0; i < V; ++i) {
        int32_t best_j = 0;
        int32_t best_v = net->charges[(size_t)i * (size_t)N + (size_t)(N - V)];
        int32_t j;
        for (j = 1; j < V; ++j) {
            int32_t v = net->charges[(size_t)i * (size_t)N + (size_t)(N - V) + (size_t)j];
            if (v > best_v) {
                best_v = v;
                best_j = j;
            }
        }
        if (best_j == perm[i]) {
            correct++;
        }
    }
    return correct;
}

static void undo_none(Undo *undo) {
    memset(undo, 0, sizeof(*undo));
    undo->op = UNDO_NONE;
}

static void do_flip(Net *net, Undo *undo) {
    int32_t idx;
    if (net->alive_n == 0) {
        undo_none(undo);
        return;
    }
    idx = (int32_t)(rand() % net->alive_n);
    undo->op = UNDO_FLIP;
    undo->idx = idx;
    undo->r = net->alive_r[idx];
    undo->c = net->alive_c[idx];
    net->alive_s[idx] = (signed char)(-net->alive_s[idx]);
    net->mask[(size_t)undo->r * (size_t)net->N + (size_t)undo->c] = net->alive_s[idx];
}

static void do_add(Net *net, Undo *undo) {
    int32_t r = (int32_t)(rand() % net->N);
    int32_t c = (int32_t)(rand() % net->N);
    size_t cell = (size_t)r * (size_t)net->N + (size_t)c;
    if (r != c && net->mask[cell] == 0) {
        signed char sign = (rand() % 2) != 0 ? 1 : -1;
        undo->op = UNDO_ADD;
        undo->idx = net->alive_n;
        undo->old_n = net->alive_n;
        undo->r = r;
        undo->c = c;
        undo->sign = sign;
        append_edge(net, r, c, sign);
        return;
    }
    undo_none(undo);
}

static void do_remove(Net *net, Undo *undo) {
    int32_t idx;
    int32_t last;
    if (net->alive_n == 0) {
        undo_none(undo);
        return;
    }
    idx = (int32_t)(rand() % net->alive_n);
    last = net->alive_n - 1;

    undo->op = UNDO_REMOVE;
    undo->idx = idx;
    undo->old_n = net->alive_n;
    undo->r = net->alive_r[idx];
    undo->c = net->alive_c[idx];
    undo->sign = net->alive_s[idx];
    if (idx != last) {
        undo->moved_r = net->alive_r[last];
        undo->moved_c = net->alive_c[last];
        undo->moved_s = net->alive_s[last];
    } else {
        undo->moved_r = 0;
        undo->moved_c = 0;
        undo->moved_s = 0;
    }

    net->mask[(size_t)undo->r * (size_t)net->N + (size_t)undo->c] = 0;
    if (idx != last) {
        net->alive_r[idx] = net->alive_r[last];
        net->alive_c[idx] = net->alive_c[last];
        net->alive_s[idx] = net->alive_s[last];
    }
    net->alive_n = last;
}

#if !defined(VRAXION_REFERENCE)
static void undo_apply(Net *net, const Undo *undo) {
    if (undo->op == UNDO_FLIP) {
        net->alive_s[undo->idx] = (signed char)(-net->alive_s[undo->idx]);
        net->mask[(size_t)undo->r * (size_t)net->N + (size_t)undo->c] = net->alive_s[undo->idx];
    } else if (undo->op == UNDO_ADD) {
        net->mask[(size_t)undo->r * (size_t)net->N + (size_t)undo->c] = 0;
        net->alive_n = undo->old_n;
    } else if (undo->op == UNDO_REMOVE) {
        int32_t last = undo->old_n - 1;
        net->mask[(size_t)undo->r * (size_t)net->N + (size_t)undo->c] = undo->sign;
        if (undo->idx != last) {
            net->alive_r[last] = undo->moved_r;
            net->alive_c[last] = undo->moved_c;
            net->alive_s[last] = undo->moved_s;
        }
        net->alive_r[undo->idx] = undo->r;
        net->alive_c[undo->idx] = undo->c;
        net->alive_s[undo->idx] = undo->sign;
        net->alive_n = undo->old_n;
    }
}
#endif

static void mutate(Net *net, MutMode mode, Undo *undo) {
    undo_none(undo);
    if (mode == MUT_FLIP_ONLY) {
        do_flip(net, undo);
        return;
    }
    if (mode == MUT_ADD_ONLY) {
        do_add(net, undo);
        return;
    }
    if (mode == MUT_REMOVE_ONLY) {
        do_remove(net, undo);
        return;
    }
    {
        int32_t r = (int32_t)(rand() % 100);
        if (r < 60) {
            do_flip(net, undo);
        } else if (r < 85) {
            do_add(net, undo);
        } else {
            do_remove(net, undo);
        }
    }
}

static TrainResult train_one(int32_t V, int32_t N, int32_t seed, int32_t budget, MutMode mode) {
    TrainResult result;
    Net net;
    int32_t *perm = NULL;
    Undo undo;
    int32_t best = 0;
    int32_t att;
#if defined(VRAXION_REFERENCE)
    signed char *smask = NULL;
    int32_t *sr = NULL;
    int32_t *sc = NULL;
    signed char *ss = NULL;
    int32_t sn = 0;
#endif

    memset(&result, 0, sizeof(result));
    if (!net_init(&net, V, N)) {
        fprintf(stderr, "alloc failed for V=%" PRId32 " N=%" PRId32 "\n", V, N);
        return result;
    }
    if (!alloc_array((void **)&perm, (size_t)V, sizeof(*perm))) {
        fprintf(stderr, "alloc failed for perm V=%" PRId32 "\n", V);
        net_free(&net);
        return result;
    }

#if defined(VRAXION_REFERENCE)
    {
        size_t mask_bytes = (size_t)N * (size_t)N * sizeof(*smask);
        size_t alive_bytes_i = (size_t)net.alive_cap * sizeof(*sr);
        size_t alive_bytes_s = (size_t)net.alive_cap * sizeof(*ss);
        if (!alloc_array((void **)&smask, (size_t)N * (size_t)N, sizeof(*smask)) ||
            !alloc_array((void **)&sr, (size_t)net.alive_cap, sizeof(*sr)) ||
            !alloc_array((void **)&sc, (size_t)net.alive_cap, sizeof(*sc)) ||
            !alloc_array((void **)&ss, (size_t)net.alive_cap, sizeof(*ss))) {
            fprintf(stderr, "snapshot alloc failed for N=%" PRId32 "\n", N);
            free(perm);
            net_free(&net);
            free(smask);
            free(sr);
            free(sc);
            free(ss);
            return result;
        }
        (void)mask_bytes;
        (void)alive_bytes_i;
        (void)alive_bytes_s;
    }
#endif

    net.alive_n = 0;
    srand((unsigned int)seed);

    for (att = 0; att < N * N; ++att) {
        if (att / N == att % N) {
            continue;
        }
        {
            int32_t r = (int32_t)(rand() % 100);
            if (r < 4) {
                append_edge(&net, att / N, att % N, -1);
            } else if (r > 96) {
                append_edge(&net, att / N, att % N, 1);
            }
        }
    }

    for (att = 0; att < V; ++att) {
        perm[att] = att;
    }
    for (att = V - 1; att > 0; --att) {
        int32_t j = (int32_t)(rand() % (att + 1));
        int32_t tmp = perm[att];
        perm[att] = perm[j];
        perm[j] = tmp;
    }

    forward_net(&net);
    best = eval_net(&net, perm);

    for (att = 0; att < budget; ++att) {
#if defined(VRAXION_REFERENCE)
        size_t sn_size;
        memcpy(smask, net.mask, (size_t)N * (size_t)N * sizeof(*smask));
        sn = net.alive_n;
        sn_size = (size_t)sn;
        memcpy(sr, net.alive_r, sn_size * sizeof(*sr));
        memcpy(sc, net.alive_c, sn_size * sizeof(*sc));
        memcpy(ss, net.alive_s, sn_size * sizeof(*ss));
        mutate(&net, mode, &undo);
        forward_net(&net);
        {
            int32_t score = eval_net(&net, perm);
            if (score > best) {
                best = score;
            } else {
                memcpy(net.mask, smask, (size_t)N * (size_t)N * sizeof(*smask));
                net.alive_n = sn;
                memcpy(net.alive_r, sr, sn_size * sizeof(*sr));
                memcpy(net.alive_c, sc, sn_size * sizeof(*sc));
                memcpy(net.alive_s, ss, sn_size * sizeof(*ss));
            }
        }
#else
        mutate(&net, mode, &undo);
        forward_net(&net);
        {
            int32_t score = eval_net(&net, perm);
            if (score > best) {
                best = score;
            } else {
                undo_apply(&net, &undo);
            }
        }
#endif
    }

    result.ok = 1;
    result.best = best;
    result.alive_n = net.alive_n;
    result.mask_hash = fnv1a64_bytes(net.mask, (size_t)N * (size_t)N * sizeof(*net.mask));
    result.perm_hash = fnv1a64_bytes(perm, (size_t)V * sizeof(*perm));

#if defined(VRAXION_REFERENCE)
    free(smask);
    free(sr);
    free(sc);
    free(ss);
#endif
    free(perm);
    net_free(&net);
    return result;
}

static void print_usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s [V] [N] [budget] [seed] [--state] [--mode=mixed|flip|add|remove]\n",
            argv0);
}

int main(int argc, char **argv) {
    int32_t V = 16;
    int32_t N = 48;
    int32_t budget = 2000;
    int32_t seed = 42;
    int32_t positional = 0;
    int show_state = 0;
    MutMode mode = MUT_MIXED;
    int i;
    clock_t t0;
    clock_t elapsed_ticks;
    TrainResult result;

    for (i = 1; i < argc; ++i) {
        const char *arg = argv[i];
        if (strcmp(arg, "--state") == 0) {
            show_state = 1;
            continue;
        }
        if (strncmp(arg, "--mode=", 7) == 0) {
            const char *value = arg + 7;
            if (strcmp(value, "mixed") == 0) {
                mode = MUT_MIXED;
            } else if (strcmp(value, "flip") == 0) {
                mode = MUT_FLIP_ONLY;
            } else if (strcmp(value, "add") == 0) {
                mode = MUT_ADD_ONLY;
            } else if (strcmp(value, "remove") == 0) {
                mode = MUT_REMOVE_ONLY;
            } else {
                fprintf(stderr, "invalid mode: %s\n", value);
                print_usage(argv[0]);
                return 2;
            }
            continue;
        }
        if (positional == 0) {
            if (!parse_i32(arg, &V)) {
                fprintf(stderr, "invalid V: %s\n", arg);
                print_usage(argv[0]);
                return 2;
            }
        } else if (positional == 1) {
            if (!parse_i32(arg, &N)) {
                fprintf(stderr, "invalid N: %s\n", arg);
                print_usage(argv[0]);
                return 2;
            }
        } else if (positional == 2) {
            if (!parse_i32(arg, &budget)) {
                fprintf(stderr, "invalid budget: %s\n", arg);
                print_usage(argv[0]);
                return 2;
            }
        } else if (positional == 3) {
            if (!parse_i32(arg, &seed)) {
                fprintf(stderr, "invalid seed: %s\n", arg);
                print_usage(argv[0]);
                return 2;
            }
        } else {
            fprintf(stderr, "unexpected argument: %s\n", arg);
            print_usage(argv[0]);
            return 2;
        }
        positional++;
    }

    if (V <= 0 || N < V || budget < 0) {
        fprintf(stderr, "invalid inputs: V=%" PRId32 " N=%" PRId32 " budget=%" PRId32 "\n", V, N, budget);
        return 2;
    }

    t0 = clock();
    result = train_one(V, N, seed, budget, mode);
    if (!result.ok) {
        return 1;
    }
    {
        elapsed_ticks = clock() - t0;
        uint64_t elapsed_ms = ((uint64_t)elapsed_ticks * UINT64_C(1000)) / (uint64_t)CLOCKS_PER_SEC;
        uint64_t elapsed_tenths = ((uint64_t)elapsed_ticks * UINT64_C(10)) / (uint64_t)CLOCKS_PER_SEC;
        uint64_t pct_tenths = ((uint64_t)result.best * UINT64_C(1000)) / (uint64_t)V;
        uint64_t ms_per_att_tenths = (budget > 0) ? (elapsed_ms * UINT64_C(10)) / (uint64_t)budget : 0U;
        printf("V=%" PRId32 " N=%" PRId32 " budget=%" PRId32 " seed=%" PRId32
               " best=%" PRId32 "/%" PRId32 " (%" PRIu64 ".%" PRIu64 "%%) time=%" PRIu64 ".%" PRIu64
               "s (%" PRIu64 ".%" PRIu64 "ms/att)\n",
               V, N, budget, seed, result.best, V,
               pct_tenths / UINT64_C(10), pct_tenths % UINT64_C(10),
               elapsed_tenths / UINT64_C(10), elapsed_tenths % UINT64_C(10),
               ms_per_att_tenths / UINT64_C(10), ms_per_att_tenths % UINT64_C(10));
        if (show_state != 0) {
            printf("state alive=%" PRId32 " mask_hash=%016" PRIx64 " perm_hash=%016" PRIx64 "\n",
                   result.alive_n, result.mask_hash, result.perm_hash);
        }
    }
    return 0;
}
