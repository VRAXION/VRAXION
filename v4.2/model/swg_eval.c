/*
 * SWG Eval v2 — C accelerator with precomputed matmuls
 * =====================================================
 * Python precomputes bp_W_in (256×N) and W_out_T (IO×N).
 * C does only the sequential byte loop + sparse scatter.
 *
 * Compile: x86_64-w64-mingw32-gcc -O3 -shared -o swg_eval.dll swg_eval.c -lm
 */
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

EXPORT float eval_seqs(
    /* Sparse mask */
    const int *rows, const int *cols, const float *vals, int n_edges,
    /* Per-neuron params */
    const float *theta, const float *decay,
    /* Precomputed: bp_W_in = bp @ W_in, shape 256 × N, row-major */
    const float *bp_W_in,
    /* Precomputed: W_out transposed, shape IO × N, row-major */
    const float *W_out_T,
    /* Precomputed: bp_norm transposed, shape IO × 256, row-major */
    const float *bp_norm_T,
    /* Sequences */
    const uint8_t *texts, int n_seqs, int seq_len,
    int N, int IO
) {
    float *state  = (float *)calloc(N, sizeof(float));
    float *charge = (float *)calloc(N, sizeof(float));
    float *act    = (float *)malloc(N * sizeof(float));
    float *raw    = (float *)malloc(N * sizeof(float));
    float *out    = (float *)malloc(IO * sizeof(float));
    float *ret    = (float *)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) ret[i] = 1.0f - decay[i];

    int total_correct = 0;
    float total_prob = 0.0f;
    int total_n = 0;

    for (int s = 0; s < n_seqs; s++) {
        const uint8_t *text = texts + s * seq_len;
        memset(state, 0, N * sizeof(float));
        memset(charge, 0, N * sizeof(float));

        for (int pos = 0; pos < seq_len - 1; pos++) {
            memcpy(act, state, N * sizeof(float));

            for (int t = 0; t < 6; t++) {
                if (t == 0) {
                    /* act += bp_W_in[text[pos]] — just vector add, no matmul */
                    const float *row = bp_W_in + text[pos] * N;
                    for (int j = 0; j < N; j++)
                        act[j] += row[j];
                }

                /* Sparse: raw[c] += act[r] * val */
                memset(raw, 0, N * sizeof(float));
                for (int e = 0; e < n_edges; e++)
                    raw[cols[e]] += act[rows[e]] * vals[e];

                for (int i = 0; i < N; i++) {
                    charge[i] = (charge[i] + raw[i]) * ret[i];
                    act[i] = charge[i] > theta[i] ? charge[i] - theta[i] : 0.0f;
                    if (charge[i] > 1.0f) charge[i] = 1.0f;
                    else if (charge[i] < -1.0f) charge[i] = -1.0f;
                }
            }

            memcpy(state, act, N * sizeof(float));

            /* out = charge @ W_out — using W_out_T for cache-friendly access */
            for (int j = 0; j < IO; j++) {
                const float *wrow = W_out_T + j * N;
                float sum = 0;
                for (int i = 0; i < N; i++)
                    sum += wrow[i] * charge[i];
                out[j] = sum;
            }

            /* normalize out */
            float norm = 0;
            for (int j = 0; j < IO; j++) norm += out[j] * out[j];
            norm = sqrtf(norm) + 1e-8f;
            for (int j = 0; j < IO; j++) out[j] /= norm;

            /* sims = bp_norm @ out — using bp_norm_T (IO × 256) */
            /* sims[j] = sum_k out[k] * bp_norm_T[k * 256 + j] */
            float mx = -1e30f;
            int pred = 0;
            float sims[256];
            for (int j = 0; j < 256; j++) sims[j] = 0;
            for (int k = 0; k < IO; k++) {
                const float *brow = bp_norm_T + k * 256;
                float ok = out[k];
                for (int j = 0; j < 256; j++)
                    sims[j] += ok * brow[j];
            }
            for (int j = 0; j < 256; j++) {
                if (sims[j] > mx) { mx = sims[j]; pred = j; }
            }

            uint8_t target = text[pos + 1];
            if (pred == (int)target) total_correct++;

            float esum = 0;
            for (int j = 0; j < 256; j++) {
                sims[j] = expf(sims[j] - mx);
                esum += sims[j];
            }
            total_prob += sims[target] / esum;
            total_n++;
        }
    }

    free(state); free(charge); free(act); free(raw);
    free(out); free(ret);

    float acc = total_n > 0 ? (float)total_correct / total_n : 0;
    float avg_p = total_n > 0 ? total_prob / total_n : 0;
    return 0.5f * acc + 0.5f * avg_p;
}
