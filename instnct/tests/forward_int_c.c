/* Minimal int-only forward pass — zero float, zero dependency.
   Compile: gcc -O2 -shared -o forward_int_c.dll forward_int_c.c (Windows)
            gcc -O2 -shared -fPIC -o forward_int_c.so forward_int_c.c (Linux) */

#include <string.h>

void forward_batch_int(
    const signed char *mask,  /* N×N, values {-6, 0, +6} */
    int *charges,             /* V×N output, zeroed on entry */
    int *acts,                /* V×N scratch, zeroed on entry */
    int V, int N, int ticks,
    int retain,               /* 85 = 0.85×100 */
    int threshold,            /* 5  = 0.5×10 */
    int clip                  /* 100 = 1.0×100 */
) {
    for (int t = 0; t < ticks; t++) {
        /* Input: identity × 10 on first tick */
        if (t == 0) {
            for (int i = 0; i < V; i++)
                acts[i * N + i] = 10;
        }
        /* Matmul: acts @ mask → raw, accumulate into charges */
        for (int i = 0; i < V; i++) {
            for (int j = 0; j < N; j++) {
                int raw = 0;
                for (int k = 0; k < N; k++)
                    raw += acts[i * N + k] * (int)mask[k * N + j];
                charges[i * N + j] += raw;
            }
        }
        /* Leak + threshold + clip */
        for (int i = 0; i < V * N; i++) {
            charges[i] = charges[i] * retain / 100;
            acts[i] = charges[i] > threshold ? charges[i] - threshold : 0;
            if (charges[i] > clip) charges[i] = clip;
            if (charges[i] < -clip) charges[i] = -clip;
        }
    }
}
