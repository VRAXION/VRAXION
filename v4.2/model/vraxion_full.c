/* VRAXION v4.2 — COMPLETE runtime, pure C, zero dependency.
   Compile: gcc -O2 -o vraxion vraxion_full.c
   ~110 lines of actual logic. Runs on ANYTHING. */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

#define MAX_N 768
#define MAX_ALIVE (MAX_N*MAX_N)
#define TICKS 8
#define THRESH 50    /* 0.5 × 100 */
#define RETAIN 85    /* 85/100 = 0.85 leak */
#define DRIVE 6      /* 6/10 = 0.6 */
#define CLIP 100     /* ±1.0 × 100 */

typedef struct {
    signed char mask[MAX_N * MAX_N];
    int charges[256 * MAX_N];
    int acts[256 * MAX_N];
    int alive_r[MAX_ALIVE], alive_c[MAX_ALIVE], alive_n;
    int N, V;
} Net;

void forward(Net *net) {
    int V=net->V, N=net->N;
    memset(net->charges, 0, V*N*sizeof(int));
    memset(net->acts, 0, V*N*sizeof(int));
    for (int t=0; t<TICKS; t++) {
        if (t==0) for (int i=0; i<V; i++) net->acts[i*N+i]=100;
        for (int i=0; i<V; i++)
            for (int j=0; j<N; j++) {
                int raw=0;
                for (int k=0; k<N; k++)
                    raw += net->acts[i*N+k] * net->mask[k*N+j];
                int ch = net->charges[i*N+j]*RETAIN/100 + raw*DRIVE/10;
                if (ch>CLIP*10) ch=CLIP*10; if (ch<-CLIP*10) ch=-CLIP*10;
                net->charges[i*N+j] = ch;
                net->acts[i*N+j] = ch>THRESH ? ch-THRESH : 0;
            }
    }
}

int eval(Net *net, int *perm) {
    int V=net->V, N=net->N, correct=0;
    for (int i=0; i<V; i++) {
        int best_j=0, best_v=net->charges[i*N+(N-V)];
        for (int j=1; j<V; j++) {
            int v=net->charges[i*N+(N-V)+j];
            if (v>best_v) { best_v=v; best_j=j; }
        }
        if (best_j==perm[i]) correct++;
    }
    return correct;
}

void do_flip(Net *n) {
    if (!n->alive_n) return;
    int i=rand()%n->alive_n;
    n->mask[n->alive_r[i]*n->N+n->alive_c[i]] *= -1;
}

void do_add(Net *n) {
    int r=rand()%n->N, c=rand()%n->N;
    if (r!=c && n->mask[r*n->N+c]==0) {
        n->mask[r*n->N+c] = (rand()%2)?1:-1;
        n->alive_r[n->alive_n]=r; n->alive_c[n->alive_n]=c; n->alive_n++;
    }
}

void do_remove(Net *n) {
    if (!n->alive_n) return;
    int i=rand()%n->alive_n;
    n->mask[n->alive_r[i]*n->N+n->alive_c[i]] = 0;
    n->alive_n--;
    n->alive_r[i]=n->alive_r[n->alive_n];
    n->alive_c[i]=n->alive_c[n->alive_n];
}

int train(int V, int N, int seed, int budget) {
    static Net net;
    memset(&net, 0, sizeof(Net));
    net.V=V; net.N=N; srand(seed);
    /* Sparse init */
    for (int i=0; i<N*N; i++) {
        if (i/N==i%N) continue;
        int r=rand()%100;
        if (r<4) { net.mask[i]=-1; net.alive_r[net.alive_n]=i/N; net.alive_c[net.alive_n]=i%N; net.alive_n++; }
        else if (r>96) { net.mask[i]=1; net.alive_r[net.alive_n]=i/N; net.alive_c[net.alive_n]=i%N; net.alive_n++; }
    }
    /* Random perm */
    int perm[256]; for(int i=0;i<V;i++) perm[i]=i;
    for(int i=V-1;i>0;i--) { int j=rand()%(i+1); int t=perm[i];perm[i]=perm[j];perm[j]=t; }

    size_t mask_bytes = (size_t)N * (size_t)N * sizeof(signed char);
    size_t alive_bytes = (size_t)N * (size_t)N * sizeof(int);
    signed char *smask = (signed char*)malloc(mask_bytes);
    int *sr = (int*)malloc(alive_bytes);
    int *sc = (int*)malloc(alive_bytes);
    int sn;
    if (!smask || !sr || !sc) {
        free(smask); free(sr); free(sc);
        fprintf(stderr, "alloc failed for N=%d\n", N);
        return -1;
    }

    forward(&net); int best=eval(&net,perm);

    for (int att=0; att<budget; att++) {
        memcpy(smask, net.mask, mask_bytes); sn=net.alive_n;
        memcpy(sr, net.alive_r, (size_t)sn * sizeof(int));
        memcpy(sc, net.alive_c, (size_t)sn * sizeof(int));

        int r=rand()%100;
        if (r<60) do_flip(&net);
        else if (r<85) do_add(&net);
        else do_remove(&net);

        forward(&net);
        int s=eval(&net,perm);
        if (s>best) best=s;
        else { memcpy(net.mask,smask,N*N); net.alive_n=sn;
               memcpy(net.alive_r, sr, (size_t)sn * sizeof(int));
               memcpy(net.alive_c, sc, (size_t)sn * sizeof(int)); }
    }
    free(smask); free(sr); free(sc);
    return best;
}

int main(int argc, char **argv) {
    int V=16, N=48, budget=2000, seed=42;
    if (argc>1) V=atoi(argv[1]);
    if (argc>2) N=atoi(argv[2]);
    if (argc>3) budget=atoi(argv[3]);
    if (argc>4) seed=atoi(argv[4]);
    clock_t t0=clock();
    int best=train(V,N,seed,budget);
    double elapsed=(double)(clock()-t0)/CLOCKS_PER_SEC;
    printf("V=%d N=%d budget=%d seed=%d best=%d/%d (%.1f%%) time=%.2fs (%.2fms/att)\n",
           V,N,budget,seed,best,V,100.0*best/V,elapsed,elapsed/budget*1000);
    return 0;
}
