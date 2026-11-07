// matmul.c: one matrix multiply, three loop structures (Chapter 8).
// Same arithmetic every time; only the order memory is touched changes.
// build: clang -O3 -ffast-math -o matmul matmul.c && ./matmul      (Linux: gcc)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define N 1024
#define TILE 64
static float A[N * N], B[N * N], C[N * N], REF[N * N];

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static void ijk(void) {          // textbook order: inner loop strides down a column of B (cache-hostile)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float s = 0;
            for (int k = 0; k < N; k++) s += A[i * N + k] * B[k * N + j];
            C[i * N + j] = s;
        }
}

static void ikj(void) {          // swap two loops: inner loop walks rows of B and C in order (vectorizes)
    memset(C, 0, sizeof C);
    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++) {
            float a = A[i * N + k];
            for (int j = 0; j < N; j++) C[i * N + j] += a * B[k * N + j];
        }
}

static void tiled(void) {        // work on TILE x TILE blocks that stay in cache while they are reused
    memset(C, 0, sizeof C);
    for (int ii = 0; ii < N; ii += TILE)
        for (int kk = 0; kk < N; kk += TILE)
            for (int jj = 0; jj < N; jj += TILE)
                for (int i = ii; i < ii + TILE; i++)
                    for (int k = kk; k < kk + TILE; k++) {
                        float a = A[i * N + k];
                        for (int j = jj; j < jj + TILE; j++) C[i * N + j] += a * B[k * N + j];
                    }
}

static void run(const char *name, void (*f)(void), int check) {
    double t = now();
    f();
    double s = now() - t, err = 0;
    if (check) for (int i = 0; i < N * N; i++) err = fmax(err, fabs(C[i] - REF[i]));
    printf("%-6s %8.2f GFLOPS  %6.3fs", name, 2.0 * N * N * N / s / 1e9, s);
    if (check) printf("  max diff vs ijk %.1e", err);
    printf("\n");
}

int main(void) {
    srand(0);
    for (int i = 0; i < N * N; i++) { A[i] = rand() / (float)RAND_MAX; B[i] = rand() / (float)RAND_MAX; }
    run("ijk", ijk, 0);
    memcpy(REF, C, sizeof C);
    run("ikj", ikj, 1);
    run("tiled", tiled, 1);
    return 0;
}
