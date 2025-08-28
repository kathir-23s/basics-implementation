// matmul.c - C = A(m×k) * B(k×n)    (row-major, doubles)
// Build: gcc -O3 -march=native -fPIC -shared -o libfastmatmul.so matmul.c
// (Optional) add -fopenmp and uncomment pragmas for multithreading

#include <stddef.h>

void matmul_naive(
    const double* A, const double* B, double* C,
    int m, int k, int n
) {
    // zero C
    for (int i = 0; i < m*n; ++i) C[i] = 0.0;

    // i: 0..m-1, j: 0..n-1, p: 0..k-1
    #pragma omp parallel for collapse(2)  // <- enable if compiled with -fopenmp
    for (int i = 0; i < m; ++i) {
        for (int p = 0; p < k; ++p) {
            double a_ip = A[i*(size_t)k + p];
            const double* Bp = &B[p*(size_t)n];
            double* Ci = &C[i*(size_t)n];
            for (int j = 0; j < n; ++j) {
                Ci[j] += a_ip * Bp[j];
            }
        }
    }
}

// Cache-blocked version (often faster for large matrices)
void matmul_blocked(
    const double* A, const double* B, double* C,
    int m, int k, int n, int BS
) {
    for (int i = 0; i < m*n; ++i) C[i] = 0.0;

    #pragma omp parallel for collapse(2) schedule(static)  // with -fopenmp
    for (int ii = 0; ii < m; ii += BS) {
        for (int jj = 0; jj < n; jj += BS) {
            for (int pp = 0; pp < k; pp += BS) {
                int iimax = (ii + BS < m) ? (ii + BS) : m;
                int jjmax = (jj + BS < n) ? (jj + BS) : n;
                int ppmax = (pp + BS < k) ? (pp + BS) : k;

                for (int i = ii; i < iimax; ++i) {
                    for (int p = pp; p < ppmax; ++p) {
                        double a_ip = A[i*(size_t)k + p];
                        const double* Bp = &B[p*(size_t)n + jj];
                        double* Ci = &C[i*(size_t)n + jj];
                        for (int j = jj; j < jjmax; ++j) {
                            Ci[j - jj] += a_ip * Bp[j - jj];
                        }
                    }
                }
            }
        }
    }
}

// simple exported front-door: pick blocked by default with BS=64
// (You can tune BS = 32/64/128 depending on your CPU)
// Exposed to Python via ctypes.
void matmul(
    const double* A, const double* B, double* C,
    int m, int k, int n
) {
    // For small sizes naive can be fine; for bigger, blocked is better.
    // Heuristic switch:
    if ((long long)m * n * k < 1LL * 128 * 128 * 128) {
        matmul_naive(A, B, C, m, k, n);
    } else {
        matmul_blocked(A, B, C, m, k, n, 64);
    }
}
