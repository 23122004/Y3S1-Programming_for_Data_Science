#include <stdio.h>
#include <stdlib.h>
#include <time.h>

double** allocate_matrix(int n) {
    double** matrix = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        matrix[i] = (double*)malloc(n * sizeof(double));
    }
    return matrix;
}

void free_matrix(double** matrix, int n) {
    for (int i = 0; i < n; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

void initialize_matrix(double** matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i][j] = (double)rand() / RAND_MAX;
        }
    }
}

void multiply_matrices(double** a, double** b, double** c, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            c[i][j] = 0;
            for (int k = 0; k < n; k++) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <matrix_size>\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    if (n <= 0) {
        fprintf(stderr, "Matrix size must be a positive integer.\n");
        return 1;
    }

    srand(time(NULL));

    double** A = allocate_matrix(n);
    double** B = allocate_matrix(n);
    double** C = allocate_matrix(n);

    initialize_matrix(A, n);
    initialize_matrix(B, n);

    clock_t start = clock();
    multiply_matrices(A, B, C, n);
    clock_t end = clock();

    double time_spent_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    printf("%.2f", time_spent_ms);

    free_matrix(A, n);
    free_matrix(B, n);
    free_matrix(C, n);

    return 0;
}