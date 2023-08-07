#ifndef __MATRIX_H__
#define __MATRIX_H__

#include <assert.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "colors.h"

typedef struct Matrix {
    double *data, *free_ptr;
    size_t n, m, step, stride;
} Mat;

// Gives an entry point to specific data in the matrix.
#define MAT_AT(mat, i, j) ((mat).data[(i)*(mat).stride+(j)*(mat).step])

// Asserts m is a valid matrix.
void mat_assert(Mat m) {
    assert(m.data != NULL);
}

// Returns a sub-matrix with the i'th row of entries.
// The returned Mat doesn't need to be free'd using mat_del().
Mat mat_row(Mat m, size_t i) {
    return (Mat) {
        .data = &MAT_AT(m, i, 0),
        .free_ptr = NULL,
        .n = 1,
        .m = m.m,
        .step = m.step,
        .stride = 0,
    };
}

// Retures a sub-matrix with the j'th col of entries.
// The returned Mat doesn't need to be free'd using mat_del().
Mat mat_col(Mat m, size_t j) {
    return (Mat) {
        .data = &MAT_AT(m, 0, j),
        .free_ptr = NULL,
        .n = m.n,
        .m = 1,
        .step = 0,
        .stride = m.stride,
    };
}

double absf(double v);
static void mat_print_with_str(Mat m, const char *str, int pad) {
    printf(WHITE"%*s%s", pad, "", str);
    char buff[16];
    for (size_t i = 0; i < m.n; i++) {
        printf(BLACK"%*s[  ", pad, "");
        for (size_t j = 0; j < m.m; j++) {
            double v = MAT_AT(m, i, j);
            snprintf(buff, 6, "%.3lf", absf(v));
            printf(v < 0 ? RED"%s  " : (v == 0 ? WHITE"%s  " : GREEN"%s  "), buff);
        }
        puts(BLACK"]");
    }
    puts(WHITE);
}

void mat_print_no_nl(Mat m, const char *str) {
    printf(WHITE"%s", str);
    char buff[16];
    for (size_t i = 0; i < m.n; i++) {
        printf(BLACK"[  ");
        for (size_t j = 0; j < m.m; j++) {
            double v = MAT_AT(m, i, j);
            snprintf(buff, 6, "%.3lf", absf(v));
            printf(v < 0 ? RED"%s  " : (v == 0 ? WHITE"%s  " : GREEN"%s  "), buff);
        }
        printf(BLACK"]");
    }
    printf(WHITE);
}

// Formats and prints m.
#define mat_print(m) mat_print_with_str(m, #m":\n", 0)

void mat_print_from_layer(Mat m, size_t i) {
    if (m.n <= i) {
        printf("%*s", (int)m.m*7+4, "");
        return;
    }

    Mat row = mat_row(m, i);
    mat_print_no_nl(row, "");
}

// Returns an empty matrix.
Mat mat_new(size_t n, size_t m) {
    Mat r = {
        .data = calloc(n*m, sizeof(double)),
        .free_ptr = r.data,
        .n = n,
        .m = m,
        .step = 1,
        .stride = m,
    };

    mat_assert(r);
    return r;
}

// Generates a random value between [-1,1].
double randf() {
    return (double)rand() / (double)RAND_MAX * 2 - 1;
}

// Fills a data array with random values.
void fill_rand_data(double data[], size_t n) {
    for (size_t i = 0; i < n; i++)
        data[i] = randf();
}

// Fills a matrix with v.
Mat mat_fill(Mat m, double v) {
    for (size_t i = 0; i < m.n; i++)
        for (size_t j = 0; j < m.m; j++)
            MAT_AT(m, i, j) = v;
    return m;
}

// Returns a matrix full of random entries.
Mat mat_rand_new(size_t n, size_t m) {
    Mat r = mat_new(n, m);
    fill_rand_data(r.data, n*m);
    return r;
}

// Performs the sum between matrices a and b.
// The result is then stored in a and returned.
Mat mat_sum(Mat a, Mat b) {
    assert(a.n == b.n);
    assert(a.m == b.m);
    for (size_t i = 0; i < a.n; i++)
        for (size_t j = 0; j < a.m; j++)
            MAT_AT(a, i, j) += MAT_AT(b, i, j);
    return a;
}

// Adds every element of m and returns it's sum.
double mat_add(Mat m) {
    double sum = 0;
    for (size_t i = 0; i < m.n; i++)
        for (size_t j = 0; j < m.m; j++)
            sum += MAT_AT(m, i, j);
    return sum;
}

// Performs the product between matrix a and scalar v.
Mat mat_scalar(Mat a, double v) {
    for (size_t i = 0; i < a.n; i++)
        for (size_t j = 0; j < a.m; j++)
            MAT_AT(a, i, j) *= v;
    return a;
}

// Performs the substraction between matrices a and b.
// The result is then stored in a and returned.
Mat mat_sub(Mat a, Mat b) {
    assert(a.n == b.n);
    assert(a.m == b.m);
    for (size_t i = 0; i < a.n; i++)
        for (size_t j = 0; j < a.m; j++)
            MAT_AT(a, i, j) -= MAT_AT(b, i, j);
    return a;
}

// Performs the product between matrices a and b.
// The result is then stored in dst and returned.
Mat mat_dot(Mat dst, Mat a, Mat b) {
    assert(a.m == b.n);
    assert(dst.n == a.n);
    assert(dst.m == b.m);
    mat_fill(dst, 0);
    size_t n = a.m;
    for (size_t i = 0; i < a.n; i++)
        for (size_t j = 0; j < b.m; j++)
            for (size_t k = 0; k < n; k++)
                MAT_AT(dst, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
    return dst;
}

// Performs the product between matrices a and b.
// The result is summed to dst and returned.
Mat mat_dot_sum(Mat dst, Mat a, Mat b) {
    assert(a.m == b.n);
    assert(dst.n == a.n);
    assert(dst.m == b.m);
    size_t n = a.m;
    for (size_t i = 0; i < a.n; i++)
        for (size_t j = 0; j < b.m; j++)
            for (size_t k = 0; k < n; k++)
                MAT_AT(dst, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
    return dst;
}

// Performs the Hadamard product between a and b.
// The result is then stored in a and returned.
Mat mat_mul(Mat a, Mat b) {
    assert(a.n == b.n);
    assert(a.m == b.m);
    for (size_t i = 0; i < a.n; i++)
        for (size_t j = 0; j < a.m; j++)
            MAT_AT(a, i, j) *= MAT_AT(b, i, j);
    return a;
}

// Returns a transposed matrix and returns it.
Mat mat_t(Mat x) {
    return (Mat) {
        .data = &MAT_AT(x, 0, 0),
        .free_ptr = NULL,
        .n = x.m,
        .m = x.n,
        .step = x.stride,
        .stride = x.step,
    };
}

// Copies matrix b to a and returns it.
Mat mat_copy(Mat a, Mat b) {
    assert(a.n == b.n);
    assert(a.m == b.m);
    for (size_t i = 0; i < a.n; i++)
        for (size_t j = 0; j < a.m; j++)
            MAT_AT(a, i, j) = MAT_AT(b, i, j);
    return a;
}

// Applies f to m and stores it's result in n returning it.
Mat mat_func(Mat n, Mat m, double (*f)(double x)) {
    if (!f) return mat_copy(n, m);

    assert(m.n == n.n);
    assert(m.m == n.m);
    for (size_t i = 0; i < n.n; i++)
        for (size_t j = 0; j < n.m; j++)
            MAT_AT(n, i, j) = f(MAT_AT(m, i, j));
    return n;
}

// Saves m to a file.
void mat_save(Mat m, FILE *f) {
    fwrite(&m.n, sizeof(m.n), 1, f);
    fwrite(&m.m, sizeof(m.m), 1, f);
    fwrite(m.data, sizeof(double), m.n*m.m, f);
}

// Loads a matrix from a file.
Mat mat_from(FILE *f) {
    size_t n, m;
    fread(&n, sizeof(n), 1, f);
    fread(&m, sizeof(m), 1, f);
    Mat r = mat_new(n, m);
    fread(r.data, sizeof(double), n*m, f);
    return r;
}

// Frees the memory used by m.
void mat_del(Mat m) {
    free(m.free_ptr);
}

#endif // __MATRIX_H__
