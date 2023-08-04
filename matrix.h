#ifndef __MATRIX_H__
#define __MATRIX_H__

#include <assert.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

typedef struct Matrix {
    double *data, *free_ptr;
    size_t n, m, stride;
} Mat;

// Gives an entry point to specific data in the matrix.
#define MAT_AT(mat, i, j) (mat).data[(i)*(mat).stride+(j)]

void mat_print_no_nl(Mat m, const char *str) {
    printf("\033[0;37m%s: ", str);
    char buff[16];
    for (size_t i = 0; i < m.n; i++) {
        printf("\033[0;30m[\033[0;37m  ");
        for (size_t j = 0; j < m.m; j++) {
            double v = MAT_AT(m, i, j);
            snprintf(buff, 6, "%.3lf", v);
            printf(v < 0 ? "%.2lf  " : "%.3lf  ", v);
        }
        printf("\033[0;30m]");
    }
    printf("\033[0;37m");
}

double absf(double v);
static void mat_print_with_str(Mat m, const char *str, int pad) {
    printf("\033[0;37m%*s%s:\n", pad, "", str);
    char buff[16];
    for (size_t i = 0; i < m.n; i++) {
        printf("\033[0;30m%*s[  ", pad, "");
        for (size_t j = 0; j < m.m; j++) {
            double v = MAT_AT(m, i, j);
            snprintf(buff, 6, "%.3lf", absf(v));
            printf(v < 0 ? "\033[0;31m%s  " : "\033[0;32m%s  ", buff);
        }
        puts("\033[0;30m]");
    }
    puts("\033[0;37m");
}

// Formats and prints m.
#define mat_print(m) mat_print_with_str(m, #m, 0)

// Asserts m is a valid matrix.
void mat_assert(Mat m) {
    assert(m.data != NULL);
}

// Returns an empty matrix.
Mat mat_new(size_t n, size_t m) {
    Mat r = {
        .n = n,
        .m = m,
        .stride = m,
        .data = calloc(n*m, sizeof(double)),
        .free_ptr = r.data,
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
void mat_fill(Mat m, double v) {
    size_t prod = m.n * m.m;
    for (size_t i = 0; i < prod; i++)
        m.data[i] = v;
}

// Returns a matrix full of random entries.
Mat mat_rand_new(size_t n, size_t m) {
    Mat r = mat_new(n, m);
    fill_rand_data(r.data, n*m);
    return r;
}

// Returns a matrix from a C style matrix.
Mat mat_from(size_t n, size_t m, double x[n][m]) {
    Mat r = mat_new(n, m);
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < m; j++)
            MAT_AT(r, i, j) = x[i][j];
    return r;
}

// Performs the sum between matrices a and b.
// The result is then stored in a and returned.
Mat mat_sum(Mat a, Mat b) {
    assert(a.n == b.n);
    assert(a.m == b.m);
    size_t prod = a.n * a.m;
    for (size_t i = 0; i < prod; i++)
        a.data[i] += b.data[i];
    return a;
}

// Performs the subtraction between matrices a and b.
// The result is then stored in a and returned.
Mat mat_sub(Mat a, Mat b) {
    assert(a.n == b.n);
    assert(a.m == b.m);
    size_t prod = a.n * a.m;
    for (size_t i = 0; i < prod; i++)
        a.data[i] -= b.data[i];
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

// Returns a sub-matrix with the i'th row of entries.
// The returned Mat doesn't need to be free'd using mat_del().
Mat mat_row(Mat m, size_t i) {
    return (Mat) {
        .data = &m.data[i*m.stride],
        .free_ptr = NULL,
        .n = 1,
        .m = m.m,
        .stride = m.stride,
    };
}

// Retures a sub-matrix with the j'th col of entries.
// The returned Mat doesn't need to be free'd using mat_del().
Mat mat_col(Mat m, size_t j) {
    return (Mat) {
        .data = &m.data[j],
        .free_ptr = NULL,
        .n = m.n,
        .m = 1,
        .stride = m.m,
    };
}

// Returns a sub-matrix of m of the first j columns.
// The returned Mat doesn't need to be free'd using mat_del().
Mat mat_upto_col(Mat m, size_t j) {
    return (Mat) {
        .data = m.data,
        .free_ptr = NULL,
        .n = m.n,
        .m = j,
        .stride = m.m,
    };
}

// Returns a sub-matrix of m from the j'th column to the end.
// The returned Mat doesn't need to be free'd using mat_del().
Mat mat_from_col(Mat m, size_t j) {
    return (Mat) {
        .data = &m.data[j],
        .free_ptr = NULL,
        .n = m.n,
        .m = m.m - j,
        .stride = m.m,
    };
}

// Applies f to every entry in m.
Mat mat_func(Mat m, double (*f)(double x)) {
    if (!f) return m;
    size_t n = m.n * m.m;
    for (size_t i = 0; i < n; i++)
        m.data[i] = f(m.data[i]);
    return m;
}

// Saves m to a file.
void mat_save(Mat m, FILE *f) {
    fwrite(&m.n, sizeof(size_t), 1, f);
    fwrite(&m.m, sizeof(size_t), 1, f);
    fwrite(m.data, sizeof(double), m.n*m.m, f);
}

// Loads a matrix from a file.
Mat mat_from_file(FILE *f) {
    size_t n, m;
    fread(&n, sizeof(size_t), 1, f);
    fread(&m, sizeof(size_t), 1, f);
    Mat r = mat_new(n, m);
    fread(r.data, sizeof(double), n*m, f);
    return r;
}

// Frees the memory used by m.
void mat_del(Mat m) {
    free(m.free_ptr);
}

#endif // __MATRIX_H__