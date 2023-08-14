#ifndef __MATRIX_H__
#define __MATRIX_H__

#include <stdlib.h>
#include <stdio.h>

typedef struct Matrix {
    double *data, *free_ptr;
    size_t n, m, step, stride;
} Mat;

// Gives an entry point to specific data in the matrix.
#define MAT_AT(mat, i, j) ((mat).data[(i)*(mat).stride+(j)*(mat).step])

// Allocates a matrix on the stack.
#define MAT_ON_STACK(mat, N, M)\
    double _data_arr[(N)*(M)]; \
    (mat) = (Mat) {            \
        .data = _data_arr,     \
        .free_ptr = NULL,      \
        .n = (N),              \
        .m = 1,                \
        .step = 1,             \
    };                         \

void mat_assert(Mat m);
Mat mat_new(size_t n, size_t m);
Mat mat_rand_new(size_t n, size_t m);
Mat mat_fill(Mat m, double v);
Mat mat_row(Mat m, size_t i);
Mat mat_col(Mat m, size_t j);
Mat mat_sum(Mat a, Mat b);
double mat_add(Mat m);
Mat mat_scalar(Mat a, double v);
Mat mat_sub(Mat a, Mat b);
Mat mat_t(Mat m);
Mat mat_dot(Mat dst, Mat a, Mat b);
Mat mat_dot_sum(Mat dst, Mat a, Mat b);
Mat mat_mul(Mat a, Mat b);
Mat mat_copy(Mat a, Mat b);
Mat mat_func(Mat n, Mat m, double (*f)(double x));
Mat mat_from(FILE *f);
size_t mat_argmax(Mat m);
void mat_save(Mat m, FILE *f);
void mat_del(Mat m);

void mat_print_with_str(Mat m, const char *str, int pad);
void mat_print_no_nl(Mat m, const char *str);
void mat_print_from_layer(Mat m, size_t i);

void static fill_rand_data(double data[], size_t n);

// Formats and prints m.
#define mat_print(m) mat_print_with_str(m, #m":\n", 0)

#endif // __MATRIX_H__
