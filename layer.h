#ifndef __LAYER_H__
#define __LAYER_H__

#include "matrix.h"
#include <assert.h>
#include <string.h>

typedef struct Layer {
    Mat w, b, a;
    double (*act_func)(double);
    size_t len;
} Layer;

// Asserts that every matrix
// in l is valid.
void lay_assert(Layer l) {
    mat_assert(l.a);
    mat_assert(l.w);
    mat_assert(l.b);
}

// Creates a new Layer for the nn.
Layer lay_new(size_t len, size_t input_size, double (*act_func)(double)) {
    Layer l = (Layer) {
        .a = mat_new(1, len),
        .w = mat_rand_new(input_size, len),
        .b = mat_rand_new(1, len),
        .act_func = act_func,
        .len = len,
    };

    lay_assert(l);
    return l;
}

// Calculates the sum of the product of weights
// applying the activation function.
Mat lay_forward(Layer l, Mat x) {
    mat_fill(l.a, 0);
    return mat_func(mat_sum(mat_dot(l.a, x, l.w), l.b), l.act_func);
}

// Prints the matrices of l.
void lay_print(Layer l) {
    Mat W = l.w;
    Mat B = l.b;
    mat_print(W);
    mat_print(B);
}

// Frees the memory used by l.
void lay_del(Layer l) {
    mat_del(l.w);
    mat_del(l.b);
}

#endif // __LAYER_H__