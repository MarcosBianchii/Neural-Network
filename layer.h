#ifndef __LAYER_H__
#define __LAYER_H__

#include "matrix.h"
#include <string.h>

double relu(double x);
double sigmoid(double x);

double (*funcs[3])(double) = {relu, tanh, sigmoid};
typedef enum ACT_FUNC { RELU, TANH, SIGMOID } ACT_FUNC;

typedef struct Layer {
    Mat w, b, a;
    ACT_FUNC act_func;
    double (*act)(double);
} Layer;

// Asserts that every matrix
// in l is valid.
void lay_assert(Layer l) {
    mat_assert(l.a);
    mat_assert(l.w);
    mat_assert(l.b);
}

// Creates a new Layer for the nn.
Layer lay_new(size_t len, size_t input_size, ACT_FUNC act_func) {
    Layer l = (Layer) {
        .a = mat_new(1, len),
        .w = mat_rand_new(input_size, len),
        .b = mat_rand_new(1, len),
        .act_func = act_func,
        .act = act_func == -1 ? NULL : funcs[act_func],
    };

    lay_assert(l);
    return l;
}

// Calculates the sum of the product of weights
// applying the activation function.
Mat lay_forward(Layer l, Mat x) {
    return mat_func(mat_sum(mat_dot(l.a, x, l.w), l.b), l.act);
}

Mat lay_weights(Layer l) {
    return l.w;
}

Mat lay_biases(Layer l) {
    return l.b;
}

// Prints the matrices of l.
void lay_print(Layer l, size_t i) {
    char buff[16];
    snprintf(buff, sizeof(buff), "W%li", i);
    mat_print_with_str(l.w, buff, 4);
    snprintf(buff, sizeof(buff), "B%li", i);
    mat_print_with_str(l.b, buff, 4);
}

// Saves the layer to a file.
void lay_save(Layer l, FILE *f) {
    fwrite(&l.act_func, sizeof(l.act_func), 1, f);
    mat_save(l.w, f);
    mat_save(l.b, f);
}

Layer lay_from_file(FILE *f) {
    ACT_FUNC act = -1;
    fread(&act, sizeof(act), 1, f);
    Layer l = (Layer) {
        .w = mat_from_file(f),
        .b = mat_from_file(f),
        .a = mat_new(1, l.b.m),
        .act_func = act,
        .act = act == -1 ? NULL : funcs[act],
    };

    return l;
}

// Frees the memory used by l.
void lay_del(Layer l) {
    mat_del(l.w);
    mat_del(l.b);
    mat_del(l.a);
}

#endif // __LAYER_H__