#ifndef __LAYER_H__
#define __LAYER_H__

#include "matrix.h"
#include <string.h>

double relu(double x);
double sigmoid(double x);
double lineal(double x);

typedef double (*act_func_t)(double);
act_func_t funcs[4] = { relu, tanh, sigmoid, lineal };
enum ACT_FUNC { RELU, TANH, SIGMOID, LINEAL };

typedef struct Layer {
    Mat w, b, a, z;
    enum ACT_FUNC act_func;
    double (*act)(double);
} Layer;

// Asserts that every matrix
// in l is valid.
void lay_assert(Layer l) {
    mat_assert(l.w);
    mat_assert(l.b);
    mat_assert(l.z);
    mat_assert(l.a);
}

// Creates a new Layer for the nn.
Layer lay_new(size_t len, size_t input_size, enum ACT_FUNC act_func) {
    Layer l = (Layer) {
        .w = mat_rand_new(len, input_size),
        .b = mat_rand_new(len, 1),
        .z = mat_new(len, 1),
        .a = mat_new(len, 1),
        .act_func = act_func,
        .act = funcs[act_func],
    };

    lay_assert(l);
    return l;
}

// Calculates the sum of the product of weights
// applying the activation function.
Mat lay_forward(Layer l, Mat x) {
    mat_sum(mat_dot(l.z, l.w, x), l.b);
    return mat_func(l.a, l.z, l.act);
}

Mat lay_weights(Layer l) {
    return l.w;
}

Mat lay_biases(Layer l) {
    return l.b;
}

// Prints the matrices of l.
void lay_print(Layer l, size_t i) {
    int pad = 4;
    char wbuff[16];
    char bbuff[16];
    char abuff[16];
    snprintf(wbuff, sizeof(wbuff), "W%li", i);
    snprintf(bbuff, sizeof(bbuff), "B%li", i);
    snprintf(abuff, sizeof(abuff), i == 0 ? "X" : "A%li", i);

    printf("%*s%s:%*s", pad, "", wbuff, (int)l.w.m*7+2, "");
    printf("%s:%*s", abuff, 7 - (i == 0 ? 0 : 1) , "");
    printf("%s:\n", bbuff);
    for (size_t j = 0; j < l.w.m; j++) {
        printf("%*s", pad, "");
        mat_print_from_layer(l.w, j);
        printf(BLACK" ["WHITE"  %s%li  "BLACK"] "WHITE, i == 0 ? "x" : "a", j);
        mat_print_from_layer(l.b, j);
        puts("");
    }

    puts("");
}

Layer lay_new_zero(Layer l) {
    return (Layer) {
        .w = mat_new(l.w.n, l.w.m),
        .b = mat_new(l.b.n, l.b.m),
        .z = mat_new(l.z.n, l.z.m),
        .a = mat_new(l.a.n, l.a.m),
        .act_func = l.act_func,
        .act = l.act,
    };
}

void lay_fill_zeros(Layer l) {
    mat_fill(l.w, 0);
    mat_fill(l.b, 0);
    mat_fill(l.z, 0);
    mat_fill(l.a, 0);
}

// Saves the layer to a file.
void lay_save(Layer l, FILE *f) {
    fwrite(&l.act_func, sizeof(l.act_func), 1, f);
    mat_save(l.w, f);
    mat_save(l.b, f);
}

// Creates a layer from a file.
Layer lay_from(FILE *f) {
    enum ACT_FUNC act;
    fread(&act, sizeof(act), 1, f);
    Layer l = (Layer) {
        .w = mat_from(f),
        .b = mat_from(f),
        .z = mat_new(l.b.n, 1),
        .a = mat_new(l.b.n, 1),
        .act_func = act,
        .act = act == -1 ? NULL : funcs[act],
    };

    return l;
}

// Frees the memory used by l.
void lay_del(Layer l) {
    mat_del(l.w);
    mat_del(l.b);
    mat_del(l.z);
    mat_del(l.a);
}

#endif // __LAYER_H__
