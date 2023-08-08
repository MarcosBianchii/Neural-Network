#ifndef __LAYER_H__
#define __LAYER_H__

#include "matrix.h"
#include <math.h>

double sigmoid(double x);
double relu(double x);
double lineal(double x);

typedef double (*act_func_t)(double);
enum ACT_FUNC { RELU, TANH, SIGMOID, LINEAL };

typedef struct Layer {
    Mat w, b, a, z;
    enum ACT_FUNC act_func;
    act_func_t act;
} Layer;

void lay_assert(Layer l);
Layer lay_new(size_t len, size_t input_size, enum ACT_FUNC act_func);
Layer lay_new_zero(Layer l);
Mat lay_forward(Layer l, Mat x);
void lay_print(Layer l, size_t i, size_t prev_size);
void lay_fill_zeros(Layer l);
void lay_save(Layer l, FILE *f);
Layer lay_from(FILE *f);
void lay_del(Layer l);

#endif // __LAYER_H__
