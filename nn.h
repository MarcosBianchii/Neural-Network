#ifndef __NN_H__
#define __NN_H__

#include <math.h>
#include "layer.h"

double EPS = 10e-5;
double LEARNING_RATE = 10e-1;
size_t MAX_ITER = 10e+5;

double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

double relu(double x) {
    return x > 0 ? x : 0;
}

typedef struct NeuralNetwork {
    size_t xs, len;
    Layer *l;
} NN;

// arch: an array of [params, [hidden layers], outputs]
NN nn_new(size_t arch[], size_t len) {
    NN n = (NN) {
        .l = malloc(sizeof(*n.l) * (len-1)),
        .xs = arch[0],
        .len = len-1,
    };

    assert(n.l != NULL);
    size_t input_size = arch[0];
    for (size_t i = 0; i < len-1; i++) {
        n.l[i] = lay_new(arch[i+1], input_size, sigmoid);
        lay_assert(n.l[i]);
        input_size = arch[i+1];
    }

    return n;
}

static Mat forward_rec(Layer *l, Mat x, size_t n, size_t i) {
    if (i == n) return x;
    return forward_rec(l, lay_forward(l[i], x), n, i+1);
}

// Tries to predict y given x.
Mat forward(NN n, Mat x) {
    return forward_rec(n.l, x, n.len, 0);
}

// Calculates the cost for the current state
// of the neural network given two sets of data.
double cost(NN n, Mat x, Mat y) {
    double sum = 0;
    size_t data_len = x.n;
    for (int i = 0; i < data_len; i++) {
        Mat x_row = mat_row(x, i);
        Mat y_row = mat_row(y, i);
        double pred = forward(n, x_row).data[0];
        double diff = y_row.data[0] - pred;
        sum += diff * diff;
    }

    return sum / data_len;
}

static void diff_between_matrices(Mat m, Mat g) {
    for (size_t i = 0; i < m.n; i++)
        for (size_t j = 0; j < m.m; j++)
            MAT_AT(m, i, j) -= MAT_AT(g, i, j) * LEARNING_RATE;
}

static void learn(Layer l, Mat gw, Mat gb) {
    diff_between_matrices(l.w, gw);
    diff_between_matrices(l.b, gb);
}

static Mat diff_in_matrix(NN n, Mat m, Mat x, Mat y) {
    Mat g = mat_new(m.n, m.m);
    double c = cost(n, x, y);
    for (size_t i = 0; i < m.n; i++)
        for (size_t j = 0; j < m.m; j++) {
            double prev_value = MAT_AT(m, i, j);
            MAT_AT(m, i, j) += EPS;
            MAT_AT(g, i, j) = (cost(n, x, y) - c) / EPS;
            MAT_AT(m, i, j) = prev_value;
        }
    return g;
}

static void finite_diff(NN n, Mat x, Mat y) {
    for (size_t k = 0; k < n.len; k++) {
        Mat w = n.l[k].w;
        Mat b = n.l[k].b;
        Mat gw = diff_in_matrix(n, w, x, y);
        Mat gb = diff_in_matrix(n, b, x, y);
        learn(n.l[k], gw, gb);
        mat_del(gw);
        mat_del(gb);
    }
}

// Trains the network with the given set.
// set: Mat{[x1, x2, ..., xn, y]}.
// Returns the amount of iterations ran.
size_t train(NN n, Mat set) {
    Mat x = mat_upto_col(set, set.m-1);
    Mat y = mat_col(set, set.m-1);

    size_t iters = 0;
    double c = EPS;
    do finite_diff(n, x, y);
    while ((c = cost(n, x, y)) > EPS && ++iters != MAX_ITER);

    return iters;
}

// Prints the matrices of the nn.
void nn_print(NN n) {
    for (size_t i = 0; i < n.len; i++)
        lay_print(n.l[i]);
}

// Prints the results of the nn
// compared to the given set.
void nn_see_results(NN n, Mat set) {
    system("clear");
    nn_print(n);
    Mat x = mat_upto_col(set, n.xs);
    Mat y = mat_from_col(set, n.xs);
    printf("cost: %lf\n", cost(n, x, y));
    for (size_t i = 0; i < x.n; i++) {
        Mat x_row = mat_row(x, i);
        Mat y_row = mat_row(y, i);
        Mat pred = forward(n, x_row);

        mat_print_no_nl(y_row, "y");
        putchar(' ');
        mat_print_no_nl(pred, "p");
        putchar('\n');
    }
}

// Frees the memory used by the nn.
void nn_del(NN n) {
    for (size_t i = 0; i < n.len; i++)
        lay_del(n.l[i]);
}

#endif // __NN_H__
