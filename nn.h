#ifndef __NN_H__
#define __NN_H__

#include <math.h>
#include "layer.h"
#include "set.h"

double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

double relu(double x) {
    return x > 0 ? x : 0;
}

// Architecture of the neural network.
size_t ARCH[] = {4, 5, 5, 3};
ACT_FUNC ARCH_FUNCS[] = {RELU, TANH, SIGMOID};
size_t ARCH_LEN = sizeof(ARCH) / sizeof(ARCH[0]);
double EPS = 10e-5;
double LEARNING_RATE = 10e-1;
size_t MAX_ITER = 10e+4;
double MIN_ERROR = 10e-5;

typedef struct NeuralNetwork {
    size_t xs, len;
    Layer *l;
} NN;

double absf(double x) {
    return x < 0 ? -x : x;
}

// Converts the matrix into a Set.
Set mat_to_set(Mat m) {
    return (Set) {
        .data = m.data,
        .free_ptr = m.free_ptr,
        .n = m.n,
        .m = m.m,
        .stride = m.stride,
    };
}

// Converts the set into a matrix.
Mat set_to_mat(Set s) {
    return (Mat) {
        .data = s.data,
        .free_ptr = s.free_ptr,
        .n = s.n,
        .m = s.m,
        .stride = s.stride,
    };
}

// arch: an array of [params, [hidden layers], outputs]
NN nn_new(size_t arch[], ACT_FUNC *f, size_t len) {
    assert(len > 1);
    NN n = (NN) {
        .l = malloc(sizeof(*n.l) * (len-1)),
        .xs = arch[0],
        .len = len-1,
    };

    assert(n.l != NULL);
    size_t input_size = arch[0];
    for (size_t i = 0; i < len-1; i++) {
        ACT_FUNC g = f ? f[i] : -1;
        n.l[i] = lay_new(arch[i+1], input_size, g);
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
Set nn_forward(NN n, Set x) {
    Mat s = set_to_mat(x);
    return mat_to_set(forward_rec(n.l, s, n.len, 0));
}

// Calculates the cost for the current state
// of the neural network given two sets of data.
double nn_cost(NN n, Set x, Set y) {
    double sum = 0;
    size_t data_len = x.n;
    for (int k = 0; k < y.m; k++) {
        for (int i = 0; i < data_len; i++) {
            Set x_row = set_row(x, i);
            Set y_row = set_row(y, i);
            double pred = nn_forward(n, x_row).data[k];
            double diff = SET_AT(y_row, 0, k) - pred;
            sum += diff * diff;
        }
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

static Mat diff_in_matrix(NN n, Mat m, Set x, Set y) {
    Mat g = mat_new(m.n, m.m);
    double c = nn_cost(n, x, y);
    for (size_t i = 0; i < m.n; i++)
        for (size_t j = 0; j < m.m; j++) {
            double prev_value = MAT_AT(m, i, j);
            MAT_AT(m, i, j) += EPS;
            MAT_AT(g, i, j) = (nn_cost(n, x, y) - c) / EPS;
            MAT_AT(m, i, j) = prev_value;
        }
    return g;
}

static void finite_diff(NN n, Set x, Set y) {
    for (size_t k = 0; k < n.len; k++) {
        Mat w = lay_weights(n.l[k]);
        Mat b = lay_biases(n.l[k]);
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
size_t nn_fit(NN n, Set set) {
    Set x = set_get_x(set, n.xs);
    Set y = set_get_y(set, n.xs);

    size_t iters = 0;
    double c = EPS;
    do {
        finite_diff(n, x, y);
        printf("%li: cost = %lf\n", iters, c);
    } while ((c = nn_cost(n, x, y)) > MIN_ERROR && ++iters != MAX_ITER);

    return iters;
}

// Prints the matrices of the nn.
void nn_print(NN n) {
    puts("Neural Network:");
    for (size_t i = 0; i < n.len; i++)
        lay_print(n.l[i], i);
}

// Prints the results of the nn
// compared to the given set.
void nn_results(NN n, Set set) {
    system("clear");
    nn_print(n);
    Set x = set_get_x(set, n.xs);
    Set y = set_get_y(set, n.xs);

    printf("cost:\033[0;33m %lf\n", nn_cost(n, x, y));
    for (size_t i = 0; i < x.n; i++) {
        Set x_row = set_row(x, i);
        Set y_row = set_row(y, i);
        Set pred = nn_forward(n, x_row);
        set_print_no_nl(x_row, "x");
        printf("   ");
        set_print_no_nl(y_row, "y'");
        printf("   ");
        set_print_no_nl(pred, "y");
        puts("   ");
    }
}

// Saves the nn to a file.
void nn_save(NN n, const char *path) {
    FILE *f = fopen(path, "w");
    assert(f != NULL);
    fwrite(&n.xs, sizeof(size_t), 1, f);
    fwrite(&n.len, sizeof(size_t), 1, f);
    for (size_t i = 0; i < n.len; i++)
        fwrite(&n.l[i].act_func, sizeof(ACT_FUNC), 1, f);
    for (size_t i = 0; i < n.len; i++)
        lay_save(n.l[i], f);
    fclose(f);
}

NN nn_new_with(size_t xs, size_t len) {
    NN n = (NN) {
        .xs = xs,
        .len = len,
        .l = malloc(sizeof(Layer) * len),
    };

    assert(n.l != NULL);
    return n;
}

// Loads a nn from a file.
NN nn_from(const char *path) {
    FILE *f = fopen(path, "r");
    assert(f != NULL);

    size_t xs, len;
    fread(&xs, sizeof(size_t), 1, f);
    fread(&len, sizeof(size_t), 1, f);

    NN n = nn_new_with(xs, len);
    for (size_t i = 0; i < n.len; i++)
        n.l[i] = lay_from_file(f);
    
    fclose(f);
    return n;
}

// Frees the memory used by the nn.
void nn_del(NN n) {
    for (size_t i = 0; i < n.len; i++)
        lay_del(n.l[i]);
    free(n.l);
}

#endif // __NN_H__
