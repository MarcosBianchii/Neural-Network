#ifndef __NN_H__
#define __NN_H__

#include <math.h>
#include "layer.h"
#include "set.h"

// Architecture of the neural network.
size_t ARCH[] = { 4, 5, 3 };
ACT_FUNC ARCH_FUNCS[] = { TANH, SIGMOID };
size_t ARCH_LEN = sizeof(ARCH) / sizeof(ARCH[0]);
double EPS = 10e-5;
double LEARNING_RATE = 10e-1;
size_t MAX_ITER = 10;
double MIN_ERROR = 10e-5;

typedef struct NeuralNetwork {
    size_t xs, len;
    Layer *l;
} NN;

double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

double sigmoid_der(double x) {
    double s = sigmoid(x);
    return s * (1 - s);
}

double relu(double x) {
    return x > 0 ? x : 0;
}

double relu_der(double x) {
    return x > 0 ? 1 : 0;
}

double tanh_der(double x) {
    double t = tanh(x);
    return 1 - t * t;
}

double lineal(double x) {
    return x;
}

double lineal_der(double x) {
    return 1;
}

double absf(double x) {
    return x < 0 ? -x : x;
}

#define NN_OUTPUT(n, i) (MAT_AT((n).l[(n).len-1].a, 0, i))

// Converts the matrix into a Set.
Set mat_to_set(Mat m) {
    return (Set) {
        .data = m.data,
        .free_ptr = NULL,
        .n = m.n,
        .m = m.m,
        .stride = m.stride,
    };
}

// Converts the set into a matrix.
Mat set_to_mat(Set s) {
    return (Mat) {
        .data = s.data,
        .free_ptr = NULL,
        .n = s.n,
        .m = s.m,
        .step = 1,
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
        ACT_FUNC g = f ? f[i] : LINEAL;
        n.l[i] = lay_new(arch[i+1], input_size, g);
        lay_assert(n.l[i]);
        input_size = arch[i+1];
    }

    return n;
}

// Prints the matrices of the nn.
void nn_print(NN n) {
    puts("Neural Network:");
    for (size_t i = 0; i < n.len; i++)
        lay_print(n.l[i], i);
}

// Forwards the input values through the network.
static Mat forward_rec(Layer *l, Mat x, size_t n, size_t i) {
    if (i == n) return x;
    return forward_rec(l, lay_forward(l[i], x), n, i+1);
}

// Tries to predict y given x.
Mat nn_forward(NN n, Mat x) {
    return forward_rec(n.l, x, n.len, 0);
}

// Calculates the cost for the current state
// of the neural network given two sets of data.
double nn_cost(NN n, Mat x, Mat y) {
    double sum = 0;
    size_t data_len = x.m;
    for (int k = 0; k < y.n; k++) {
        for (int i = 0; i < data_len; i++) {
            Mat x_col = mat_col(x, i);
            Mat y_col = mat_col(y, i);
            double pred = MAT_AT(nn_forward(n, x_col), 0, k);
            double diff = MAT_AT(y_col, 0, k) - pred;
            sum += diff * diff;
        }
    }

    return sum / data_len;
}

// Applies the changes to the layer given.
static void learn(Layer l, Mat gw, Mat gb) {
    mat_sub(l.w, mat_scalar(gw, LEARNING_RATE));
    mat_sub(l.b, mat_scalar(gb, LEARNING_RATE));
}

// Aproximates the derivative of the cost function.
static Mat diff_in_matrix(NN n, Mat m, Mat x, Mat y) {
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

// Aproximates the gradient of the cost function for
// every parameter using the finite difference method.
static void finite_diff(NN n, Mat x, Mat y) {
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

static double fn_der(ACT_FUNC f, double x) {
    switch (f) {
        case SIGMOID: return sigmoid_der(x);
        case RELU: return relu_der(x);
        case TANH: return tanh_der(x);
        case LINEAL: return lineal_der(x);
        default: 1;
    }
}

// Backpropagation algorithm for neural network learning.
void backprop(NN n, NN g, Set x, Set y) {
    // size_t len = x.n;
    // for (size_t s = 0; s < len; s++) {
    //     Set inp = set_row(x, s);
    //     Set out = nn_forward(n, inp);
    //     Set rvs = set_row(y, s);

    //     for (size_t i = 0; i < y.m; i++)
    //         NN_OUTPUT(g, i) = SET_AT(out, 0, i) - SET_AT(rvs, 0, i);

    //     for (int l = n.len-1; l >= 0; l--) {
    //         Layer curr_lay  = n.l[l];
    //         Layer curr_grad = g.l[l];
    //         Layer prev_lay  = l > 0 ? n.l[l-1] : (Layer) {
    //             .a = set_to_mat(inp),
    //         };

    //         Layer prev_grad = l > 0 ? g.l[l-1] : (Layer) {
    //             .a = mat_fill(mat_new(1, curr_lay.w.m), 1),
    //         };

    //         for (size_t j = 0; j < curr_lay.w.n; j++) {
    //             double curr_act = MAT_AT(curr_lay.a,  0, j);
    //             double diff_act = MAT_AT(curr_grad.a, 0, j);
    //             double b = MAT_AT(curr_lay.b, 0, j);
    //             double z = MAT_AT(curr_lay.z, 0, j);
    //             MAT_AT(curr_grad.b, 0, j) += 2 * diff_act * fn_der(curr_lay.act_func, z);

    //             for (size_t k = 0; k < prev_lay.w.n; k++) {
    //                 double prev_act = MAT_AT(prev_lay.a, 0, j);
    //                 double w = MAT_AT(curr_lay.w, j, k);
    //                 MAT_AT(curr_grad.w, j, k) += 2 * diff_act * fn_der(curr_lay.act_func, z) * prev_act;
    //                 MAT_AT(prev_grad.a, 0, j) += 2 * diff_act * fn_der(curr_lay.act_func, z) * w;
    //             }
    //         }

    //         if (l == 0) { mat_del(prev_grad.a); }
    //     }
    // }

    // for (size_t l = 0; l < n.len; l++) {
    //     mat_scalar(g.l[l].w, 1.0 / len);
    //     mat_scalar(g.l[l].b, 1.0 / len);
    //     learn(n.l[l], g.l[l].w, g.l[l].b);
    // }
}

// Returns a new neural network filled with zeros.
NN nn_new_zero(NN n) {
    NN g = (NN) {
        .l = malloc(sizeof(*g.l) * n.len),
        .xs = n.xs,
        .len = n.len,
    };

    assert(g.l != NULL);
    for (size_t i = 0; i < n.len; i++)
        g.l[i] = lay_new_zero(n.l[i]);

    return g;
}

// Frees the memory used by the nn.
void nn_del(NN n) {
    for (size_t i = 0; i < n.len; i++)
        lay_del(n.l[i]);
    free(n.l);
}

// Trains the network with the given set.
// Returns the amount of iterations ran.
size_t nn_fit(NN n, Set set) {
    Mat x = set_to_mat(set_get_x(set, n.xs));
    Mat y = set_to_mat(set_get_y(set, n.xs));
    x = mat_t(x);
    y = mat_t(y);

    size_t iters = 0;
    double c = MIN_ERROR;
    // NN g = nn_new_zero(n);

    do {
        finite_diff(n, x, y);
        // backprop(n, g, x, y);
        printf("%li: cost = %lf\n", iters, c);
    } while ((c = nn_cost(n, x, y)) > MIN_ERROR && ++iters != MAX_ITER);

    // nn_del(g);
    return iters;
}

// Prints the results of the nn
// compared to the given set.
void nn_results(NN n, Set set) {
    system("clear");
    nn_print(n);
    Mat x = set_to_mat(set_get_x(set, n.xs));
    Mat y = set_to_mat(set_get_y(set, n.xs));
    x = mat_t(x);
    y = mat_t(y);

    printf("ERROR:\033[0;33m %lf\n", nn_cost(n, x, y));
    for (size_t i = 0; i < x.m; i++) {
        Mat x_col = mat_col(x, i);
        Mat y_col = mat_col(y, i);
        Mat pred = nn_forward(n, x_col);
        mat_print_no_nl(x_col, "x");
        printf("   ");
        mat_print_no_nl(y_col, "y");
        printf("   ");
        mat_print_no_nl(pred, "y'");
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

// Returns an empty nn with the given amount of layers.
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
        n.l[i] = lay_from(f);
    
    fclose(f);
    return n;
}

#endif // __NN_H__
