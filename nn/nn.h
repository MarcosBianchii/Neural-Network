#ifndef __NN_H__
#define __NN_H__

#include "layer.h"
#include "set.h"
#include <assert.h>
#include <time.h>

// Architecture of the neural network.
size_t ARCH[] = { 4, 5, 5, 3 };
enum ACT_FUNC ARCH_FUNCS[] = { TANH, TANH, SIGMOID };
size_t ARCH_LEN = sizeof(ARCH) / sizeof(ARCH[0]);

// Hyperparameters.
double LEARNING_RATE = 10e-1;
size_t MAX_ITER = 10e+4;
double MIN_ERROR = 10e-5;

typedef struct NeuralNetwork {
    size_t xs, len;
    Layer *l;
} NN;

double static absf(double x) {
    return x < 0 ? -x : x;
}

double sigmoid_der(double x) {
    double s = sigmoid(x);
    return s * (1 - s);
}

double relu_der(double x) {
    return x > 0 ? 1 : 0;
}

double tanh_der(double x) {
    double t = tanh(x);
    return 1 - t * t;
}

double lineal_der(double x) {
    return 1;
}

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
NN nn_new(size_t arch[], enum ACT_FUNC *f, size_t len) {
    assert(len > 1);
    assert(f != NULL);
    NN n = (NN) {
        .l = malloc(sizeof(*n.l) * (len-1)),
        .xs = arch[0],
        .len = len-1,
    };

    assert(n.l != NULL);
    size_t input_size = arch[0];
    for (size_t i = 0; i < len-1; i++) {
        n.l[i] = lay_new(arch[i+1], input_size, f[i]);
        lay_assert(n.l[i]);
        input_size = arch[i+1];
    }

    return n;
}

// Frees the memory used by the nn.
void nn_del(NN n) {
    for (size_t i = 0; i < n.len; i++)
        lay_del(n.l[i]);
    free(n.l);
}

// Prints the matrices of the nn.
void nn_print(NN n) {
    puts("Neural Network:");
    for (size_t i = 0; i < n.len; i++)
        lay_print(n.l[i], i);
}

// Forwards the input values through the network.
Mat static forward_rec(Layer *l, Mat x, size_t n, size_t i) {
    if (i == n) return x;
    return forward_rec(l, lay_forward(l[i], x), n, i+1);
}

// Tries to predict y given x.
Mat static forward(NN n, Mat x) {
    return forward_rec(n.l, x, n.len, 0);
}

// Returns the Matrix of predicted values given the set.
Mat nn_forward(NN n, Set set) {
    return forward(n, mat_t(set_to_mat(set_get_x(set, n.xs))));
}

// Calculates the cost for the current state
// of the neural network given two sets of data.
double cost(NN n, Mat x, Mat y) {
    size_t len = x.m;
    Mat errors = mat_new(len, 1);
    for (size_t i = 0; i < len; i++) {
        Mat pred = forward(n, mat_col(x, i));
        Mat diff = mat_sub(pred, mat_col(y, i));
        MAT_AT(errors, i, 0) = mat_add(mat_mul(diff, diff));
    }

    double cost = mat_add(errors) / len;
    mat_del(errors);
    return cost;
}

// Returns a new neural network filled with zeros.
NN static nn_new_zero(NN n) {
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

act_func_t static fn_der(enum ACT_FUNC f) {
    switch (f) {
        case RELU:    return relu_der;
        case TANH:    return tanh_der;
        case SIGMOID: return sigmoid_der;
        case LINEAL:  return lineal_der;
        default: 1;
    }
}

// Fills the matrices with zeros.
void static nn_fill_zeros(NN n) {
    for (size_t l = 0; l < n.len; l++)
        lay_fill_zeros(n.l[l]);
}

// Backpropagation algorithm for neural network learning.
void static backpropagate(NN n, NN g, Mat x, Mat y) {
    nn_fill_zeros(g);
    size_t len = x.m;
    for (size_t s = 0; s < len; s++) {
        Mat inp = mat_col(x, s);
        Mat out = forward(n, inp);
        Mat rvs = mat_col(y, s);
        Mat diff = mat_sub(out, rvs);

        for (int l = n.len-1; l >= 0; l--) {
            Layer curr = n.l[l];
            Layer grad = g.l[l];
            Mat post_delta = mat_mul(diff, mat_func(grad.a, curr.z, fn_der(curr.act_func)));
            Mat prev_a = l > 0 ? n.l[l-1].a : inp;
            Mat prev_z = l > 0 ? g.l[l-1].z : inp;

            Mat dJdW = mat_dot_sum(grad.w, post_delta, mat_t(prev_a));
            Mat dJdB = post_delta;
            mat_sum(grad.b, dJdB);

            if (l > 0) diff = mat_dot(prev_z, mat_t(curr.w), post_delta);
        }
    }

    for (size_t l = 0; l < n.len; l++) {
        mat_sub(n.l[l].w, mat_scalar(g.l[l].w, LEARNING_RATE / len));
        mat_sub(n.l[l].b, mat_scalar(g.l[l].b, LEARNING_RATE / len));
    }
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
    NN g = nn_new_zero(n);

    do {
        backpropagate(n, g, x, y);
        printf("%li: cost = %lf\n", iters, c);
    } while ((c = cost(n, x, y)) > MIN_ERROR && ++iters < MAX_ITER);

    nn_del(g);
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

    printf("ERROR:\033[0;33m %lf\n", cost(n, x, y));
    for (size_t i = 0; i < x.m; i++) {
        Mat x_col = mat_col(x, i);
        Mat y_col = mat_col(y, i);
        Mat pred = forward(n, x_col);
        mat_print_no_nl(x_col, "x:");
        printf("   ");
        mat_print_no_nl(y_col, "y:");
        printf("   ");
        mat_print_no_nl(pred, "y':");
        puts("   ");
    }
}

// Saves the nn to a file.
void nn_save(NN n, const char *path) {
    FILE *f = fopen(path, "w");
    assert(f != NULL);

    fwrite(&n.xs, sizeof(n.xs), 1, f);
    fwrite(&n.len, sizeof(n.len), 1, f);
    for (size_t i = 0; i < n.len; i++)
        lay_save(n.l[i], f);
    fclose(f);
}

// Returns an empty nn with the given amount of layers.
NN static nn_new_with(size_t xs, size_t len) {
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
    fread(&xs, sizeof(xs), 1, f);
    fread(&len, sizeof(len), 1, f);

    NN n = nn_new_with(xs, len);
    for (size_t i = 0; i < n.len; i++)
        n.l[i] = lay_from(f);
    
    fclose(f);
    return n;
}

#endif // __NN_H__
