#ifndef __NN_H__
#define __NN_H__

#include "layer.h"
#include "set.h"
#include <assert.h>
#include <time.h>
#include <string.h>
#include <stdbool.h>

// Architecture of the neural network.
size_t ARCH[] = { 4, 5, 5, 3 };
enum ACT_FUNC ARCH_FUNCS[] = { TANH, TANH, SIGMOID };
size_t ARCH_LEN = sizeof(ARCH) / sizeof(ARCH[0]);

// Hyperparameters.
double LEARNING_RATE = 10e-1;
size_t MAX_EPOCHS = 10e+4;
double MIN_ERROR = 10e-5;
size_t BATCH_SIZE = 10;

typedef struct NeuralNetwork {
    size_t xs, len;
    Layer *l;
} NN;

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
        lay_print(n.l[i], i, n.l[i].w.m);
}

// Forwards the input values through the network.
Mat static forward_rec(Layer *l, Mat x, size_t n, size_t i) {
    if (i == n) return x;
    return forward_rec(l, lay_forward(l[i], x), n, i+1);
}

Mat static forward(NN n, Mat x) {
    return forward_rec(n.l, x, n.len, 0);
}

// Returns the Matrix of predicted values given x.
Mat nn_forward(NN n, Set x) {
    return forward(n, mat_t(set_to_mat(x)));
}

// Returns a new neural network filled with zeros.
NN static new_nn_zero(NN n) {
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

// Fills the matrices with zeros.
void static fill_nn_zeros(NN n) {
    for (size_t l = 0; l < n.len; l++)
        lay_fill_zeros(n.l[l]);
}

// Calculates the loss of the network
// using Mean Squared Error.
double mse(NN n, Mat x, Mat y) {
    size_t len = y.m;
    Mat errors;
    MAT_ON_STACK(errors, len, 1);

    for (size_t i = 0; i < len; i++) {
        Mat pred = forward(n, mat_col(x, i));
        Mat diff = mat_sub(pred, mat_col(y, i));
        MAT_AT(errors, i, 0) = mat_add(mat_mul(diff, diff));
    }

    return mat_add(errors) / len;
}

// Backpropagation algorithm for neural network learning.
void static backpropagation(NN n, NN g, Mat x, Mat y) {
    fill_nn_zeros(g);
    size_t len = x.m;
    for (size_t s = 0; s < len; s++) {
        Mat inp = mat_col(x, s);
        Mat out = forward(n, inp);
        Mat rvs = mat_col(y, s);
        Mat diff = mat_scalar(mat_sub(out, rvs), 2);

        for (int l = n.len-1; l >= 0; l--) {
            Layer curr = n.l[l];
            Layer grad = g.l[l];
            Mat post_delta = mat_mul(diff, lay_der(curr, grad.a, curr.z));
            Mat prev_a = l > 0 ? n.l[l-1].a : inp;
            Mat prev_z = l > 0 ? g.l[l-1].z : (Mat) {0};

            // dJdW
            mat_dot_sum(grad.w, post_delta, mat_t(prev_a));
            // dJdB
            mat_sum(grad.b, post_delta);
            if (l > 0) diff = mat_dot(prev_z, mat_t(curr.w), post_delta);
        }
    }

    for (size_t l = 0; l < n.len; l++) {
        mat_sub(n.l[l].w, mat_scalar(g.l[l].w, LEARNING_RATE / len));
        mat_sub(n.l[l].b, mat_scalar(g.l[l].b, LEARNING_RATE / len));
    }
}

// Trains the network with the given set.
// Returns the amount of epochs ran.
size_t nn_fit(NN n, Set set) {
    Mat x = set_to_mat(set_get_x(set, n.xs));
    Mat y = set_to_mat(set_get_y(set, n.xs));
    x = mat_t(x);
    y = mat_t(y);

    size_t epochs = 0;
    double c = MIN_ERROR;
    NN g = new_nn_zero(n);

    Set copy;
    SET_ON_STACK(copy, set.n, set.m);
    SET_STACK_COPY(copy, set);

    do {
        Set shuffled = set_shuffle(copy);
        for (size_t i = 0; i < shuffled.n; i += BATCH_SIZE) {
            Set batch = set_batch(shuffled, i, i+BATCH_SIZE);
            Mat x_batch = mat_t(set_to_mat(set_get_x(batch, n.xs)));
            Mat y_batch = mat_t(set_to_mat(set_get_y(batch, n.xs)));
            backpropagation(n, g, x_batch, y_batch);
        }

        printf("%li: cost = %lf\n", epochs, c);
    } while ((c = mse(n, x, y)) > MIN_ERROR && ++epochs < MAX_EPOCHS);

    nn_del(g);
    return epochs;
}

// Prints the results of the nn
// compared to the given set.
void nn_results(NN n, Set set) {
    if (system("clear") == -1)
        return;
    
    nn_print(n);
    Mat x = set_to_mat(set_get_x(set, n.xs));
    Mat y = set_to_mat(set_get_y(set, n.xs));
    x = mat_t(x);
    y = mat_t(y);

    printf("ERROR:\033[0;33m %lf\n", mse(n, x, y));
    for (size_t i = 0; i < x.m; i++) {
        Mat x_col = mat_col(x, i);
        Mat y_col = mat_col(y, i);
        Mat pred = forward(n, x_col);
        mat_print_no_nl(x_col, "x:");
        printf("   ");
        mat_print_no_nl(y_col, "y:");
        printf("   ");
        mat_print_no_nl(pred, "y':");
        puts("");
    }
}

// Saves the nn to a file.
void nn_save(NN n, const char *path) {
    FILE *f = fopen(path, "wb");
    assert(f != NULL);

    size_t written = 0;
    written += fwrite(&n.xs, sizeof(n.xs), 1, f);
    written += fwrite(&n.len, sizeof(n.len), 1, f);
    if (written != 2) {
        fprintf(stderr, "Error saving nn");
        fclose(f);
        return;
    }

    for (size_t i = 0; i < n.len; i++)
        lay_save(n.l[i], f);
    fclose(f);
}

// Returns an empty nn with the given amount of layers.
NN static new_nn_with(size_t xs, size_t len) {
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
    FILE *f = fopen(path, "rb");
    assert(f != NULL);

    size_t xs, len, read = 0;
    read += fread(&xs,  sizeof(xs),  1, f);
    read += fread(&len, sizeof(len), 1, f);
    if (read != 2) {
        fprintf(stderr, "Error reading nn");
        fclose(f);
        exit(1);
    }

    NN n = new_nn_with(xs, len);
    for (size_t i = 0; i < n.len; i++)
        n.l[i] = lay_from(f);
    
    fclose(f);
    return n;
}

#endif // __NN_H__
