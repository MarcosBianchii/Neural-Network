# nn.h (WIP)

###### needs updating...

This framework is an implementation of a neural network made in C. It can accomodate any number of layers and neurons. It is also possible to use different activation functions per layer. The framework is designed to be as easy to use as possible.

## Usage

```bash
$ ./build.sh # Compile and run
$ ./main     # Run
```
The user of the framework only needs to specify the architecture of the neural network and the activation functions for each layer. The framework will take care of managing the matrix operations and training of the neural network.

## Example
```C
#include "nn.h"

int main() {
    // Set seed for generating the random weights and biases.
    srand(time(NULL));
    
    // Define the architecture of the neural network in nn.h.
    // ARCH:       [input layer, neurons per layer, ..., output layer]
    // ARCH_FUNCS: [activation function per layer (not counting input layer)]
    // ARCH_LEN:   length of ARCH array.
    NN n = nn_new(ARCH, ARCH_FUNCS, ARCH_LEN);

    // Load set from file.
    Set s = set_from("binary_sum.csv", ",");

    // Train the neural network.
    nn_fit(n, s);

    // Print the neural network and test it's predictions.
    nn_results(n, s);

    // Save the nn's parameters in a file.
    nn_save(n, "nn.txt");

    // Free memory.
    nn_del(n);
    set_del(s);
    return 0;
}

```

## Activation Functions
The following activation functions are available:

* NULL: Lineal function.
* RELU: Rectified Linear Unit.
* TANH: Hyperbolic tangent.
* SIGMOID: Sigmoid function.

```C
// When defining the architecture make sure to use
// this enum type and not the function pointers.
typedef enum ACT_FUNC { RELU, TANH, SIGMOID } ACT_FUNC;
```


## Configuration
The neural network can be configured in `nn.h` by changing the following variables:

```C
// Neural network architecture.
size_t ARCH[] = {4, 5, 5, 3};
// Activation functions per layer (always 1 less than architecture).
// If ARCH_FUNCS is NULL, every layer will be lineal.
// For a specific layer to be lineal set ARCH_FUNCS[i] = NULL.
ACT_FUNC ARCH_FUNCS[] = {RELU, TANH, SIGMOID};
// Length of ARCH array.
size_t ARCH_LEN = sizeof(ARCH) / sizeof(ARCH[0]);
```
Hyperparameters can be changed in `nn.h`:
```C
// Epsilon for the finite difference method.
double EPS = 10e-5;
// Coefficient for the regularization term.
double LEARNING_RATE = 10e-1;
// Cap for the number of iterations.
size_t MAX_ITER = 10e+4;
// Minimum error for the neural network to stop training.
double MIN_ERROR = 10e-5;
```