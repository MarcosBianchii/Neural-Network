#include "nn/nn.h"

#include "nn/threadpool.h"
#include <stdio.h>
#include <unistd.h>

int main() {
    srand(time(NULL));
    
    NN n = nn_new(ARCH, ARCH_FUNCS, ARCH_LEN);
    Set s = set_from_csv("data/binary_sum.csv", ",");

    nn_fit(n, s);
    nn_results(n, s);

    nn_del(n);
    set_del(s);
    return 0;
}
