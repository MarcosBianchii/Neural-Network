#include "nn.h"

size_t ARCH[] = {2, 2, 1};
size_t ARCH_LEN = sizeof(ARCH) / sizeof(ARCH[0]);
act_func ARCH_FUNCS[] = {sigmoid, sigmoid};

double MAT[][3] = {
    {0,0,0},
    {0,1,1},
    {1,0,1},
    {1,1,0},
};

size_t MAT_LEN = sizeof(MAT) / sizeof(MAT[0]);
size_t MAT_DATA_LEN = sizeof(MAT[0]) / sizeof(MAT[0][0]);

int main() {
    srand(time(NULL));
    
    NN n = nn_new(ARCH, ARCH_FUNCS, ARCH_LEN);
    Mat set = mat_from(MAT_LEN, MAT_DATA_LEN, MAT);
    train(n, set);

    nn_see_results(n, set);

    nn_del(n);
    mat_del(set);
    return 0;
}
