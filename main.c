#include "nn.h"

double SET[][7] = {
    {0,0, 0,0, 0,0,0},
    {0,0, 0,1, 0,0,1},
    {0,0, 1,0, 0,1,0},
    {0,0, 1,1, 0,1,1},
    {0,1, 0,0, 0,0,1},
    {0,1, 0,1, 0,1,0},
    {0,1, 1,0, 0,1,1},
    {0,1, 1,1, 1,0,0},
    {1,0, 0,0, 0,1,0},
    {1,0, 0,1, 0,1,1},
    {1,0, 1,0, 1,0,0},
    {1,0, 1,1, 1,0,1},
    {1,1, 0,0, 0,1,1},
    {1,1, 0,1, 1,0,0},
    {1,1, 1,0, 1,0,1},
    {1,1, 1,1, 1,1,0},
};

size_t SET_LEN = sizeof(SET) / sizeof(SET[0]);
size_t SET_DATA_LEN = sizeof(SET[0]) / sizeof(SET[0][0]);

int main() {
    srand(time(NULL));
    
    NN n = nn_new(ARCH, ARCH_FUNCS, ARCH_LEN);
    Mat set = mat_from(SET_LEN, SET_DATA_LEN, SET);

    nn_fit(n, set);
    nn_see_results(n, set);

    nn_del(n);
    mat_del(set);
    return 0;
}
