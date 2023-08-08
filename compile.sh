# !/bin/bash

gcc nn/set.c -O3 -g -c -o nn/set.o &&
gcc nn/matrix.c -O3 -g -c -o nn/matrix.o &&
gcc nn/layer.c -O3 -g -c -o nn/layer.o
