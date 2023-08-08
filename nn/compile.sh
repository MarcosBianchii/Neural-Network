# !/bin/bash

gcc set.c -O3 -g -c -o set.o &&
gcc matrix.c -O3 -g -c -o matrix.o &&
gcc layer.c -O3 -g -c -o layer.o
