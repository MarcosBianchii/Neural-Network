# !/bin/bash

gcc set.c -O3 -g -c -lm -o set.o &&
gcc matrix.c -O3 -g -c -lm -o matrix.o &&
gcc layer.c -O3 -g -c -o layer.o &&
gcc threadpool.c -O3 -g -c -pthread -o threadpool.o
