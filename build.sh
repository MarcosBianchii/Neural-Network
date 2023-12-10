# !/bin/bash

gcc main.c nn/*.o -O3 -g -lm -pthread -o main && ./main
