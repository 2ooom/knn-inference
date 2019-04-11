#!/usr/bin/env bash

/usr/local/opt/llvm/bin/clang++ src/main/cpp/hello.cpp -ltensorflow -std=c++11 -openmp -march=native -fpic -fopenmp -w -ftree-vectorize -o hello
./hello