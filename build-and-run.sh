#!/usr/bin/env bash

#/usr/local/opt/llvm/bin/clang++ src/main/cpp/hello.cpp -ltensorflow -std=c++11 -openmp -march=native -fpic -fopenmp -w -ftree-vectorize -o hello
/usr/local/opt/llvm/bin/clang++ src/main/cpp/hello.cpp -ltensorflow -march=x86-64 -m64 -O3 -std=c++11 -Wl,-rpath,@loader_path/. -Wall -fPIC -undefined dynamic_lookup -o hello
./hello