#!/usr/bin/env bash

#/usr/local/opt/llvm/bin/clang++ src/main/cpp/hello.cpp -ltensorflow -std=c++11 -openmp -march=native -fpic -fopenmp -w -ftree-vectorize -o hello
#basic - /Users/d.parfenchik/libtensorflow/lib
#optimized-vector - /Users/d.parfenchik/Dev/knn-inference/tf-builds/arch-native/libtensorflow/lib
#mlk - /Users/d.parfenchik/Dev/knn-inference/tf-builds/mkl/libtensorflow/lib
#export DYLD_LIBRARY_PATH=/Users/d.parfenchik/Dev/knn-inference/tf-builds/arch-native/libtensorflow/lib:$DYLD_LIBRARY_PATH
# export DYLD_LIBRARY_PATH=/usr/local/opt/llvm/lib:/Users/d.parfenchik/Dev/knn-inference/tf-builds/mkl/libtensorflow/lib:$DYLD_LIBRARY_PATH
/usr/local/opt/llvm/bin/clang++ src/main/cpp/bench.cpp -ltensorflow -lbenchmark -L/Users/d.parfenchik/Dev/benchmark/build/src/ -march=x86-64 -m64 -O3 -std=c++11 -Wl,-rpath,@loader_path/. -Wall -fPIC -undefined dynamic_lookup -o bin/bench
#/usr/local/opt/llvm/bin/clang++ src/main/cpp/hello.cpp -L/Users/d.parfenchik/Dev/knn-inference/tf-builds/arch-native/ -march=x86-64 -m64 -O3 -std=c++11 -Wl,-rpath,@loader_path/. -Wall -fPIC -undefined dynamic_lookup -o hello
./bin/bench