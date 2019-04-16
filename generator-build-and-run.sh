#!/usr/bin/env bash
/usr/local/opt/llvm/bin/clang++ src/main/cpp/generator.cpp -march=x86-64 -m64 -O3 -std=c++11 -o bin/generator
bin/generator ${1}