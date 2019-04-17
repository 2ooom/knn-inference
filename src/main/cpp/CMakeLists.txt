cmake_minimum_required (VERSION 2.6)
project (hnsw_lib_jni)

include_directories("${PROJECT_BINARY_DIR}")

set(CMAKE_CXX_COMPILER  /usr/local/opt/llvm/bin/clang++ )
set(CMAKE_CXX_LINKER /usr/local/opt/llvm/bin/ld )

set(SOURCE_LIB jnibindings.cpp)

add_library(jnibindings STATIC ${SOURCE_LIB})
link_directories(
       /usr/local/opt/llvm/lib
   )

SET( CMAKE_CXX_FLAGS  "-Ofast -DNDEBUG -std=c++11 -DHAVE_CXX0X -openmp -march=native -fpic -w -fopenmp -ftree-vectorize" )
target_link_libraries(jnibindings) 
