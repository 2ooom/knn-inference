#include <assert.h>
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>

const int nb_iterations = 15000;
const int nb_items = 10000;
const int nb_embeddings = 50;
const int embeddings_dimension = 100;
const int random_seed = 42;
const int M = 16;
const int efConstruction = 200;

const char delimeter = ' ';
const std::string path_to_scenario = "input-scenario-10k.csv";
const std::string path_to_index = "index-10k-random.hnsw";

const float RAND_MAX_FLOAT = (float)(RAND_MAX);

int get_random_id(int nb_items) {
    return abs((int)rand()) % nb_items;
}

float get_random_float() {
    return (float)rand()/RAND_MAX_FLOAT;
}

std::chrono::time_point<std::chrono::high_resolution_clock, std::chrono::high_resolution_clock::duration> now() {
    return std::chrono::high_resolution_clock::now();
}

float get_elepased_microseconds(std::chrono::time_point<std::chrono::high_resolution_clock, std::chrono::high_resolution_clock::duration> start) {
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    return (float)std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
}

float get_elepased_seconds(std::chrono::time_point<std::chrono::high_resolution_clock, std::chrono::high_resolution_clock::duration> start) {
    return get_elepased_microseconds(start)/1000000.0f;
}