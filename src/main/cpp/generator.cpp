#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>

#include "common.h"
#include "hnswindex.h"
#include "hnswlib.h"

Index<float> *create_euclidean(int dimension, const size_t nbItems, const size_t M, const size_t efConstruction, const size_t random_seed);

int main(int argc, char* argv[]) {
    std::string cmd = (const char* )argv[1];
    if(cmd == "index") {
        create_euclidean(embeddings_dimension, nb_items, M, efConstruction, random_seed);
    } else if (cmd == "scenario") {
        FILE * fp;
        fp = fopen (path_to_scenario.c_str(), "w+");
        for(int i = 0; i < nb_iterations; i++) {
            for (int i = 0; i < nb_embeddings; i++) {
                fprintf(fp, "%d", get_random_id(nb_items));
                if (i < nb_embeddings - 1)
                    fprintf(fp, "%c", delimeter);
            }
            fprintf(fp, "\n");
        }
    }
    return 0;
}

Index<float> *create_euclidean(int dimension, const size_t nbItems, const size_t M, const size_t efConstruction, const size_t random_seed) {
    fprintf(stdout, "Building index\n");
    auto start = now();
    hnswlib::SpaceInterface<float> *space = new hnswlib::L2Space(dimension);
    bool normalize = false;
    Index<float> *index = new Index<float>(space, dimension, normalize);
    index->initNewIndex(nbItems, M, efConstruction, random_seed);
    std::vector<float> embedding(dimension);
    for (int i = 0; i < nbItems; i++) {
        for(int j = 0; j < dimension; j++) {
            embedding[j] = get_random_float();
        }
        index->addItem(embedding.data(), i);
    }
    fprintf(stdout, "Built index of %zu items in: %.2fs\n", nbItems, get_elepased_seconds(start));
    index->saveIndex(path_to_index);
    std::cout << "Saved index to " << path_to_index << "\n";
    return index;
}
