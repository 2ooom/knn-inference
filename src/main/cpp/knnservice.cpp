#include <iostream>
#include <vector>
#include <string>
#include "hnswindex.h"

enum Model {
    Average = 1
};

class KnnService {
public:
    KnnService(Distance distance, const int dim, const int ef_search) :
        distance(distance), dim(dim), ef_search(ef_search) {
    }

    void loadIndex(int index_id, const std::string &path_to_index) {
        Index<float> *index = new Index<float>(distance, dim);
        index->loadIndex(path_to_index);
        index->appr_alg->setEf(ef_search);
        indices_by_id.emplace(index_id, index);
        fprintf(stdout, "Loaded index %d with %zu items\n", index_id, index->appr_alg->cur_element_count);
    }

    size_t getClosestItems(int * index_ids, size_t * query_labels, size_t nb_items, size_t query_index, size_t * result_items, float * result_distances, size_t k, Model model = Average) {
        size_t matrix_size = nb_items * dim;
        std::vector<float> query_vectors(matrix_size);
        int * index_id = index_ids;
        size_t * label = query_labels;
        float * vector = query_vectors.data();

        for(size_t i = 0; i < nb_items; i++) {
            Index<float> * index = indices_by_id[*index_id];
            index->getDataPointerByLabel(*label, vector);
            index_id++;
            label++;
            vector += dim;
        }

        std::vector<float> query(dim);
        float * query_data = query.data();
        if(model == Average) {
            compute_average(nb_items, query_vectors.data(), query_data);
        }
        else {
            throw std::runtime_error("Model not supported" + std::to_string(model));
        }
        Index<float> * index = indices_by_id[query_index];

        size_t nbResults = index->knnQuery(query_data, result_items, result_distances, k);
        return nbResults;
    }

    void compute_average(int nb_items, float* input, float* result) {
        float *cell_ptr = input;
        float *result_ptr = result;
        for(int i = 0; i < nb_items; i++) {
            double sum = 0;
            for(int j = 0; j < dim; j++) {
                sum += *cell_ptr;
                cell_ptr ++;
            }
            *result_ptr = (float)(sum / dim);
            result_ptr++;
        }
    }

    Distance distance;
    int dim;
    int ef_search;
    std::unordered_map<int, Index<float> * > indices_by_id;

    ~KnnService() {
        for( const auto& pair : indices_by_id )
            delete pair.second;
    }
};