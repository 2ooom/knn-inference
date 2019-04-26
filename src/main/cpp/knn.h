#include "hnswindex.h"
#include "hnswlib.h"

Index<float> *load_euclidean(int dimension, const std::string &path, int efSearch) {
    hnswlib::SpaceInterface<float> *space = new hnswlib::L2Space(dimension);
    bool normalize = false;
    Index<float> *index = new Index<float>(space, dimension, normalize);
    index->loadIndex(path);
    index->appr_alg->setEf(efSearch);
    //fprintf(stdout, "Loaded index of %zu items with efConstruction=%zu ef=%zu\n", index->appr_alg->cur_element_count, index->appr_alg->ef_construction_, index->appr_alg->ef_);
    return index;
}

float* get_model_input(int dim, int extra_dimension, Index<float> * index, int* ids, float* result, int rows) {
    int cols = dim + extra_dimension;
    float* row_ptr = result;
    int* id_ptr = ids;
    for(int i = 0; i < rows; i++) {
        index->getDataPointerByLabel(*id_ptr, row_ptr);
        row_ptr += dim;
        for(int j = dim; j < cols; j++) {
            *row_ptr = *(row_ptr - dim);
            row_ptr++;
        }
        //std::cout << "Input "<< i << " [" << *id_ptr << "] First = " << *(row_ptr-cols) << "; Last = " << *(row_ptr - 1) << " index = " << row_ptr - result<<"\n";
        id_ptr++;
    }
    //std::cout << "Input first="<< result[0] << " Last=" << result[rows*cols - 1] << "\n";
    return result;
}

void compute_average(int rows, int cols, float* input, float* result) {
    float *cell_ptr = input;
    float *result_ptr = result;
    for(int i = 0; i < rows; i++) {
        double sum = 0;
        for(int j = 0; j < cols; j++) {
            sum += *cell_ptr;
            cell_ptr ++;
        }
        *result_ptr = (float)(sum / cols);
        result_ptr++;
    }
}