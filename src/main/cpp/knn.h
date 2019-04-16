#include "hnswindex.h"
#include "hnswlib.h"

Index<float> *load_euclidean(int dimension, const std::string &path) {
    hnswlib::SpaceInterface<float> *space = new hnswlib::L2Space(dimension);
    bool normalize = false;
    Index<float> *index = new Index<float>(space, dimension, normalize);
    index->loadIndex(path);
    fprintf(stdout, "Loaded index of %zu items\n", index->appr_alg->cur_element_count);
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