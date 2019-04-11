#include <assert.h>
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tensorflow/c/c_api.h>
#include <algorithm>
#include <iterator>
#include <vector>
#include <cmath>
#include "hnswindex.h"
#include "hnswlib.h"

TF_Buffer *read_file(const char *path);
TF_Graph *read_graph(const char *path);
Index<float> *create_euclidean(int dimension, const size_t nbItems, const size_t M, const size_t efConstruction, const size_t random_seed);
Index<float> *load_euclidean(int dimension);
float** get_model_input(int dimension, int extra_dimension, int nb_embeddings, Index<float> * index, int nb_items);

const float RAND_MAX_FLOAT = (float)(RAND_MAX);

const std::string path_to_index = "index-10k-random.hnsw";
const char* path_to_model = "/Users/d.parfenchik/Dev/knn-inference/model/model-const.pb";

float get_random_float() {
    return (float)rand()/RAND_MAX_FLOAT;
}
int get_random_id(int nb_items) {
    return abs((int)rand()) % nb_items;
}

void free_buffer(void *data, size_t length) {
    free(data);
}

void free_tensor(void* data, size_t len, void* arg) {
    free(data);
}

int main() {
    int nb_items = 10000;
    int nb_embeddings = 50;
    int embeddings_dimension = 100;
    int extra_dimension = 5;
    int dimension = embeddings_dimension + extra_dimension;
    int M = 16;
    int efConstruction = 200;
    int random_seed = 42;

    TF_Graph *graph = read_graph(path_to_model);
    //Index<float> *index = create_euclidean(embeddings_dimension, nb_items, M, efConstruction, random_seed);
    Index<float> *index = load_euclidean(embeddings_dimension);
    float **values = get_model_input(embeddings_dimension, extra_dimension, nb_embeddings, index, nb_items);
    // Use the graph

    // Create variables to store the size of the input and output variables
    const int num_bytes_in = dimension * nb_items * sizeof(float);
    const int num_bytes_out = dimension * sizeof(float);

    int64_t in_dims[] = {nb_embeddings, dimension};
    int64_t out_dims[] = {dimension};

    std::vector<TF_Output> inputs;
    std::vector<TF_Tensor*> input_values;

    std::cout << "retrieving operations\n";

    TF_Operation* input_op = TF_GraphOperationByName(graph, "product_embeddings");

    std::cout << "retrieved" << input_op << "\n";
    TF_Output input_opout = {input_op, 0};
    inputs.push_back(input_opout);

    TF_Tensor* input = TF_NewTensor(TF_FLOAT, in_dims, 2 /*in_dims.length*/, values, num_bytes_in, free_tensor, 0);
    input_values.push_back(input);

    std::cout << "Input op info: " << TF_OperationNumOutputs(input_op) << "\n";
    std::cout << "Input data info: " << TF_Dim(input, 0) << "\n";
/*
    // ######################
    // Set up graph outputs (similar to setting up graph inputs)
    // ######################

    // Create vector to store graph output operations
    std::vector<TF_Output> outputs;
    TF_Operation* output_op = TF_GraphOperationByName(graph, "output_node0");
    TF_Output output_opout = {output_op, 0};
    outputs.push_back(output_opout);

    // Create TF_Tensor* vector
    std::vector<TF_Tensor*> output_values(outputs.size(), nullptr);

    // Similar to creating the input tensor, however here we don't yet have the
    // output values, so we use TF_AllocateTensor()
    TF_Tensor* output_value = TF_AllocateTensor(TF_FLOAT, out_dims, 2, num_bytes_out);
    output_values.push_back(output_value);

    // As with inputs, check the values for the output operation and output tensor
    std::cout << "Output: " << TF_OperationName(output_op) << "\n";
    std::cout << "Output info: " << TF_Dim(output_value, 0) << "\n";

    // ######################
    // Run graph
    // ######################
    fprintf(stdout, "Running session...\n");
    TF_SessionOptions* sess_opts = TF_NewSessionOptions();
    TF_Session* session = TF_NewSession(graph, sess_opts, status);
    assert(TF_GetCode(status) == TF_OK);

    // Call TF_SessionRun
    TF_SessionRun(session, nullptr,
                    &inputs[0], &input_values[0], inputs.size(),
                    &outputs[0], &output_values[0], outputs.size(),
                    nullptr, 0, nullptr, status);

    // Assign the values from the output tensor to a variable and iterate over them
    float* out_vals = static_cast<float*>(TF_TensorData(output_values[0]));
    for (int i = 0; i < 9; ++i)
    {
        std::cout << "Output values info: " << *out_vals++ << "\n";
    }

    fprintf(stdout, "Successfully run session\n");

    // Delete variables
    TF_CloseSession(session, status);
    TF_DeleteSession(session, status);
    TF_DeleteSessionOptions(sess_opts);
*/
    TF_DeleteGraph(graph);
    return 0;
}

TF_Graph *read_graph(const char *path) {
    TF_Buffer *graph_def = read_file(path);
    TF_Graph *graph = TF_NewGraph();
    TF_Status *status = TF_NewStatus();
    TF_ImportGraphDefOptions *opts = TF_NewImportGraphDefOptions();
    TF_GraphImportGraphDef(graph, graph_def, opts, status);
    TF_DeleteImportGraphDefOptions(opts);
    assert(TF_GetCode(status) == TF_OK);
    fprintf(stdout, "Really Successfully imported graph\n");
    TF_DeleteStatus(status);
    TF_DeleteBuffer(graph_def);

    return 0;
}

TF_Buffer *read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    void *data = malloc(fsize);
    fread(data, fsize, 1, f);
    fclose(f);

    TF_Buffer *buf = TF_NewBuffer();
    buf->data = data;
    buf->length = fsize;
    buf->data_deallocator = free_buffer;
    return buf;
}

Index<float> *create_euclidean(int dimension, const size_t nbItems, const size_t M, const size_t efConstruction, const size_t random_seed) {
    fprintf(stdout, "Building index\n");
    auto start = std::chrono::high_resolution_clock::now();
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
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    float seconds = (float)std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count()/1000000.0f;
    fprintf(stdout, "Built index of %d items in: %.2fs\n", nbItems, seconds);
    index->saveIndex(path_to_index);
    std::cout << "Saved index to " << path_to_index << "\n";
    return index;
}

Index<float> *load_euclidean(int dimension) {
    fprintf(stdout, "Loading index\n");
    hnswlib::SpaceInterface<float> *space = new hnswlib::L2Space(dimension);
    bool normalize = false;
    Index<float> *index = new Index<float>(space, dimension, normalize);
    auto start = std::chrono::high_resolution_clock::now();
    index->loadIndex(path_to_index);
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    float seconds = (float)std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count()/1000000.0f;
    fprintf(stdout, "Loaded index of %d items in: %.2fs\n", index->appr_alg->cur_element_count, seconds);
    return index;
}

float** get_model_input(int dimension, int extra_dimension, int nb_embeddings, Index<float> * index, int nb_items) {
    float** result = new float*[nb_embeddings];
    for(int i = 0; i < nb_embeddings; i++) {
        result[i] = new float[dimension + extra_dimension];
        float* input = result[i];
        int id = get_random_id(nb_items);
        std::vector<float> embedding = index->appr_alg->template getDataByLabel<float>(id);
        //std::cout << "Picked [" << id << "] = " << embedding[0] <<"\n";
        for(int j = 0; j < dimension; j++) {
            input[j] = embedding[j];
        }
        for(int j = dimension; j < dimension + extra_dimension; j++) {
            input[j] = embedding[j];
        }
    }
    return result;
}