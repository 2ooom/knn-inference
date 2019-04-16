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
TF_Graph *read_graph(const char *path, TF_Status *status);
Index<float> *create_euclidean(int dimension, const size_t nbItems, const size_t M, const size_t efConstruction, const size_t random_seed);
Index<float> *load_euclidean(int dimension);
float* get_model_input(int dimension, int extra_dimension, Index<float> * index, int * ids, float* result, int rows);
void set_random_ids(int* ids, int len, int max_id);
std::chrono::time_point<std::chrono::high_resolution_clock, std::chrono::high_resolution_clock::duration> now();
float get_elepased_microseconds(std::chrono::time_point<std::chrono::high_resolution_clock, std::chrono::high_resolution_clock::duration> start);
float get_elepased_seconds(std::chrono::time_point<std::chrono::high_resolution_clock, std::chrono::high_resolution_clock::duration> start);

const float RAND_MAX_FLOAT = (float)(RAND_MAX);

const std::string path_to_index = "index-10k-random.hnsw";
const char* path_to_model = "model/model-nn.pb";
const char* input_node_name = "product_embeddings";
const char* output_node_name = "user_embeddings";

const int nb_iterations = 15000;
const int nb_items = 10000;
const int nb_embeddings = 50;
const int embeddings_dimension = 100;
const int extra_dimension = 5;
const int dimension = embeddings_dimension + extra_dimension;
const int M = 16;
const int k = 20;
const int efConstruction = 200;
const int random_seed = 42;

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
    TF_Status *status = TF_NewStatus();
    TF_Graph *graph = read_graph(path_to_model, status);

    // Configure input
    std::vector<TF_Output> inputs;
    TF_Operation* input_op = TF_GraphOperationByName(graph, input_node_name);
    std::cout << "Input retrieved operation '" << TF_OperationName(input_op) << "' Type '" << TF_OperationOpType(input_op) << "'\n";
    TF_Output input_opout = {input_op, 0};
    inputs.push_back(input_opout);

    // Configure output
    std::vector<TF_Output> outputs;
    TF_Operation* output_op = TF_GraphOperationByName(graph, output_node_name);
    std::cout << "Retrieved operation '" << TF_OperationName(output_op) << "' Type '" << TF_OperationOpType(output_op) << "'\n";
    TF_Output output_opout = {output_op, 0};
    outputs.push_back(output_opout);

    //Index<float> *index = create_euclidean(embeddings_dimension, nb_items, M, efConstruction, rand());
    Index<float> *index = load_euclidean(embeddings_dimension);

    // Creating session
    TF_SessionOptions* sess_opts = TF_NewSessionOptions();
    TF_Session* session = TF_NewSession(graph, sess_opts, status);
    assert(TF_GetCode(status) == TF_OK);

    // Create variables to store the size of the input and output variables
    const int input_size = dimension * nb_embeddings;
    const int num_bytes_in = input_size * sizeof(float);
    const int num_bytes_out = dimension * sizeof(float);

    int64_t in_dims[] = {nb_embeddings, dimension};
    int64_t out_dims[] = {dimension};

    for (int i = 0; i < nb_iterations; i++) {
        // Creating input
        std::vector<int> ids(nb_embeddings);
        set_random_ids(ids.data(), nb_embeddings, nb_items);
        auto start = now();
        std::vector<float> values(input_size);
        float* values_data = values.data();
        get_model_input(embeddings_dimension, extra_dimension, index, ids.data(), values_data, nb_embeddings);
        float get_model_input_microseconds = get_elepased_microseconds(start);

        start = now();
        TF_Tensor* input = TF_NewTensor(TF_FLOAT, in_dims, 2/*len(in_dims)*/, values_data, num_bytes_in, free_tensor, 0);
        std::vector<TF_Tensor*> input_values;
        input_values.push_back(input);
        float input_tensor_microseconds = get_elepased_microseconds(start);
        //std::cout << "Input data info: " << TF_Dim(input, 0) << "x"<< TF_Dim(input, 1) << "\n";

        // Allocating output
        start = now();
        TF_Tensor* output_value = TF_AllocateTensor(TF_FLOAT, out_dims, 1/*len(out_dims)*/, num_bytes_out);
        std::vector<TF_Tensor*> output_values(outputs.size(), nullptr);
        output_values.push_back(output_value);
        float output_tensor_microseconds = get_elepased_microseconds(start);
        //std::cout << "Output info: " << TF_Dim(output_value, 0) << "\n";

        // Running graph
        start = now();
        TF_SessionRun(session, nullptr,
                        &inputs[0], &input_values[0], inputs.size(),
                        &outputs[0], &output_values[0], outputs.size(),
                        nullptr, 0, nullptr, status);
        float run_session_microseconds = get_elepased_microseconds(start);

        start = now();
        float* out_vals = static_cast<float*>(TF_TensorData(output_values[0]));
        std::vector<float> query(embeddings_dimension);
        for(int i = 0; i < embeddings_dimension; i++) {
            query[i] = *out_vals++;
        }
        /*for (int i = 0; i < dimension; ++i) {
            std::cout << "Output values info: " << *out_vals++ << "\n";
        }*/
        float read_output_microseconds = get_elepased_microseconds(start);
        
        start = now();
        std::vector<float> distances(k);
        std::vector<size_t> items(k);
        index->knnQuery(query.data(), items.data(), distances.data(), k);
        float knn_microseconds = get_elepased_microseconds(start);
        /*for(int i = 0; i < k; i++) {
            std::cout << items[i] << " -> " << distances[i] << "\n";
        }*/
        float total_elapsed = get_model_input_microseconds + input_tensor_microseconds + output_tensor_microseconds + run_session_microseconds + read_output_microseconds + knn_microseconds;
        std::cout <<
            get_model_input_microseconds << ", " <<
            input_tensor_microseconds << ", " <<
            output_tensor_microseconds << ", " <<
            run_session_microseconds << ", " <<
            read_output_microseconds << ", " <<
            knn_microseconds << ", " <<
            total_elapsed << "\n";
    }

    fprintf(stdout, "Successfully run session\n");

    // Delete variables
    TF_CloseSession(session, status);
    TF_DeleteSession(session, status);
    TF_DeleteSessionOptions(sess_opts);
    TF_DeleteStatus(status);
    TF_DeleteGraph(graph);
    return 0;
}

TF_Graph *read_graph(const char *path, TF_Status *status) {
    TF_Buffer *graph_def = read_file(path);
    TF_Graph *graph = TF_NewGraph();
    TF_ImportGraphDefOptions *opts = TF_NewImportGraphDefOptions();
    TF_GraphImportGraphDef(graph, graph_def, opts, status);
    TF_DeleteImportGraphDefOptions(opts);
    assert(TF_GetCode(status) == TF_OK);
    std::cout << "Really Successfully imported graph\n";
    TF_DeleteBuffer(graph_def);
    return graph;
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
    fprintf(stdout, "Built index of %d items in: %.2fs\n", nbItems, get_elepased_seconds(start));
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
    fprintf(stdout, "Loaded index of %d items in: %.2fs\n", index->appr_alg->cur_element_count, get_elepased_seconds(start));
    return index;
}

float* get_model_input(int dimension, int extra_dimension, Index<float> * index, int* ids, float* result, int rows) {
    int cols = dimension + extra_dimension;
    // We need to allocate contiguous memory block to pass into TF_Tensor
    float* row_ptr = result;
    int* id_ptr = ids;
    for(int i = 0; i < rows; i++) {
        index->getDataPointerByLabel(*id_ptr, row_ptr);
        row_ptr += dimension;
        for(int j = dimension; j < cols; j++) {
            *row_ptr = *(row_ptr - dimension);
            row_ptr++;
        }
        //std::cout << "Input "<< i << " [" << *id_ptr << "] First = " << *(row_ptr-cols) << "; Last = " << *(row_ptr - 1) << " index = " << row_ptr - result<<"\n";
        id_ptr++;
    }
    //std::cout << "Input first="<< result[0] << " Last=" << result[rows*cols - 1] << "\n";
    return result;
}

void set_random_ids(int* ids, int len, int max_id) {
    for (int i = 0; i < len; i++) {
        ids[i] = get_random_id(max_id);
    }
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