#include <assert.h>
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <sstream>
#include <iostream>
#include <tensorflow/c/c_api.h>
#include <algorithm>
#include <iterator>
#include <vector>
#include <cmath>
#include "common.h"
#include "tf.h"
#include "knn.h"

const char* path_to_model = "model/model-const.pb";
const char* input_node_name = "product_embeddings";
const char* output_node_name = "user_embeddings";

const int extra_dimension = 5;
const int dimension = embeddings_dimension + extra_dimension;
const int k = 20;

int main(int argc, char* argv[]) {
    char* output_path = argv[1];
    srand(random_seed);
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

    Index<float> *index = load_euclidean(embeddings_dimension, path_to_index);

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
    double avg_get_model = 0;
    double avg_fetch = 0;
    double avg_knn = 0;
    double avg_total = 0;
    std::fstream scenario_input(path_to_scenario, std::ios::in);
    std::fstream bench_output(output_path, std::ios::out);
    for (int i = 0; i < nb_iterations; i++) {
        // Creating input
        std::vector<int> ids(nb_embeddings);
        read_ids_from_file(ids.data(), nb_embeddings, scenario_input);

        auto start = now();
        std::vector<float> values(input_size);
        float* values_data = values.data();
        get_model_input(embeddings_dimension, extra_dimension, index, ids.data(), values_data, nb_embeddings);
        float get_model_input_microseconds = get_elepased_microseconds(start);
        avg_get_model += get_model_input_microseconds;

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
        avg_fetch += run_session_microseconds;

        start = now();
        float* out_vals = static_cast<float*>(TF_TensorData(output_values[0]));
        std::vector<float> query(embeddings_dimension);
        for(int i = 0; i < embeddings_dimension; i++) {
            query[i] = *out_vals++;
        }
        /*std::cout << "Output values info: \n";
        for (int j = 0; j < dimension; j++) {
            std::cout << query[j] << " ";
        }
        std::cout << "\n";*/
        float read_output_microseconds = get_elepased_microseconds(start);

        start = now();
        std::vector<float> distances(k);
        std::vector<size_t> items(k);
        index->knnQuery(query.data(), items.data(), distances.data(), k);
        float knn_microseconds = get_elepased_microseconds(start);
        avg_knn += knn_microseconds;
        /*for(int i = 0; i < k; i++) {
            std::cout << items[i] << " -> " << distances[i] << "\n";
        }*/
        float total_elapsed = get_model_input_microseconds + input_tensor_microseconds + output_tensor_microseconds + run_session_microseconds + read_output_microseconds + knn_microseconds;
        avg_total += total_elapsed;
        /*bench_output <<
            get_model_input_microseconds << " " <<
            input_tensor_microseconds << " " <<
            output_tensor_microseconds << " " <<
            run_session_microseconds << " " <<
            read_output_microseconds << " " <<
            knn_microseconds << " " <<
            total_elapsed << "\n";*/
    }
    scenario_input.close();
    bench_output.close();
    std::cout<<
        "Avg get model: " << avg_get_model/nb_iterations << "\n" <<
        "Avg tf_fetch: " << avg_fetch/nb_iterations << "\n" <<
        "Avg knn: " << avg_knn/nb_iterations << "\n" <<
        "Avg total: " << avg_total/nb_iterations << "\n";

    //fprintf(stdout, "Successfully run session\n");

    // Delete variables
    TF_CloseSession(session, status);
    TF_DeleteSession(session, status);
    TF_DeleteSessionOptions(sess_opts);
    TF_DeleteStatus(status);
    TF_DeleteGraph(graph);
    delete index;
    return 0;
}

void compute_average(int rows, int cols, float* input, float* result) {
    for(int i = 0; i < rows; i++) {
        double sum = 0;
        for(int j = 0; j < cols; j++) {
            sum += input[i*cols + j];
        }
        result[i] = (float)(sum / cols);
    }
}