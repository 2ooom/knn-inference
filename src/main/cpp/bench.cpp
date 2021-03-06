#include <assert.h>
#include <benchmark/benchmark.h>
#include <vector>
#include "common.h"
#include "tf.h"
#include "knnservice.cpp"

#define nb_repetitions 10

Index<float> *load_euclidean(int dimension, const std::string &path, int efSearch) {
    Index<float> * index = new Index<float>(Euclidian, dimension);
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

class TfAttentionModel : public benchmark::Fixture {
    public: const char* path_to_model = "model/model-const.pb";
    public: const char* input_node_name = "product_embeddings";
    public: const char* output_node_name = "user_embeddings";
    public: const int efSearch = 50;
    public: const int extra_dimension = 5;
    public: const int dimension = embeddings_dimension + extra_dimension;

    public: const int input_size = dimension * nb_embeddings;
    public: const int num_bytes_in = input_size * sizeof(float);
    public: const int num_bytes_out = dimension * sizeof(float);

    public: const int k = 20;

    public: TF_Status *status;
    public: TF_Graph *graph;

    public: std::vector<TF_Output> inputs;
    public: std::vector<TF_Output> outputs;

    public: Index<float> *index;

    public: TF_SessionOptions* sess_opts;
    public: TF_Session* session;

    public: const int64_t in_dims[2] = {nb_embeddings, dimension};
    public: const int64_t out_dims[1] = {dimension};

    public: std::vector<int> scenario_input;
    public: std::vector<int>::iterator scenario_input_it;

    public:
    void SetUp(const ::benchmark::State& state) {
        //std::cout<<"\nInitializing up\n";
        status = TF_NewStatus();
        graph = read_graph(path_to_model, status);

        index = load_euclidean(embeddings_dimension, path_to_index, efSearch);

        // Creating session
        sess_opts = TF_NewSessionOptions();
        session = TF_NewSession(graph, sess_opts, status);
        assert(TF_GetCode(status) == TF_OK);
        
        // Configure input
        inputs.clear();
        TF_Operation* input_op = TF_GraphOperationByName(graph, input_node_name);
        TF_Output input_opout = {input_op, 0};
        inputs.push_back(input_opout);

        // Configure output
        outputs.clear();
        TF_Operation* output_op = TF_GraphOperationByName(graph, output_node_name);
        TF_Output output_opout = {output_op, 0};
        outputs.push_back(output_opout);

        std::fstream scenario_input_file = std::fstream(path_to_scenario, std::ios::in);
        //std::vector<int> scenario_input(nb_embeddings * nb_iterations);
        scenario_input.clear();
        int id;
        while (scenario_input_file >> id) {
            scenario_input.push_back(id);
        }
        //std::cout<<"Read " << scenario_input.size() << "\n";
        scenario_input_it = scenario_input.begin();
        scenario_input_file.close();
    }

    void TearDown(const ::benchmark::State& state) {
        //std::cout<<"\nCleaning up\n";
        TF_CloseSession(session, status);
        TF_DeleteSession(session, status);
        TF_DeleteSessionOptions(sess_opts);
        TF_DeleteStatus(status);
        TF_DeleteGraph(graph);
        delete index;
    }

    public:
    void read_ids_from_it(int* ids, int len) {
        for(int i = 0; i < len; i++) {
            ids[i] = *scenario_input_it;
            scenario_input_it++;
            //std::cout << ids[i] << " ";
        }
        //std::cout << "\n";
    }
};

BENCHMARK_DEFINE_F(TfAttentionModel, FullTest)(benchmark::State& st) {
    for (auto _ : st) {
        //std::cout<<".";
        std::vector<int> ids(nb_embeddings);
        read_ids_from_it(ids.data(), nb_embeddings);

        std::vector<float> values(input_size);
        float* values_data = values.data();
        get_model_input(embeddings_dimension, extra_dimension, index, ids.data(), values_data, nb_embeddings);
        //std::cout<<"In dim: " << in_dims[0] <<"x"<< in_dims[1] << "\n";
        TF_Tensor* input = TF_NewTensor(TF_FLOAT, in_dims, 2/*len(in_dims)*/, values_data, num_bytes_in, free_tensor, 0);
        std::vector<TF_Tensor*> input_values;
        input_values.push_back(input);

        //std::cout<<"Out dim: " << out_dims[0] <<"\n";
        TF_Tensor* output_value = TF_AllocateTensor(TF_FLOAT, out_dims, 1/*len(out_dims)*/, num_bytes_out);
        std::vector<TF_Tensor*> output_values(outputs.size(), nullptr);
        output_values.push_back(output_value);
        //std::cout<<"Before running..." << "num_bytes_in=" << num_bytes_in <<"; input operAddr="<<inputs[0].oper<<"; last="<<values_data[input_size - 1]<<"\n";
        TF_SessionRun(session, nullptr,
                        &inputs[0], &input_values[0], inputs.size(),
                        &outputs[0], &output_values[0], outputs.size(),
                        nullptr, 0, nullptr, status);

        //std::cout<<"After running...\n";
        float* out_vals = static_cast<float*>(TF_TensorData(output_values[0]));
        std::vector<float> query(embeddings_dimension);
        for(int i = 0; i < embeddings_dimension; i++) {
            query[i] = *out_vals++;
        }
        /*std::cout << "Output values info: \n";
        for (int i = 0; i < dimension; i++) {
            std::cout << query[i] << " ";
        }
        std::cout << "\n";*/
        std::vector<float> distances(k);
        std::vector<size_t> items(k);
        index->knnQuery(query.data(), items.data(), distances.data(), k);
    }
}

BENCHMARK_DEFINE_F(TfAttentionModel, IndexLookup)(benchmark::State& st) {
    for (auto _ : st) {
        std::vector<int> ids(nb_embeddings);
        read_ids_from_it(ids.data(), nb_embeddings);

        std::vector<float> values(dimension * nb_embeddings);
        float* values_data = values.data();
        get_model_input(embeddings_dimension, extra_dimension, index, ids.data(), values_data, nb_embeddings);
    }
}


BENCHMARK_REGISTER_F(TfAttentionModel, FullTest)->
    Threads(1)->
    Repetitions(nb_repetitions)->
    Unit(benchmark::kMicrosecond)->
    DisplayAggregatesOnly(true)->
    Iterations(nb_iterations);

BENCHMARK_REGISTER_F(TfAttentionModel, IndexLookup)->
    Threads(1)->
    Repetitions(nb_repetitions)->
    Unit(benchmark::kMicrosecond)->
    DisplayAggregatesOnly(true)->
    Iterations(nb_iterations);


class KnnServiceBenchmark : public benchmark::Fixture {
public:
    const int efSearch = 50;
    const int k = 20;
    const int index_id = 5;
    KnnService* knn_service;
    std::vector<int> index_ids;
    std::vector<int> scenario_input;
    std::vector<int>::iterator scenario_input_it;

    void SetUp(const ::benchmark::State& state) {
        //std::cout<<"\nKnnServiceBenchmark Initializing\n";
        knn_service = new KnnService(Euclidian, embeddings_dimension, efSearch);
        knn_service->loadIndex(index_id, path_to_index);

        for(int i = 0; i < nb_embeddings; i++) {
            index_ids.push_back(index_id);
        }

        std::fstream scenario_input_file = std::fstream(path_to_scenario, std::ios::in);
        scenario_input.clear();
        int id;
        while (scenario_input_file >> id) {
            scenario_input.push_back(id);
        }
        //std::cout<<"Read " << scenario_input.size() << "\n";
        scenario_input_it = scenario_input.begin();
        scenario_input_file.close();
    }

    void TearDown(const ::benchmark::State& state) {
        //std::cout<<"\nCleaning up\n";
        delete knn_service;
    }

    void read_ids_from_it(size_t* ids, int len) {
        for(int i = 0; i < len; i++) {
            ids[i] = *scenario_input_it;
            scenario_input_it++;
            //std::cout << ids[i] << " ";
        }
        //std::cout << "\n";
    }

};


BENCHMARK_DEFINE_F(KnnServiceBenchmark, FullInference)(benchmark::State& st) {
    for (auto _ : st) {
        std::vector<size_t> ids(nb_embeddings);
        read_ids_from_it(ids.data(), nb_embeddings);

        std::vector<float> result_distances(k);
        std::vector<size_t> result_items(k);
        auto nb_items = knn_service->getClosestItems(index_ids.data(), ids.data(), nb_embeddings, index_id, result_items.data(), result_distances.data(), k, Average);
        //std::cout<<"Retrieved " << nb_items << " vectors. Closest = " << result_items[0] << "; Distance = " << result_distances[0] << "\n";
    }
}

BENCHMARK_REGISTER_F(KnnServiceBenchmark, FullInference)->
    Threads(1)->
    Repetitions(nb_repetitions)->
    Unit(benchmark::kMicrosecond)->
    DisplayAggregatesOnly(true)->
    Iterations(nb_iterations);

BENCHMARK_MAIN();