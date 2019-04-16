#include <iostream>
#include "hnswlib.h"

template<typename dist_t, typename data_t=float>
class Index {
public:
    Index(hnswlib::SpaceInterface<float> *space, const int dim, bool normalize = false) :
            space(space), dim(dim), normalize(normalize) {
        appr_alg = NULL;
    }

    void initNewIndex(const size_t maxElements, const size_t M, const size_t efConstruction, const size_t random_seed) {
        if (appr_alg) {
            throw new std::runtime_error("The index is already initiated.");
        }
        appr_alg = new hnswlib::HierarchicalNSW<dist_t>(space, maxElements, M, efConstruction, random_seed);
    }

    void saveIndex(const std::string &path_to_index) {
        appr_alg->saveIndex(path_to_index);
    }

    void loadIndex(const std::string &path_to_index) {
        if (appr_alg) {
            std::cerr<<"Warning: Calling load_index for an already inited index. Old index is being deallocated.";
            delete appr_alg;
        }
        appr_alg = new hnswlib::HierarchicalNSW<dist_t>(space, path_to_index, false, 0);
    }

    void normalizeVector(dist_t *data, dist_t *norm_array){
        dist_t norm=0.0f;
        for(int i=0;i<dim;i++)
            norm+=data[i]*data[i];
        norm= 1.0f / (sqrtf(norm) + 1e-30f);
        for(int i=0;i<dim;i++)
            norm_array[i]=data[i]*norm;
    }

    void addItem(dist_t * vector, size_t id) {
        dist_t* vector_data = vector;
        std::vector<dist_t> norm_array(dim);
        if(normalize) {                    
            normalizeVector(vector_data, norm_array.data());
            vector_data = norm_array.data();
        }
        appr_alg->addPoint(vector_data, (size_t) id);
    }

    void getDataPointerByLabel(size_t label, data_t* dst) {
        hnswlib::tableint label_c;
        auto search = appr_alg->label_lookup_.find(label);
        if (search == appr_alg->label_lookup_.end()) {
            throw std::runtime_error("Label not found");
        }
        label_c = search->second;
        data_t* data_ptr = (data_t*)appr_alg->getDataByInternalId(label_c);
        memcpy(dst, data_ptr, appr_alg->data_size_);
    }

    std::vector<size_t> getIdsList() {
        std::vector<size_t> ids;

        for(auto kv : appr_alg->label_lookup_) {
            ids.push_back(kv.first);
        }
        return ids;
    }

    size_t knnQuery(dist_t * vector, size_t * items, dist_t * distances, size_t k) {
        dist_t* vector_data = vector;
        std::vector<dist_t> norm_array(dim);
        if(normalize) {                    
            normalizeVector(vector_data, norm_array.data());
            vector_data = norm_array.data();
        }

        std::priority_queue<std::pair<dist_t, hnswlib::labeltype >> result = appr_alg->searchKnn(
                (void *) vector_data, k);
        size_t nbResults = result.size();
        if (nbResults != k)
            std::cout << "Retrieved " << nbResults << " items instead of " <<
                k << ". Items in the index: " << appr_alg->cur_element_count;

        for (int i = nbResults - 1; i >= 0; i--) {
            auto &result_tuple = result.top();
            distances[i] = result_tuple.first;
            items[i] = (size_t)result_tuple.second;
            result.pop();
        }
        return nbResults;
    }

    hnswlib::SpaceInterface<float> *space;
    int dim;
    bool normalize;
    hnswlib::HierarchicalNSW<dist_t> *appr_alg;

    ~Index() {
        delete space;
        if (appr_alg)
            delete appr_alg;
    }
};

extern "C" {
    long createAngular(int dim) {
        hnswlib::SpaceInterface<float> *space = new hnswlib::InnerProductSpace(dim);
        bool normalize = true;
        return (long)new Index<float>(space, dim, normalize);
    }

    long createEuclidean(int dim) {
        hnswlib::SpaceInterface<float> *space = new hnswlib::L2Space(dim);
        bool normalize=false;
        return (long)new Index<float>(space, dim, normalize);
    }

    long createDotProduct(int dim) {
        hnswlib::SpaceInterface<float> *space = new hnswlib::InnerProductSpace(dim);
        bool normalize=false;
        return (long)new Index<float>(space, dim, normalize);
    }

    void destroyIndex(long index) {
        delete ((Index<float> *)index);
    }

    void initNewIndex(long index, const size_t maxElements, const size_t M, const size_t efConstruction, const size_t random_seed) {
        ((Index<float> *)index)->initNewIndex(maxElements, M, efConstruction, random_seed);
    }

    void setEf(long index, size_t ef) {
        ((Index<float> *)index)->appr_alg->ef_ = ef;
    }

    void saveIndex(long index, const std::string &path_to_index) {
        ((Index<float> *)index)->saveIndex(path_to_index);
    }

    void loadIndex(long index, const std::string &path_to_index) {
        ((Index<float> *)index)->loadIndex(path_to_index);
    }

    void addItem(long index, float * vector, size_t id) {
        ((Index<float> *)index)->addItem(vector, id);
    }

    std::vector<float> getItem(long index, size_t id) {
        return ((Index<float> *)index)->appr_alg->template getDataByLabel<float>(id);
    }

    std::vector<size_t> getIdsList(long index) {
        return ((Index<float> *)index)->getIdsList();
    }

    size_t getNbItems(long index) {
        return ((Index<float> *)index)->appr_alg->cur_element_count;
    }

    size_t knnQuery(long index, float * vector, size_t * items, float * distances, size_t k) {
        //auto start = std::chrono::high_resolution_clock::now();
        auto nbItems = ((Index<float> *)index)->knnQuery(vector, items, distances, k);
        //auto elapsed = std::chrono::high_resolution_clock::now() - start;
        //long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
        //std::cout << "query time: " << microseconds << " us" << "\n";
        return nbItems;
    }
}
