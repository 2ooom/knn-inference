#include "knnservice.cpp"

extern "C" {
    long createAngular(int dim, const int ef_search) {
        return (long)new KnnService(Angular, dim, ef_search);
    }

    long createEuclidean(int dim, int ef_search) {
        return (long)new KnnService(Euclidian, dim, ef_search);
    }

    long createInnerProduct(int dim, int ef_search) {
        return (long)new KnnService(InnerProduct, dim, ef_search);
    }

    void destroy(long knn) {
        delete ((KnnService *)knn);
    }

    void loadIndex(long knn, int index_id, const std::string &path_to_index) {
        ((KnnService *)knn)->loadIndex(index_id, path_to_index);
    }

    size_t getClosestItemsAvg(long knn, int * index_ids, size_t * query_labels, size_t nb_items, size_t query_index, size_t * result_items, float * result_distances, size_t k) {
        return ((KnnService *)knn)->getClosestItems(index_ids, query_labels, nb_items, query_index, result_items, result_distances, k, Average);
    }
}
