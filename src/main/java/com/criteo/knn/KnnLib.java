// Targeted by JavaCPP version 1.4.4: DO NOT EDIT THIS FILE

package com.criteo.knn;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

public class KnnLib extends com.criteo.knn.KnnLibConfig {
    static { Loader.load(); }

// Parsed from knn_api.h

// #include "knnservice.cpp"
    public static native long createAngular(int dim, int ef_search);

    public static native long createEuclidean(int dim, int ef_search);

    public static native long createInnerProduct(int dim, int ef_search);

    public static native void destroy(long knn);

    public static native void loadIndex(long knn, int index_id, @StdString BytePointer path_to_index);
    public static native void loadIndex(long knn, int index_id, @StdString String path_to_index);

    public static native @Cast("size_t") long getClosestItemsAvg(long knn, IntPointer index_ids, @Cast("size_t*") SizeTPointer query_labels, @Cast("size_t") long nb_items, @Cast("size_t") long query_index, @Cast("size_t*") SizeTPointer result_items, FloatPointer result_distances, @Cast("size_t") long k);
    public static native @Cast("size_t") long getClosestItemsAvg(long knn, IntBuffer index_ids, @Cast("size_t*") SizeTPointer query_labels, @Cast("size_t") long nb_items, @Cast("size_t") long query_index, @Cast("size_t*") SizeTPointer result_items, FloatBuffer result_distances, @Cast("size_t") long k);
    public static native @Cast("size_t") long getClosestItemsAvg(long knn, int[] index_ids, @Cast("size_t*") SizeTPointer query_labels, @Cast("size_t") long nb_items, @Cast("size_t") long query_index, @Cast("size_t*") SizeTPointer result_items, float[] result_distances, @Cast("size_t") long k);


}
