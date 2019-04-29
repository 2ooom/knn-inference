package com.criteo.knn;

import org.bytedeco.javacpp.SizeTPointer;

public class KnnService {
    public final int dimension;
    public final int efSearch;
    public final Distance distance;
    private long _knn;

    public KnnService(Distance distance, int dimension, int efSearch) {
        this.dimension = dimension;
        this.distance = distance;
        this.efSearch = efSearch;
        if (distance == Distance.Euclidean) {
            _knn = KnnLib.createEuclidean(dimension, efSearch);
        } else if (distance == Distance.Angular) {
            _knn = KnnLib.createAngular(dimension, efSearch);
        } else if (distance == Distance.InnerProduct) {
            _knn = KnnLib.createInnerProduct(dimension, efSearch);
        } else {
            throw new UnsupportedOperationException("Unknown distance metric: " + distance);
        }
    }

    public void loadIndex(int indexId, String pathToIndex) {
        KnnLib.loadIndex(_knn, indexId, pathToIndex);
    }

    public KnnResult[] getClosestItems(int[] indicesIds, long[] labels, int queryIndexId, int k, KnnModel model) {
        if (indicesIds.length != labels.length) {
            throw new RuntimeException("Labels array and indices array have different size (indices=" + indicesIds.length + "; labels=" + labels.length + ")");
        }
        SizeTPointer resultLabels = new SizeTPointer(k);
        float[] distances = new float[k];
        SizeTPointer labelsPointer = new SizeTPointer(labels.length);
        for (int i = 0; i < labels.length; i++) {
            labelsPointer.put(i, labels[i]);
        }
        long resultSize;
        if (model == KnnModel.Average) {
            resultSize = KnnLib.getClosestItemsAvg(_knn, indicesIds, labelsPointer, (long)labels.length, (long)queryIndexId, resultLabels, distances, k);
        } else {
            throw new UnsupportedOperationException("Unknown knn model: " + model);
        }
        KnnResult[] result = new KnnResult[k];
        for (int i = 0; i < resultSize; i++) {
            result[i] = new KnnResult(resultLabels.get(i), distances[i]);
        }
        return result;
    }

    public void destroy() {
        KnnLib.destroy(_knn);
    }
}
