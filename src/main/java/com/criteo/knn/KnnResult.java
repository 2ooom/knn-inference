package com.criteo.knn;

public class KnnResult {
    public final long item;
    public final float distance;

    public KnnResult(long item, float distance) {
        this.item = item;
        this.distance = distance;
    }
}
