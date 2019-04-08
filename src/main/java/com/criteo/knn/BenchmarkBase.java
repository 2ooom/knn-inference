package com.criteo.knn;

import com.criteo.hnsw.Metrics;

import java.util.function.Function;

public abstract class BenchmarkBase {
    public static int dimension = 100;
    public static int nbItems = 10000;
    public static float seedValue = 0.5f;

    public static int queryId = nbItems / 2;
    public static int k = 20;
    public static String metric = Metrics.Euclidean;
    public static Function<Integer, Float> getValueById = (id) -> seedValue / (id + 1);

    public static float[] getVector(int dimension, float value) {
        float[] vector = new float[dimension];
        for(int i = 0; i < dimension; i++) {
            vector[i] = value;
        }
        return vector;
    }
}
