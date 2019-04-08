package com.criteo.knn;

import com.criteo.annoy.AnnoyIndex;
import org.openjdk.jmh.annotations.*;

import java.util.List;
import java.util.concurrent.TimeUnit;


@State(Scope.Benchmark)
@Fork(warmups = 1, value = 1)
public class AnnoyBenchmark extends BenchmarkBase {
    public int numTrees = 32;
    public AnnoyIndex index;

    public float[] query;

    @Setup(Level.Trial)
    public void setUp() {
        index = AnnoyIndex.create(metric, dimension);
        float[][] items = new float[nbItems][];
        long[] ids = new long[nbItems];
        for (int i = 0; i < nbItems; i++) {
            float[] vector = getVector(dimension, getValueById.apply(i));
            items[i] = vector;
            ids[i] = i;
        }
        index.build(items, ids, numTrees);
        query = getVector(dimension, getValueById.apply(queryId));
        System.out.println("Created Annoy index for " + index.getNItems() + " items");
    }

    @TearDown(Level.Trial)
    public void tearDown() {
        index.unload();
        System.out.println("Unloading Annoy index");
    }


    @Benchmark
    @Fork(value = 1, warmups = 1)
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.MICROSECONDS)
    public void hnsw() {
        List<com.criteo.annoy.KnnResult> results = index.getClosestVectors(query, k, -1);
    }
}
