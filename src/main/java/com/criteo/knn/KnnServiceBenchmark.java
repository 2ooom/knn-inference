package com.criteo.knn;

import org.openjdk.jmh.annotations.*;

import java.io.IOException;
import java.util.concurrent.TimeUnit;

@State(Scope.Benchmark)
@Fork(warmups = 1, value = 1)
public class KnnServiceBenchmark extends BenchmarkBase {

    public KnnService knn;
    public int efSearch = 50;
    public int indexId = 5;

    @Setup(Level.Trial)
    public void setUp() throws IOException {
        knn = new KnnService(distance, dimension, efSearch);
        knn.loadIndex(indexId,"./benchmark-data/index-10k.hnsw");
        System.out.println("Created knn service with efSearch = " + knn.efSearch+ "; dimension = " + knn.dimension + ";");
    }

    @TearDown(Level.Trial)
    public void tearDown() {
        knn.destroy();
        System.out.println("Destroyed knn service");
    }

    @Benchmark
    @Fork(value = 1, warmups = 1)
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.MICROSECONDS)
    public KnnResult[] getClosestItemsAvg() throws IOException {
        long[] ids = getIdsIterator().next();
        int[] indicesIds = new int[ids.length];

        for(int i = 0; i < ids.length; i++) {
            indicesIds[i] = indexId;
        }
        return knn.getClosestItems(indicesIds, ids, indexId, k, KnnModel.Average);
    }

}
