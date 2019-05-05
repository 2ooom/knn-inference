package com.criteo.knn;

import com.criteo.hnsw.KnnResult;
import org.openjdk.jmh.annotations.*;
import com.criteo.hnsw.HnswIndex;

import java.io.IOException;
import java.util.concurrent.TimeUnit;

@State(Scope.Benchmark)
@Fork(warmups = 1, value = 1)
public class HnswBenchmarck extends BenchmarkBase {
    public HnswIndex index;

    @Setup(Level.Trial)
    public void setUp() {
        index = HnswIndex.create(metric, dimension);
        index.load("./benchmark-data/index-10k.hnsw");
        index.setEf(efSearch);
        System.out.println("Created knn service with efSearch = " + efSearch + "; dimension = " + dimension + "; nbItems = " + index.getNbItems());
    }

    @TearDown(Level.Trial)
    public void tearDown() {
        index.unload();
        System.out.println("Unloading index");
    }

    //@Benchmark
    @Fork(value = 1, warmups = 1)
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.MICROSECONDS)
    public KnnResult[] getNnsByRandomVector() {
        float[] query = getRandomVector(dimension);
        return index.knnQuery(query, k);
    }

    @Benchmark
    @Fork(value = 1, warmups = 1)
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.MICROSECONDS)
    public KnnResult[] getClosestItemsAvg() throws IOException {
        long[] ids = getIdsIterator().next();
        float[] query = new float[dimension];
        for(int i = 0; i < ids.length; i++) {
            float[] productEmbedding = index.getItem(ids[i]);
            for(int j = 0; j < productEmbedding.length; j++ ){
                query[j] += productEmbedding[j];
            }
        }
        for(int i = 0; i < dimension; i++) {
            query[i] /= nbItems;
        }
        return index.knnQuery(query, k);
    }

    //@Benchmark
    @Fork(value = 1, warmups = 1)
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.MICROSECONDS)
    public KnnResult[] getNnsByRandomItem() {
        float[] query = getItem();

        return index.knnQuery(query, k);
    }

    //@Benchmark
    @Fork(value = 1, warmups = 1)
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.MICROSECONDS)
    public float[] getItem() {
        return index.getItem(getRandomId());
    }
}
