package com.criteo.knn;

import com.criteo.hnsw.KnnResult;
import org.openjdk.jmh.annotations.*;
import com.criteo.hnsw.HnswIndex;
import java.util.concurrent.TimeUnit;

@State(Scope.Benchmark)
@Fork(warmups = 1, value = 1)
public class HnswBenchmarck extends BenchmarkBase {

    public int M = 16;
    public int efConstruction = 200;
    public HnswIndex index;

    @Setup(Level.Trial)
    public void setUp() {
        index = HnswIndex.create(metric, dimension);
        index.initNewIndex(nbItems, M, efConstruction);
        for (int i = 0; i < nbItems; i++) {
            float[] vector = getRandomVector(dimension);
            index.addItem(vector, i);
        }
        System.out.println("Created index for " + index.getNbItems() + " items");
    }

    @TearDown(Level.Trial)
    public void tearDown() {
        index.unload();
        System.out.println("Unloading index");
    }

    @Benchmark
    @Fork(value = 1, warmups = 1)
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.MICROSECONDS)
    public void getNnsByRandomVector() {
        float[] query = getRandomVector(dimension);
        KnnResult[] results = index.knnQuery(query, k);
    }

    @Benchmark
    @Fork(value = 1, warmups = 1)
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.MICROSECONDS)
    public void getNnsByRandomItem() {
        long id = Math.abs(r.nextLong() % nbItems);
        float[] query = index.getItem(id);

        KnnResult[] results = index.knnQuery(query, k);
    }

    @Benchmark
    @Fork(value = 1, warmups = 1)
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.MICROSECONDS)
    public void getItem() {
        long id = Math.abs(r.nextLong() % nbItems);
        float[] item = index.getItem(id);
    }
}
