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
    public KnnResult[] getNnsByRandomVector() {
        float[] query = getRandomVector(dimension);
        return index.knnQuery(query, k);
    }

    @Benchmark
    @Fork(value = 1, warmups = 1)
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.MICROSECONDS)
    public KnnResult[] aggregateItemsAndKnn() {
        int nbProducts = 50;
        float[] query = new float[dimension];
        for(int i = 0; i < nbProducts; i++) {
            float[] productEmbedding = index.getItem(getRandomId());
            for(int j = 0; j < productEmbedding.length; j++ ){
                query[j] += productEmbedding[j];
            }
        }
        for(int i = 0; i < dimension; i++) {
            query[i] /= nbItems;
        }
        return index.knnQuery(query, k);
    }

    @Benchmark
    @Fork(value = 1, warmups = 1)
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.MICROSECONDS)
    public KnnResult[] getNnsByRandomItem() {
        float[] query = getItem();

        return index.knnQuery(query, k);
    }

    @Benchmark
    @Fork(value = 1, warmups = 1)
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.MICROSECONDS)
    public float[] getItem() {
        return index.getItem(getRandomId());
    }
}
