package com.criteo.knn;

import com.criteo.hnsw.HnswIndex;
import com.criteo.hnsw.KnnResult;
import org.openjdk.jmh.annotations.*;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Random;
import java.util.concurrent.TimeUnit;

@State(Scope.Benchmark)
@Fork(warmups = 1, value = 1)
public class TfLinearTransformBenchmark  {

    public int nbProducts = 50;

    public int dimension = 100;
    public int extra_dimension = 5;
    public int result_dimension = dimension + extra_dimension;
    public Session session;
    public Graph graph;
    public Tensor constProductEmbeddingsTensor;
    public Random r = new Random();
    public float[][] productEmbeddings;
    public HnswIndex index;
    public float[] query;

    @Setup(Level.Trial)
    public void setUp() {
        HnswBenchmarck b = new HnswBenchmarck();
        b.setUp();
        index = b.index;

        setUpSession();
        query = index.getItem(BenchmarkBase.getRandomId());
        productEmbeddings = new float[nbProducts][];
        for(int i = 0; i < productEmbeddings.length; i++) {
            productEmbeddings[i] = BenchmarkBase.getRandomVector(result_dimension);
        }
        constProductEmbeddingsTensor = Tensor.create(productEmbeddings, Float.class);
    }

    public void setUpSession() {
        Path modelPath = Paths.get("/Users/d.parfenchik/Dev/knn-inference/model/model-const.pb");
        System.out.println("Reading Model");
        byte[] graphDef = new byte[0];
        try {
            graphDef = Files.readAllBytes(modelPath);
        } catch (IOException e) {
            e.printStackTrace();
        }
        graph = new Graph();
        graph.importGraphDef(graphDef);
        session = new Session(graph);
    }

    @TearDown(Level.Trial)
    public void tearDown() {
        session.close();
        graph.close();
        System.out.println("Unloading Graph and Session");
    }

    @Benchmark
    @Fork(value = 1, warmups = 1)
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.MICROSECONDS)
    public float[] indexLookupAndPredict() {
        float[][] productEmbeddings = new float[nbProducts][];
        for(int i = 0; i < nbProducts; i++) {
            productEmbeddings[i] = new float[result_dimension];
            float[] productEmbedding = index.getItem(BenchmarkBase.getRandomId());
            for(int j = 0; j < productEmbedding.length; j++ ){
                productEmbeddings[i][j] = productEmbedding[j];
            }
            for(int j = dimension; j < dimension + extra_dimension; j++ ) {
                productEmbeddings[i][j] = r.nextFloat();
            }
        }
        return predict(productEmbeddings);
    }

    @Benchmark
    @Fork(value = 1, warmups = 1)
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.MICROSECONDS)
    public KnnResult[] predictAndKnnFromRandomItems() {
        float[] query = indexLookupAndPredict();
        return index.knnQuery(query, BenchmarkBase.k);
    }

    @Benchmark
    @Fork(value = 1, warmups = 1)
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.MICROSECONDS)
    public KnnResult[] knn() {
        return index.knnQuery(query, BenchmarkBase.k);
    }

    @Benchmark
    @Fork(value = 1, warmups = 1)
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.MICROSECONDS)
    public float[] predictAndMaterialize() {
        return predict(productEmbeddings);
    }

    public float[] predict(float[][] input) {
        Tensor inputTensor = Tensor.create(input, Float.class);
        Tensor result = fetch(inputTensor);
        float[] outputBuffer = new float[dimension + extra_dimension];
        result.copyTo(outputBuffer);
        return outputBuffer;
    }


    @Benchmark
    @Fork(value = 1, warmups = 1)
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.MICROSECONDS)
    public Tensor createTensor() {
        return Tensor.create(productEmbeddings, Float.class);
    }

    @Benchmark
    @Fork(value = 1, warmups = 1)
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.MICROSECONDS)
    public void fetchOnly() {
        Tensor result = fetch(constProductEmbeddingsTensor);
    }

    public Tensor fetch(Tensor inputTensor) {
        return session.runner()
            .feed("product_embeddings", inputTensor)
            .fetch("user_embeddings").run().get(0);
    }
}
