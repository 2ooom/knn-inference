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
import java.util.concurrent.TimeUnit;

@State(Scope.Benchmark)
@Fork(warmups = 1, value = 1)
public class TfLineraTransforBenchmark {

    public int nbProducts = 50;
    public int dimension = 105;
    public Session session;
    public Graph graph;
    public Tensor productEmbeddingsTensor;

    @Setup(Level.Trial)
    public void setUp() throws IOException {
        Path modelPath = Paths.get("/Users/d.parfenchik/Dev/knn-inference/model/model-const.pb");
        System.out.println("Reading Model");
        byte[] graphDef = Files.readAllBytes(modelPath);
        graph = new Graph();
        graph.importGraphDef(graphDef);
        session = new Session(graph);
        float[][] productEmbeddings = new float[nbProducts][];
        for(int i = 0; i < productEmbeddings.length; i++) {
            productEmbeddings[i] = BenchmarkBase.getRandomVector(dimension);
        }
        productEmbeddingsTensor = Tensor.create(productEmbeddings, Float.class);
    }

    @TearDown(Level.Trial)
    public void tearDown() {
        session.close();
        graph.close();
        System.out.println("Unloading Graph and Session");
    }

    //@Benchmark
    @Fork(value = 1, warmups = 1)
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.MICROSECONDS)
    public void inferenceWithInput() {
        float[][] productEmbeddings = new float[nbProducts][];
        for(int i = 0; i < productEmbeddings.length; i++) {
            productEmbeddings[i] = BenchmarkBase.getRandomVector(dimension);
        }
        Tensor productEmbeddingsTensor = Tensor.create(productEmbeddings, Float.class);
        Tensor result = session.runner()
                .feed("product_embeddings", productEmbeddingsTensor)
                .fetch("user_embeddings").run().get(0);
        float[] outputBuffer = new float[dimension];
        result.copyTo(outputBuffer);
    }

    @Benchmark
    @Fork(value = 1, warmups = 1)
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.MICROSECONDS)
    public void inferenceOnly() {
        Tensor result = session.runner()
                .feed("product_embeddings", productEmbeddingsTensor)
                .fetch("user_embeddings").run().get(0);
    }
}
