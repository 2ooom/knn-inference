package com.criteo.knn;

import annoy4s.Annoy;
import annoy4s.AnnoyLibrary;
import com.sun.jna.Pointer;
import org.openjdk.jmh.annotations.*;

import java.util.concurrent.TimeUnit;

@State(Scope.Benchmark)
@Fork(warmups = 1, value = 1)
public class Annoy4sBenchmark extends BenchmarkBase {
    public int numTrees = 32;
    public Pointer index;
    AnnoyLibrary annoyLib;

    @Setup(Level.Trial)
    public void setUp() {
        annoyLib = Annoy.annoyLib();
        index = annoyLib.createEuclidean(dimension);
        for (int i = 0; i < nbItems; i++) {
            float[] vector = getRandomVector(dimension);
            annoyLib.addItem(index, i, vector);
        }

        annoyLib.build(index, numTrees);

        System.out.println("Created Annoy4s index for " + annoyLib.getNItems(index) + " items");
    }

    @TearDown(Level.Trial)
    public void tearDown() {
        annoyLib.deleteIndex(index);
        System.out.println("Unloading Annoy4s index");
    }


    @Benchmark
    @Fork(value = 1, warmups = 1)
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.MICROSECONDS)
    public void knnByRandomVector() {
        float[] query = getRandomVector(dimension);
        int[] results = new int[k];
        float[] distances = new float[k];
        annoyLib.getNnsByVector(index, query, k, -1, results, distances);
    }

    @Benchmark
    @Fork(value = 1, warmups = 1)
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.MICROSECONDS)
    public void getNnsByRandomItem() {
        int id = Math.abs(r.nextInt() % nbItems);
        float[] query = new float[dimension];
        annoyLib.getItem(index, id, query);
        int[] results = new int[k];
        float[] distances = new float[k];
        annoyLib.getNnsByVector(index, query, k, -1, results, distances);
    }

    @Benchmark
    @Fork(value = 1, warmups = 1)
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.MICROSECONDS)
    public void getItem() {
        int id = Math.abs(r.nextInt() % nbItems);
        float[] vector = new float[dimension];
        annoyLib.getItem(index, id, vector);
    }

    @Benchmark
    @Fork(value = 1, warmups = 1)
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.MICROSECONDS)
    public void getRandomVector() {
        float[] vector = getRandomVector(dimension);
    }
}
