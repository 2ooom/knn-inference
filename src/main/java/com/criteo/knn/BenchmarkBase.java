package com.criteo.knn;

import com.criteo.hnsw.Metrics;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

public abstract class BenchmarkBase {
    public static int dimension = 100;
    public static int nbItems = 10000;
    public static float seedValue = 0.5f;
    public static String scenarioPath = "/Users/d.parfenchik/Dev/knn-inference/input-scenario-10k.csv";
    public static int k = 20;
    public static String metric = Metrics.Euclidean;
    public static Distance distance = Enum.valueOf(Distance.class, metric);

    public static Function<Integer, Float> getValueById = (id) -> seedValue / (id + 1);
    public static Random r = new Random();

    public static float[] getRandomVector(int dimension) {
        float[] vector = new float[dimension];
        for(int i = 0; i < dimension; i++) {
            vector[i] = r.nextFloat();
        }
        return vector;
    }

    public static long getRandomId() {
        return Math.abs(r.nextLong() % nbItems);
    }

    private List<long[]> ids;
    private Iterator<long[]> idsIterator;

    private static List<long[]> readLongs(String path) throws IOException {
        return Files.readAllLines(new File(path).toPath()).stream().map(line -> Arrays.stream(line.split(" ")).map(Long::parseLong).mapToLong(Long::longValue).toArray()).collect(Collectors.toList());
    }

    public Iterator<long[]> getIdsIterator() throws IOException {
        if(idsIterator == null || !idsIterator.hasNext()) {
            if(ids == null) {
                ids = readLongs(scenarioPath);
            }
            idsIterator = ids.iterator();
        }
        return idsIterator;
    }
}
