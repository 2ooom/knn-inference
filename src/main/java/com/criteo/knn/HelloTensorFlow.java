package com.criteo.knn;

import org.tensorflow.*;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class HelloTensorFlow {
    private static int nbProducts = 50;
    private static int dimension = 105;

    public static void main(String[] args) throws Exception {

        Path modelPath = Paths.get("/Users/d.parfenchik/Dev/knn-inference/model/model-const.pb");
        byte[] graph = Files.readAllBytes(modelPath);
        try (Graph g = new Graph()) {
            g.importGraphDef(graph);
            float[][] productEmbeddings = new float[nbProducts][];
            for(int i = 0; i < productEmbeddings.length; i++) {
                productEmbeddings[i] = BenchmarkBase.getRandomVector(dimension);
            }
            Tensor productEmbeddingsTensor = Tensor.create(productEmbeddings, Float.class);
            try (Session sess = new Session(g)) {
                float[] userEmbedding = predict(sess, productEmbeddingsTensor);
                for (float v : userEmbedding) {
                    System.out.println(v);
                }
            }
        }
    }

    private static float[] predict(Session sess, Tensor inputTensor) {
        Tensor result = sess.runner()
                .feed("product_embeddings", inputTensor)
                .fetch("user_embeddings").run().get(0);
        float[] outputBuffer = new float[dimension];
        result.copyTo(outputBuffer);
        return outputBuffer;
    }
}