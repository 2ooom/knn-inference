package com.criteo.knn;

import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

import java.io.IOException;

public class HelloKnn {
    public static void main(String[] args) throws IOException, RunnerException {
        //org.openjdk.jmh.Main.main(args);
        Options opt = new OptionsBuilder()
                .include(TfLineraTransforBenchmark.class.getSimpleName())
                .forks(1)
                .build();

        new Runner(opt).run();
    }
}
