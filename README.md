# Knn inferrence service

Realtime Service for building vectorized items representation with [TensorFlow](https://www.tensorflow.org/) and running kNN search in the provided
embedding space using [hnswlib](https://github.com/nmslib/hnswlib).

### KnnService comparison C++
```
-------------------------------------------------------------------------------------------------------------------------
Benchmark                                                                               Time             CPU   Iterations
-------------------------------------------------------------------------------------------------------------------------
KnnServiceBenchmark/FullInference/iterations:15000/repeats:10/threads:1_mean         65.9 us         64.8 us           10
KnnServiceBenchmark/FullInference/iterations:15000/repeats:10/threads:1_median       66.1 us         64.9 us           10
KnnServiceBenchmark/FullInference/iterations:15000/repeats:10/threads:1_stddev       3.25 us         2.56 us           10
```
In order to launc it use `./bench-build-and-run.sh`

### KnnService comparison Java

```
Benchmark                               Mode  Cnt    Score    Error  Units
KnnServiceBenchmark.getClosestItemsAvg  avgt    5  103.625 ± 46.253  us/op
```
In order to launc it use `./gradlew run`