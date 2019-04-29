# Knn inferrence service

Realtime Service for building vectorized items representation with [TensorFlow](https://www.tensorflow.org/) and running kNN search in the provided
embedding space using [hnswlib](https://github.com/nmslib/hnswlib).

### KnnService comparison C++

The scenario included reading 50 random embeddings from the index, aggregating them and performing kNN Query

Parameters used for the index:
 * efContruction = 50
 * M = 16

Parameters used for querying:
 * efSearch = 50
 
```
------------------------------------------------------------------------
Benchmark                                         Time             CPU
------------------------------------------------------------------------
KnnServiceBenchmark/FullInference_mean         65.9 us         64.8 us
KnnServiceBenchmark/FullInference_median       66.1 us         64.9 us
KnnServiceBenchmark/FullInference_stddev       3.25 us         2.56 us

* repetitions:15000
* iterations:10
* threads:1

```
In order to launch it use `./bench-build-and-run.sh`

### KnnService comparison Java

```
Benchmark                               Mode  Cnt    Score    Error  Units
KnnServiceBenchmark.getClosestItemsAvg  avgt    5  103.625 ± 46.253  us/op
```
In order to launch it use `./gradlew run`