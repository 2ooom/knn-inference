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

 - repetitions:15000
 - iterations:10
 - threads:1
```
In order to launch it use `./bench-build-and-run.sh`

### KnnService comparison Java

```
Benchmark                               Mode  Cnt   Score   Error  Units
KnnServiceBenchmark.getClosestItemsAvg  avgt    5  83.311 ± 3.501  us/op

Result "com.criteo.knn.KnnServiceBenchmark.getClosestItemsAvg":
  83.311 ±(99.9%) 3.501 us/op [Average]
  (min, avg, max) = (82.665, 83.311, 84.894), stdev = 0.909
  CI (99.9%): [79.810, 86.811] (assumes normal distribution)
```
In order to launch it use `./gradlew run`