# Knn inferrence service

Realtime Service for building vectorized items representation with [TensorFlow](https://www.tensorflow.org/) and running kNN search in the provided
embedding space using [hnswlib](https://github.com/nmslib/hnswlib).

## Benchmarks

Knn Performance comparison:
```
Benchmark         Mode  Cnt    Score    Error  Units
----------------------------------------------------
Annoy4s           avgt    5  297.294 ± 34.408  us/op
Annoy-jni         avgt    5   69.152 ±  6.432  us/op
Hnsw-jni          avgt    5   13.996 ±  2.307  us/op
```
