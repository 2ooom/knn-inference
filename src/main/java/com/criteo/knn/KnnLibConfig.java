package com.criteo.knn;

import org.bytedeco.javacpp.annotation.*;
import org.bytedeco.javacpp.tools.*;

@Properties(
        value = @Platform(
                compiler = {"cpp11"},
                include = {"knn_api.h"}
        ),
        target = "com.criteo.knn",
        global = "com.criteo.knn.KnnLib"
)
public class KnnLibConfig implements InfoMapper {
    public void map(InfoMap infoMap) {
    }
}