#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <tensorflow/c/c_api.h>

TF_Buffer *read_file(const char *path);
TF_Graph *read_graph(const char *path, TF_Status *status);

void free_buffer(void *data, size_t length) {
    free(data);
}

void free_tensor(void* data, size_t len, void* arg) {
    free(data);
}

TF_Graph *read_graph(const char *path, TF_Status *status) {
    TF_Buffer *graph_def = read_file(path);
    TF_Graph *graph = TF_NewGraph();
    TF_ImportGraphDefOptions *opts = TF_NewImportGraphDefOptions();
    TF_GraphImportGraphDef(graph, graph_def, opts, status);
    TF_DeleteImportGraphDefOptions(opts);
    assert(TF_GetCode(status) == TF_OK);
    //std::cout << "Really Successfully imported graph\n";
    TF_DeleteBuffer(graph_def);
    return graph;
}

TF_Buffer *read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    void *data = malloc(fsize);
    fread(data, fsize, 1, f);
    fclose(f);

    TF_Buffer *buf = TF_NewBuffer();
    buf->data = data;
    buf->length = fsize;
    buf->data_deallocator = free_buffer;
    return buf;
}