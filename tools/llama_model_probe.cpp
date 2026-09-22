#include "llama.h"

#include <cstdio>

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: llama-model-probe model.gguf\n");
        return 2;
    }
    llama_backend_init();
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = 0;
    params.check_tensors = true;
    llama_model * model = llama_model_load_from_file(argv[1], params);
    if (model == nullptr) {
        std::fprintf(stderr, "llama_model_load_from_file failed\n");
        llama_backend_free();
        return 1;
    }
    char description[256] = {};
    llama_model_desc(model, description, sizeof(description));
    std::printf(
        "{\"status\":\"pass\",\"description\":\"%s\","
        "\"embedding_length\":%d,\"layer_count\":%d,"
        "\"parameter_count\":%llu,\"model_bytes\":%llu}\n",
        description,
        llama_model_n_embd(model),
        llama_model_n_layer(model),
        static_cast<unsigned long long>(llama_model_n_params(model)),
        static_cast<unsigned long long>(llama_model_size(model)));
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
