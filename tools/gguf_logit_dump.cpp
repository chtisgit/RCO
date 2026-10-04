#include "ggml-backend.h"
#include "llama.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr char INPUT_MAGIC[8] = {'R', 'C', 'O', 'N', 'L', 'L', '1', '\0'};
constexpr char OUTPUT_MAGIC[8] = {'R', 'C', 'O', 'L', 'O', 'G', '1', '\0'};

template <typename T>
T read_scalar(std::ifstream & input) {
    T value{};
    input.read(reinterpret_cast<char *>(&value), sizeof(value));
    if (!input) {
        throw std::runtime_error("truncated input bundle");
    }
    return value;
}

std::string read_string(std::ifstream & input) {
    const auto size = read_scalar<uint32_t>(input);
    std::string value(size, '\0');
    input.read(value.data(), size);
    if (!input) {
        throw std::runtime_error("truncated input-bundle string");
    }
    return value;
}

struct Input {
    std::string text;
    std::vector<llama_token> tokens;
};

Input read_input(const std::string & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open input bundle: " + path);
    }
    char magic[8]{};
    input.read(magic, sizeof(magic));
    if (!input || std::memcmp(magic, INPUT_MAGIC, sizeof(magic)) != 0) {
        throw std::runtime_error("invalid input-bundle magic");
    }
    if (read_scalar<uint32_t>(input) != 1 || read_scalar<uint32_t>(input) != 1) {
        throw std::runtime_error("logit dump requires a version-1 one-document bundle");
    }
    (void) read_string(input);
    Input result;
    result.text = read_string(input);
    const auto count = read_scalar<uint32_t>(input);
    result.tokens.resize(count);
    input.read(
        reinterpret_cast<char *>(result.tokens.data()),
        static_cast<std::streamsize>(count * sizeof(llama_token)));
    if (!input || input.peek() != std::ifstream::traits_type::eof()) {
        throw std::runtime_error("invalid trailing input-bundle data");
    }
    if (result.tokens.size() < 2) {
        throw std::runtime_error("at least two tokens are required");
    }
    return result;
}

std::vector<llama_token> tokenize(
    const llama_vocab * vocab, const std::string & text) {
    const int32_t required = llama_tokenize(
        vocab, text.data(), static_cast<int32_t>(text.size()),
        nullptr, 0, false, false);
    if (required == std::numeric_limits<int32_t>::min()) {
        throw std::runtime_error("token count overflow");
    }
    const int32_t count = required < 0 ? -required : required;
    std::vector<llama_token> tokens(count);
    if (llama_tokenize(
            vocab, text.data(), static_cast<int32_t>(text.size()),
            tokens.data(), count, false, false) != count) {
        throw std::runtime_error("llama.cpp tokenization failed");
    }
    return tokens;
}

struct Arguments {
    std::string model;
    std::string input;
    std::string output;
    int32_t gpu_layers = 0;
    int32_t threads = 4;
    int32_t ubatch = 64;
};

Arguments parse_arguments(int argc, char ** argv) {
    Arguments result;
    for (int index = 1; index < argc; ++index) {
        const std::string name = argv[index];
        if (index + 1 >= argc) {
            throw std::runtime_error("missing value for argument: " + name);
        }
        const std::string value = argv[++index];
        if (name == "--model") result.model = value;
        else if (name == "--input") result.input = value;
        else if (name == "--output") result.output = value;
        else if (name == "--gpu-layers") result.gpu_layers = std::stoi(value);
        else if (name == "--threads") result.threads = std::stoi(value);
        else if (name == "--ubatch") result.ubatch = std::stoi(value);
        else throw std::runtime_error("unknown argument: " + name);
    }
    if (result.model.empty() || result.input.empty() || result.output.empty()) {
        throw std::runtime_error("--model, --input, and --output are required");
    }
    if (result.gpu_layers < 0 || result.threads < 1 || result.ubatch < 1) {
        throw std::runtime_error("invalid execution parameter");
    }
    return result;
}

}  // namespace

int main(int argc, char ** argv) {
    llama_model * model = nullptr;
    llama_context * context = nullptr;
    llama_batch batch{};
    bool batch_initialized = false;
    try {
        const auto arguments = parse_arguments(argc, argv);
        const auto input = read_input(arguments.input);
        llama_log_set([](enum ggml_log_level level, const char * text, void *) {
            if (level >= GGML_LOG_LEVEL_ERROR) std::cerr << text;
        }, nullptr);
        ggml_backend_load_all();
        auto model_params = llama_model_default_params();
        model_params.n_gpu_layers = arguments.gpu_layers;
        model = llama_model_load_from_file(arguments.model.c_str(), model_params);
        if (model == nullptr) throw std::runtime_error("failed to load GGUF model");
        const llama_vocab * vocab = llama_model_get_vocab(model);
        if (tokenize(vocab, input.text) != input.tokens) {
            throw std::runtime_error("llama.cpp and bundle tokenization differ");
        }
        const int32_t token_count = static_cast<int32_t>(input.tokens.size());
        const int32_t vocabulary_size = llama_vocab_n_tokens(vocab);
        auto context_params = llama_context_default_params();
        context_params.n_ctx = static_cast<uint32_t>(token_count);
        context_params.n_batch = static_cast<uint32_t>(token_count);
        context_params.n_ubatch = static_cast<uint32_t>(
            std::min(token_count, arguments.ubatch));
        context_params.n_seq_max = 1;
        context_params.n_threads = arguments.threads;
        context_params.n_threads_batch = arguments.threads;
        context_params.n_outputs_max = static_cast<uint32_t>(token_count - 1);
        context = llama_init_from_model(model, context_params);
        if (context == nullptr) throw std::runtime_error("failed to initialize context");
        batch = llama_batch_init(token_count, 0, 1);
        batch_initialized = true;
        batch.n_tokens = token_count;
        for (int32_t index = 0; index < token_count; ++index) {
            batch.token[index] = input.tokens[index];
            batch.pos[index] = index;
            batch.n_seq_id[index] = 1;
            batch.seq_id[index][0] = 0;
            batch.logits[index] = index + 1 < token_count;
        }
        if (llama_decode(context, batch) != 0) {
            throw std::runtime_error("llama_decode failed");
        }
        std::ofstream output(arguments.output, std::ios::binary | std::ios::trunc);
        if (!output) throw std::runtime_error("cannot open output file");
        output.write(OUTPUT_MAGIC, sizeof(OUTPUT_MAGIC));
        const uint32_t version = 1;
        const uint32_t positions = static_cast<uint32_t>(token_count - 1);
        const uint32_t vocabulary = static_cast<uint32_t>(vocabulary_size);
        output.write(reinterpret_cast<const char *>(&version), sizeof(version));
        output.write(reinterpret_cast<const char *>(&positions), sizeof(positions));
        output.write(reinterpret_cast<const char *>(&vocabulary), sizeof(vocabulary));
        output.write(
            reinterpret_cast<const char *>(input.tokens.data()),
            static_cast<std::streamsize>(input.tokens.size() * sizeof(llama_token)));
        for (int32_t index = 0; index < token_count - 1; ++index) {
            const float * logits = llama_get_logits_ith(context, index);
            if (logits == nullptr) throw std::runtime_error("missing output logits");
            output.write(
                reinterpret_cast<const char *>(logits),
                static_cast<std::streamsize>(vocabulary_size * sizeof(float)));
        }
        if (!output) throw std::runtime_error("failed while writing logits");
        std::cout << "{\"token_count\":" << token_count
                  << ",\"position_count\":" << positions
                  << ",\"vocabulary_size\":" << vocabulary_size << "}\n";
        llama_batch_free(batch);
        batch_initialized = false;
        llama_free(context);
        context = nullptr;
        llama_model_free(model);
        model = nullptr;
        llama_backend_free();
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "gguf_logit_dump: " << error.what() << '\n';
        if (batch_initialized) llama_batch_free(batch);
        if (context != nullptr) llama_free(context);
        if (model != nullptr) llama_model_free(model);
        llama_backend_free();
        return 1;
    }
}
