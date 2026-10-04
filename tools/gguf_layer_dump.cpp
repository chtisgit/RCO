#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"

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
constexpr char OUTPUT_MAGIC[8] = {'R', 'C', 'O', 'L', 'A', 'Y', '1', '\0'};

template <typename T>
T read_scalar(std::ifstream & input) {
    T value{};
    input.read(reinterpret_cast<char *>(&value), sizeof(value));
    if (!input) throw std::runtime_error("truncated input bundle");
    return value;
}

std::string read_string(std::ifstream & input) {
    const auto size = read_scalar<uint32_t>(input);
    std::string value(size, '\0');
    input.read(value.data(), size);
    if (!input) throw std::runtime_error("truncated input-bundle string");
    return value;
}

struct Input {
    std::string text;
    std::vector<llama_token> tokens;
};

Input read_input(const std::string & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open input bundle: " + path);
    char magic[8]{};
    input.read(magic, sizeof(magic));
    if (!input || std::memcmp(magic, INPUT_MAGIC, sizeof(magic)) != 0) {
        throw std::runtime_error("invalid input-bundle magic");
    }
    if (read_scalar<uint32_t>(input) != 1 || read_scalar<uint32_t>(input) != 1) {
        throw std::runtime_error("layer dump requires a version-1 one-document bundle");
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
    if (result.tokens.empty()) throw std::runtime_error("tokens are required");
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

struct Captures {
    int32_t expected_layers = 0;
    int32_t token_count = 0;
    int32_t hidden_size = 0;
    std::vector<std::vector<float>> values;
    std::string error;
};

int capture_index(const char * name, int32_t expected_layers) {
    if (std::strcmp(name, "model.input_embed") == 0) return 0;
    constexpr char prefix[] = "l_out-";
    if (std::strncmp(name, prefix, sizeof(prefix) - 1) != 0) return -1;
    try {
        const int layer = std::stoi(name + sizeof(prefix) - 1);
        return 0 <= layer && layer < expected_layers ? layer + 1 : -1;
    } catch (...) {
        return -1;
    }
}

float read_float(const uint8_t * data, ggml_type type, size_t offset) {
    if (type == GGML_TYPE_F32) {
        return *reinterpret_cast<const float *>(data + offset);
    }
    if (type == GGML_TYPE_F16) {
        return ggml_fp16_to_fp32(
            *reinterpret_cast<const ggml_fp16_t *>(data + offset));
    }
    if (type == GGML_TYPE_BF16) {
        return ggml_bf16_to_fp32(
            *reinterpret_cast<const ggml_bf16_t *>(data + offset));
    }
    throw std::runtime_error("captured layer tensor has unsupported type");
}

bool capture_callback(ggml_tensor * tensor, bool ask, void * user_data) {
    auto * captures = static_cast<Captures *>(user_data);
    const int index = capture_index(tensor->name, captures->expected_layers);
    if (ask) return index >= 0;
    if (index < 0 || !captures->error.empty()) return true;
    try {
        const int32_t captured_tokens = static_cast<int32_t>(tensor->ne[1]);
        if (captured_tokens < 1 || captured_tokens > captures->token_count
                || tensor->ne[2] != 1 || tensor->ne[3] != 1) {
            throw std::runtime_error("captured layer tensor has unexpected shape");
        }
        if (captures->hidden_size == 0) {
            captures->hidden_size = static_cast<int32_t>(tensor->ne[0]);
        }
        if (tensor->ne[0] != captures->hidden_size) {
            throw std::runtime_error("captured hidden width changed");
        }
        std::vector<uint8_t> copied;
        const uint8_t * data = nullptr;
        if (ggml_backend_buffer_is_host(tensor->buffer)) {
            data = static_cast<const uint8_t *>(tensor->data);
        } else {
            copied.resize(ggml_nbytes(tensor));
            ggml_backend_tensor_get(tensor, copied.data(), 0, copied.size());
            data = copied.data();
        }
        auto & output = captures->values.at(index);
        const size_t prior_size = output.size();
        output.resize(prior_size + static_cast<size_t>(captured_tokens)
                      * captures->hidden_size);
        for (int32_t token = 0; token < captured_tokens; ++token) {
            for (int32_t hidden = 0; hidden < captures->hidden_size; ++hidden) {
                output[prior_size + static_cast<size_t>(token)
                       * captures->hidden_size + hidden] =
                    read_float(data, tensor->type,
                               static_cast<size_t>(token) * tensor->nb[1]
                               + static_cast<size_t>(hidden) * tensor->nb[0]);
            }
        }
    } catch (const std::exception & error) {
        captures->error = error.what();
    }
    return true;
}

struct Arguments {
    std::string model;
    std::string input;
    std::string output;
    int32_t expected_layers = 40;
    int32_t gpu_layers = 0;
    int32_t threads = 4;
    int32_t ubatch = 64;
};

Arguments parse_arguments(int argc, char ** argv) {
    Arguments result;
    for (int index = 1; index < argc; ++index) {
        const std::string name = argv[index];
        if (index + 1 >= argc) throw std::runtime_error("missing argument value");
        const std::string value = argv[++index];
        if (name == "--model") result.model = value;
        else if (name == "--input") result.input = value;
        else if (name == "--output") result.output = value;
        else if (name == "--expected-layers") result.expected_layers = std::stoi(value);
        else if (name == "--gpu-layers") result.gpu_layers = std::stoi(value);
        else if (name == "--threads") result.threads = std::stoi(value);
        else if (name == "--ubatch") result.ubatch = std::stoi(value);
        else throw std::runtime_error("unknown argument: " + name);
    }
    if (result.model.empty() || result.input.empty() || result.output.empty()) {
        throw std::runtime_error("--model, --input, and --output are required");
    }
    if (result.expected_layers < 1 || result.gpu_layers < 0
            || result.threads < 1 || result.ubatch < 1) {
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
        Captures captures;
        captures.expected_layers = arguments.expected_layers;
        captures.token_count = static_cast<int32_t>(input.tokens.size());
        captures.values.resize(static_cast<size_t>(arguments.expected_layers) + 1);
        auto context_params = llama_context_default_params();
        context_params.n_ctx = input.tokens.size();
        context_params.n_batch = input.tokens.size();
        context_params.n_ubatch = std::min<int32_t>(
            captures.token_count, arguments.ubatch);
        context_params.n_seq_max = 1;
        context_params.n_threads = arguments.threads;
        context_params.n_threads_batch = arguments.threads;
        context_params.cb_eval = capture_callback;
        context_params.cb_eval_user_data = &captures;
        context = llama_init_from_model(model, context_params);
        if (context == nullptr) throw std::runtime_error("failed to initialize context");
        batch = llama_batch_init(captures.token_count, 0, 1);
        batch_initialized = true;
        batch.n_tokens = captures.token_count;
        for (int32_t index = 0; index < captures.token_count; ++index) {
            batch.token[index] = input.tokens[index];
            batch.pos[index] = index;
            batch.n_seq_id[index] = 1;
            batch.seq_id[index][0] = 0;
            batch.logits[index] = index + 1 == captures.token_count;
        }
        if (llama_decode(context, batch) != 0) {
            throw std::runtime_error("llama_decode failed");
        }
        if (!captures.error.empty()) throw std::runtime_error(captures.error);
        for (size_t index = 0; index < captures.values.size(); ++index) {
            if (captures.values[index].size()
                    != static_cast<size_t>(captures.token_count) * captures.hidden_size) {
                throw std::runtime_error(
                    "missing captured state at index " + std::to_string(index));
            }
        }
        std::ofstream output(arguments.output, std::ios::binary | std::ios::trunc);
        if (!output) throw std::runtime_error("cannot open output file");
        output.write(OUTPUT_MAGIC, sizeof(OUTPUT_MAGIC));
        const uint32_t version = 1;
        const uint32_t count = captures.values.size();
        const uint32_t tokens = captures.token_count;
        const uint32_t hidden = captures.hidden_size;
        output.write(reinterpret_cast<const char *>(&version), sizeof(version));
        output.write(reinterpret_cast<const char *>(&count), sizeof(count));
        output.write(reinterpret_cast<const char *>(&tokens), sizeof(tokens));
        output.write(reinterpret_cast<const char *>(&hidden), sizeof(hidden));
        for (uint32_t index = 0; index < count; ++index) {
            output.write(reinterpret_cast<const char *>(&index), sizeof(index));
            output.write(
                reinterpret_cast<const char *>(captures.values[index].data()),
                static_cast<std::streamsize>(captures.values[index].size() * sizeof(float)));
        }
        if (!output) throw std::runtime_error("failed while writing layer states");
        std::cout << "{\"capture_count\":" << count
                  << ",\"token_count\":" << tokens
                  << ",\"hidden_size\":" << hidden << "}\n";
        llama_batch_free(batch);
        batch_initialized = false;
        llama_free(context);
        context = nullptr;
        llama_model_free(model);
        model = nullptr;
        llama_backend_free();
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "gguf_layer_dump: " << error.what() << '\n';
        if (batch_initialized) llama_batch_free(batch);
        if (context != nullptr) llama_free(context);
        if (model != nullptr) llama_model_free(model);
        llama_backend_free();
        return 1;
    }
}
