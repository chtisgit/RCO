#include "ggml-backend.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr char INPUT_MAGIC[8] = {'R', 'C', 'O', 'N', 'L', 'L', '1', '\0'};

struct Document {
    std::string id;
    std::string text;
    std::vector<llama_token> tokens;
};

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

std::vector<Document> read_documents(const std::string & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open input bundle: " + path);
    }
    char magic[8]{};
    input.read(magic, sizeof(magic));
    if (!input || std::memcmp(magic, INPUT_MAGIC, sizeof(magic)) != 0) {
        throw std::runtime_error("invalid input-bundle magic");
    }
    const auto version = read_scalar<uint32_t>(input);
    if (version != 1) {
        throw std::runtime_error("unsupported input-bundle version");
    }
    const auto count = read_scalar<uint32_t>(input);
    std::vector<Document> documents;
    documents.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        Document document;
        document.id = read_string(input);
        document.text = read_string(input);
        const auto token_count = read_scalar<uint32_t>(input);
        document.tokens.resize(token_count);
        input.read(
            reinterpret_cast<char *>(document.tokens.data()),
            static_cast<std::streamsize>(token_count * sizeof(llama_token)));
        if (!input) {
            throw std::runtime_error("truncated input-bundle token vector");
        }
        documents.push_back(std::move(document));
    }
    if (input.peek() != std::ifstream::traits_type::eof()) {
        throw std::runtime_error("unexpected trailing input-bundle bytes");
    }
    return documents;
}

std::string json_escape(const std::string & value) {
    std::string result;
    result.reserve(value.size() + 8);
    for (const unsigned char character : value) {
        switch (character) {
            case '\\': result += "\\\\"; break;
            case '"': result += "\\\""; break;
            case '\b': result += "\\b"; break;
            case '\f': result += "\\f"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default:
                if (character < 0x20) {
                    const char hex[] = "0123456789abcdef";
                    result += "\\u00";
                    result += hex[character >> 4];
                    result += hex[character & 0x0f];
                } else {
                    result += static_cast<char>(character);
                }
        }
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
    const int32_t actual = llama_tokenize(
        vocab, text.data(), static_cast<int32_t>(text.size()),
        tokens.data(), count, false, false);
    if (actual != count) {
        throw std::runtime_error("llama.cpp tokenization failed");
    }
    return tokens;
}

double token_nll(
    const float * logits, int32_t vocabulary_size, llama_token target) {
    if (logits == nullptr || target < 0 || target >= vocabulary_size) {
        throw std::runtime_error("invalid logits or target token");
    }
    const float maximum = *std::max_element(logits, logits + vocabulary_size);
    double exponential_sum = 0.0;
    for (int32_t i = 0; i < vocabulary_size; ++i) {
        exponential_sum += std::exp(static_cast<double>(logits[i] - maximum));
    }
    return std::log(exponential_sum) + static_cast<double>(maximum)
        - static_cast<double>(logits[target]);
}

struct Arguments {
    std::string model;
    std::string input;
    int32_t gpu_layers = 0;
    int32_t threads = 4;
    int32_t ubatch = 64;
    int32_t start_document = 0;
    int32_t document_count = -1;
};

Arguments parse_arguments(int argc, char ** argv) {
    Arguments arguments;
    for (int index = 1; index < argc; ++index) {
        const std::string name = argv[index];
        if (index + 1 >= argc) {
            throw std::runtime_error("missing value for argument: " + name);
        }
        const std::string value = argv[++index];
        if (name == "--model") {
            arguments.model = value;
        } else if (name == "--input") {
            arguments.input = value;
        } else if (name == "--gpu-layers") {
            arguments.gpu_layers = std::stoi(value);
        } else if (name == "--threads") {
            arguments.threads = std::stoi(value);
        } else if (name == "--ubatch") {
            arguments.ubatch = std::stoi(value);
        } else if (name == "--start-document") {
            arguments.start_document = std::stoi(value);
        } else if (name == "--document-count") {
            arguments.document_count = std::stoi(value);
        } else {
            throw std::runtime_error("unknown argument: " + name);
        }
    }
    if (arguments.model.empty() || arguments.input.empty()) {
        throw std::runtime_error("--model and --input are required");
    }
    if (arguments.gpu_layers < 0 || arguments.threads < 1
            || arguments.ubatch < 1 || arguments.start_document < 0
            || arguments.document_count == 0) {
        throw std::runtime_error("invalid numeric argument");
    }
    return arguments;
}

}  // namespace

int main(int argc, char ** argv) {
    llama_model * model = nullptr;
    llama_context * context = nullptr;
    llama_batch batch{};
    bool batch_initialized = false;
    try {
        const Arguments arguments = parse_arguments(argc, argv);
        const auto documents = read_documents(arguments.input);
        if (arguments.start_document >= static_cast<int32_t>(documents.size())) {
            throw std::runtime_error("start document is outside the bundle");
        }
        const int32_t stop_document = arguments.document_count < 0
            ? static_cast<int32_t>(documents.size())
            : std::min(
                static_cast<int32_t>(documents.size()),
                arguments.start_document + arguments.document_count);
        int32_t maximum_tokens = 0;
        for (int32_t i = arguments.start_document; i < stop_document; ++i) {
            maximum_tokens = std::max(
                maximum_tokens,
                static_cast<int32_t>(documents[i].tokens.size()));
        }
        if (maximum_tokens < 2) {
            throw std::runtime_error("documents must contain at least two tokens");
        }

        llama_log_set([](enum ggml_log_level level, const char * text, void *) {
            if (level >= GGML_LOG_LEVEL_ERROR) {
                std::cerr << text;
            }
        }, nullptr);
        ggml_backend_load_all();
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = arguments.gpu_layers;
        model = llama_model_load_from_file(arguments.model.c_str(), model_params);
        if (model == nullptr) {
            throw std::runtime_error("failed to load GGUF model");
        }
        const llama_vocab * vocab = llama_model_get_vocab(model);
        const int32_t vocabulary_size = llama_vocab_n_tokens(vocab);

        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = static_cast<uint32_t>(maximum_tokens);
        context_params.n_batch = static_cast<uint32_t>(maximum_tokens);
        context_params.n_ubatch = static_cast<uint32_t>(
            std::min(maximum_tokens, arguments.ubatch));
        context_params.n_seq_max = 1;
        context_params.n_threads = arguments.threads;
        context_params.n_threads_batch = arguments.threads;
        context_params.n_outputs_max = static_cast<uint32_t>(maximum_tokens - 1);
        context_params.no_perf = false;
        context = llama_init_from_model(model, context_params);
        if (context == nullptr) {
            throw std::runtime_error("failed to initialize llama.cpp context");
        }
        batch = llama_batch_init(maximum_tokens, 0, 1);
        batch_initialized = true;

        char description[256]{};
        llama_model_desc(model, description, sizeof(description));
        std::cout << "{\"type\":\"environment\",\"description\":\""
                  << json_escape(description) << "\",\"vocabulary_size\":"
                  << vocabulary_size << ",\"gpu_layers\":"
                  << arguments.gpu_layers << ",\"maximum_tokens\":"
                  << maximum_tokens << "}" << std::endl;

        for (int32_t document_index = arguments.start_document;
                document_index < stop_document; ++document_index) {
            const Document & document = documents[document_index];
            const auto runtime_tokens = tokenize(vocab, document.text);
            if (runtime_tokens != document.tokens) {
                size_t mismatch = 0;
                while (mismatch < runtime_tokens.size()
                        && mismatch < document.tokens.size()
                        && runtime_tokens[mismatch] == document.tokens[mismatch]) {
                    ++mismatch;
                }
                throw std::runtime_error(
                    "token mismatch in " + document.id + " at index "
                    + std::to_string(mismatch));
            }

            llama_memory_clear(llama_get_memory(context), true);
            batch.n_tokens = static_cast<int32_t>(document.tokens.size());
            for (int32_t token_index = 0; token_index < batch.n_tokens;
                    ++token_index) {
                batch.token[token_index] = document.tokens[token_index];
                batch.pos[token_index] = token_index;
                batch.n_seq_id[token_index] = 1;
                batch.seq_id[token_index][0] = 0;
                batch.logits[token_index] = token_index + 1 < batch.n_tokens;
            }

            const auto started = std::chrono::steady_clock::now();
            const int32_t decode_status = llama_decode(context, batch);
            if (decode_status != 0) {
                throw std::runtime_error(
                    "llama_decode failed with status "
                    + std::to_string(decode_status));
            }
            double total_nll = 0.0;
            for (int32_t token_index = 0; token_index + 1 < batch.n_tokens;
                    ++token_index) {
                total_nll += token_nll(
                    llama_get_logits_ith(context, token_index),
                    vocabulary_size, document.tokens[token_index + 1]);
            }
            const auto stopped = std::chrono::steady_clock::now();
            const double seconds = std::chrono::duration<double>(
                stopped - started).count();
            const int32_t predicted_tokens = batch.n_tokens - 1;
            const double mean_nll = total_nll / predicted_tokens;
            if (!std::isfinite(mean_nll)) {
                throw std::runtime_error("non-finite document NLL");
            }
            std::cout << std::setprecision(17)
                      << "{\"type\":\"document\",\"index\":"
                      << document_index << ",\"id\":\""
                      << json_escape(document.id)
                      << "\",\"token_count\":" << batch.n_tokens
                      << ",\"predicted_token_count\":" << predicted_tokens
                      << ",\"nll_sum\":" << total_nll
                      << ",\"mean_nll\":" << mean_nll
                      << ",\"seconds\":" << seconds << "}" << std::endl;
        }

        llama_perf_context_print(context);
        llama_batch_free(batch);
        batch_initialized = false;
        llama_free(context);
        context = nullptr;
        llama_model_free(model);
        model = nullptr;
        llama_backend_free();
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "gguf_document_nll: " << error.what() << std::endl;
        if (batch_initialized) {
            llama_batch_free(batch);
        }
        if (context != nullptr) {
            llama_free(context);
        }
        if (model != nullptr) {
            llama_model_free(model);
        }
        llama_backend_free();
        return 1;
    }
}
