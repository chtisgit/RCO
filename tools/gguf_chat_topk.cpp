// Phase 9 of RCO_PLAN_NEW.md (CUDA/offload checks): gguf_chat_kl.cpp with a
// device placement and a top-k dump.
//
// Placement: --gpu-layers N, --n-cpu-moe N (the expert tensors of the first N
// layers stay in CPU memory, as llama.cpp's --n-cpu-moe), and --device-none
// (no GPU device at all, as llama.cpp's --device none).  With --topk-out PATH
// the model's own top k at each scored position is written as a new input
// bundle with the same tokens and positions, so a second run against it
// reports KL(this placement || that placement).
//
// The rest is unchanged from gguf_chat_kl.cpp (Phase 8):
//
// Reads a bundle of conversations (token ids, the logit positions to score,
// and the reference's top-k token ids and log-probabilities at each of
// them), decodes each conversation alone from its token ids, and reports per
// scored position the NLL of the next token and KL(reference || model) over
// the reference's top k, not renormalized:
//
//     sum_j exp(v_j) * (v_j - log_softmax(logits)[i_j])
//
// which is ``top_k_kl`` of the PyTorch scorer.  One JSON line per
// conversation on stdout, with the per-position values.

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
#include <stdexcept>
#include <numeric>
#include <string>
#include <vector>

namespace {

constexpr char INPUT_MAGIC[8] = {'R', 'C', 'O', 'C', 'K', 'L', '1', '\0'};

struct Conversation {
    std::string id;
    std::vector<llama_token> tokens;
    std::vector<int32_t> positions;     // logit position t scores tokens[t + 1]
    std::vector<int32_t> reference_ids; // positions.size() x top_k
    std::vector<float> reference_log_probs;
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

template <typename T>
void read_vector(std::ifstream & input, std::vector<T> & values, size_t count) {
    values.resize(count);
    input.read(reinterpret_cast<char *>(values.data()),
               static_cast<std::streamsize>(count * sizeof(T)));
    if (!input) {
        throw std::runtime_error("truncated input-bundle vector");
    }
}

std::vector<Conversation> read_bundle(const std::string & path, uint32_t & top_k) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open input bundle: " + path);
    }
    char magic[8]{};
    input.read(magic, sizeof(magic));
    if (!input || std::memcmp(magic, INPUT_MAGIC, sizeof(magic)) != 0) {
        throw std::runtime_error("invalid input-bundle magic");
    }
    if (read_scalar<uint32_t>(input) != 1) {
        throw std::runtime_error("unsupported input-bundle version");
    }
    top_k = read_scalar<uint32_t>(input);
    const auto count = read_scalar<uint32_t>(input);
    if (top_k < 1) {
        throw std::runtime_error("top k must be positive");
    }
    std::vector<Conversation> conversations(count);
    for (auto & conversation : conversations) {
        const auto id_size = read_scalar<uint32_t>(input);
        conversation.id.resize(id_size);
        input.read(conversation.id.data(), id_size);
        read_vector(input, conversation.tokens, read_scalar<uint32_t>(input));
        const auto scored = read_scalar<uint32_t>(input);
        read_vector(input, conversation.positions, scored);
        read_vector(input, conversation.reference_ids, size_t(scored) * top_k);
        read_vector(input, conversation.reference_log_probs, size_t(scored) * top_k);
        for (size_t i = 0; i < conversation.positions.size(); ++i) {
            const int32_t position = conversation.positions[i];
            if (position < 0 || position + 1 >= static_cast<int32_t>(conversation.tokens.size())
                    || (i > 0 && position <= conversation.positions[i - 1])) {
                throw std::runtime_error("invalid scored positions in " + conversation.id);
            }
        }
    }
    if (input.peek() != std::ifstream::traits_type::eof()) {
        throw std::runtime_error("unexpected trailing input-bundle bytes");
    }
    return conversations;
}

std::string json_escape(const std::string & value) {
    std::string result;
    for (const unsigned char character : value) {
        if (character == '"' || character == '\\') {
            result += '\\';
            result += static_cast<char>(character);
        } else if (character < 0x20) {
            const char hex[] = "0123456789abcdef";
            result += "\\u00";
            result += hex[character >> 4];
            result += hex[character & 0x0f];
        } else {
            result += static_cast<char>(character);
        }
    }
    return result;
}

struct Arguments {
    std::string model;
    std::string input;
    int32_t gpu_layers = 0;
    int32_t threads = 4;
    int32_t ubatch = 64;
    int32_t start = 0;
    int32_t count = -1;
    int32_t n_cpu_moe = 0;
    bool device_none = false;
    std::string topk_out;
};

Arguments parse_arguments(int argc, char ** argv) {
    Arguments arguments;
    for (int index = 1; index < argc; ++index) {
        const std::string name = argv[index];
        if (name == "--device-none") {
            arguments.device_none = true;
            continue;
        }
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
        } else if (name == "--start") {
            arguments.start = std::stoi(value);
        } else if (name == "--count") {
            arguments.count = std::stoi(value);
        } else if (name == "--n-cpu-moe") {
            arguments.n_cpu_moe = std::stoi(value);
        } else if (name == "--topk-out") {
            arguments.topk_out = value;
        } else {
            throw std::runtime_error("unknown argument: " + name);
        }
    }
    if (arguments.model.empty() || arguments.input.empty()) {
        throw std::runtime_error("--model and --input are required");
    }
    if (arguments.gpu_layers < 0 || arguments.threads < 1 || arguments.ubatch < 1
            || arguments.start < 0 || arguments.count == 0 || arguments.n_cpu_moe < 0
            || (arguments.device_none && (arguments.gpu_layers > 0 || arguments.n_cpu_moe > 0))) {
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
        uint32_t top_k = 0;
        const auto conversations = read_bundle(arguments.input, top_k);
        const int32_t total = static_cast<int32_t>(conversations.size());
        if (arguments.start >= total) {
            throw std::runtime_error("start is outside the bundle");
        }
        const int32_t stop = arguments.count < 0
            ? total : std::min(total, arguments.start + arguments.count);
        int32_t maximum_tokens = 0;
        int32_t maximum_outputs = 1;
        for (int32_t i = arguments.start; i < stop; ++i) {
            maximum_tokens = std::max(maximum_tokens,
                                      static_cast<int32_t>(conversations[i].tokens.size()));
            maximum_outputs = std::max(maximum_outputs,
                                       static_cast<int32_t>(conversations[i].positions.size()));
        }

        llama_log_set([](enum ggml_log_level level, const char * text, void *) {
            // Buffer sizes show where the weights went (CPU, CUDA0, mapped).
            if (level >= GGML_LOG_LEVEL_ERROR || std::strstr(text, "buffer size") != nullptr) {
                std::cerr << text;
            }
        }, nullptr);
        ggml_backend_load_all();
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = arguments.gpu_layers;
        static ggml_backend_dev_t no_devices[] = {nullptr};
        if (arguments.device_none) {
            model_params.devices = no_devices;
        }
        // As llama.cpp common's llm_add_n_cpu_ffn_overrides(LLM_FFN_EXPS_REGEX).
        std::vector<std::string> override_patterns;
        for (int32_t layer = 0; layer < arguments.n_cpu_moe; ++layer) {
            override_patterns.push_back("blk\\." + std::to_string(layer)
                                        + "\\.ffn_(up|down|gate|gate_up)_(ch|)exps");
        }
        std::vector<llama_model_tensor_buft_override> overrides;
        for (const auto & pattern : override_patterns) {
            overrides.push_back({pattern.c_str(), ggml_backend_cpu_buffer_type()});
        }
        if (!overrides.empty()) {
            overrides.push_back({nullptr, nullptr});
            model_params.tensor_buft_overrides = overrides.data();
        }
        model = llama_model_load_from_file(arguments.model.c_str(), model_params);
        if (model == nullptr) {
            throw std::runtime_error("failed to load GGUF model");
        }
        const int32_t vocabulary_size = llama_vocab_n_tokens(llama_model_get_vocab(model));

        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = static_cast<uint32_t>(maximum_tokens);
        context_params.n_batch = static_cast<uint32_t>(maximum_tokens);
        context_params.n_ubatch = static_cast<uint32_t>(std::min(maximum_tokens, arguments.ubatch));
        context_params.n_seq_max = 1;
        context_params.n_threads = arguments.threads;
        context_params.n_threads_batch = arguments.threads;
        context_params.n_outputs_max = static_cast<uint32_t>(maximum_outputs);
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
                  << json_escape(description) << "\",\"vocabulary_size\":" << vocabulary_size
                  << ",\"gpu_layers\":" << arguments.gpu_layers
                  << ",\"n_cpu_moe\":" << arguments.n_cpu_moe
                  << ",\"device_none\":" << (arguments.device_none ? "true" : "false")
                  << ",\"maximum_tokens\":" << maximum_tokens << ",\"top_k\":" << top_k
                  << "}" << std::endl;

        std::vector<double> log_probs(vocabulary_size);
        std::ofstream topk;
        if (!arguments.topk_out.empty()) {
            topk.open(arguments.topk_out, std::ios::binary | std::ios::trunc);
            if (!topk) {
                throw std::runtime_error("cannot open top-k output: " + arguments.topk_out);
            }
            topk.write(INPUT_MAGIC, sizeof(INPUT_MAGIC));
            const uint32_t header[3] = {1u, top_k, static_cast<uint32_t>(stop - arguments.start)};
            topk.write(reinterpret_cast<const char *>(header), sizeof(header));
        }
        std::vector<int32_t> order(vocabulary_size);
        for (int32_t index = arguments.start; index < stop; ++index) {
            const Conversation & conversation = conversations[index];
            for (const int32_t id : conversation.reference_ids) {
                if (id < 0 || id >= vocabulary_size) {
                    throw std::runtime_error("reference token outside the vocabulary");
                }
            }
            llama_memory_clear(llama_get_memory(context), true);
            batch.n_tokens = static_cast<int32_t>(conversation.tokens.size());
            for (int32_t t = 0; t < batch.n_tokens; ++t) {
                batch.token[t] = conversation.tokens[t];
                batch.pos[t] = t;
                batch.n_seq_id[t] = 1;
                batch.seq_id[t][0] = 0;
                batch.logits[t] = false;
            }
            for (const int32_t position : conversation.positions) {
                batch.logits[position] = true;
            }
            const auto started = std::chrono::steady_clock::now();
            const int32_t status = llama_decode(context, batch);
            if (status != 0) {
                throw std::runtime_error("llama_decode failed with status " + std::to_string(status));
            }
            std::vector<double> kl(conversation.positions.size());
            std::vector<double> nll(conversation.positions.size());
            std::vector<int32_t> own_ids(conversation.positions.size() * top_k);
            std::vector<float> own_log_probs(own_ids.size());
            for (size_t i = 0; i < conversation.positions.size(); ++i) {
                const int32_t position = conversation.positions[i];
                const float * logits = llama_get_logits_ith(context, position);
                if (logits == nullptr) {
                    throw std::runtime_error("missing logits");
                }
                const float maximum = *std::max_element(logits, logits + vocabulary_size);
                double sum = 0.0;
                for (int32_t v = 0; v < vocabulary_size; ++v) {
                    sum += std::exp(static_cast<double>(logits[v] - maximum));
                }
                const double normalizer = std::log(sum) + static_cast<double>(maximum);
                nll[i] = normalizer - static_cast<double>(logits[conversation.tokens[position + 1]]);
                double divergence = 0.0;
                for (uint32_t j = 0; j < top_k; ++j) {
                    const size_t slot = i * top_k + j;
                    const double reference = conversation.reference_log_probs[slot];
                    const double model_log_prob =
                        static_cast<double>(logits[conversation.reference_ids[slot]]) - normalizer;
                    divergence += std::exp(reference) * (reference - model_log_prob);
                }
                kl[i] = divergence;
                if (topk.is_open()) {
                    std::iota(order.begin(), order.end(), 0);
                    std::partial_sort(order.begin(), order.begin() + top_k, order.end(),
                                      [logits](int32_t a, int32_t b) {
                                          return logits[a] > logits[b]
                                              || (logits[a] == logits[b] && a < b);
                                      });
                    for (uint32_t j = 0; j < top_k; ++j) {
                        own_ids[i * top_k + j] = order[j];
                        own_log_probs[i * top_k + j] = static_cast<float>(
                            static_cast<double>(logits[order[j]]) - normalizer);
                    }
                }
                if (!std::isfinite(kl[i]) || !std::isfinite(nll[i])) {
                    throw std::runtime_error("non-finite score in " + conversation.id);
                }
            }
            if (topk.is_open()) {
                const auto write_u32 = [&topk](uint32_t value) {
                    topk.write(reinterpret_cast<const char *>(&value), sizeof(value));
                };
                write_u32(static_cast<uint32_t>(conversation.id.size()));
                topk.write(conversation.id.data(), static_cast<std::streamsize>(conversation.id.size()));
                write_u32(static_cast<uint32_t>(conversation.tokens.size()));
                topk.write(reinterpret_cast<const char *>(conversation.tokens.data()),
                           static_cast<std::streamsize>(conversation.tokens.size() * sizeof(llama_token)));
                write_u32(static_cast<uint32_t>(conversation.positions.size()));
                topk.write(reinterpret_cast<const char *>(conversation.positions.data()),
                           static_cast<std::streamsize>(conversation.positions.size() * sizeof(int32_t)));
                topk.write(reinterpret_cast<const char *>(own_ids.data()),
                           static_cast<std::streamsize>(own_ids.size() * sizeof(int32_t)));
                topk.write(reinterpret_cast<const char *>(own_log_probs.data()),
                           static_cast<std::streamsize>(own_log_probs.size() * sizeof(float)));
                if (!topk) {
                    throw std::runtime_error("failed to write the top-k output");
                }
            }
            const double seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started).count();
            std::cout << std::setprecision(17) << "{\"type\":\"conversation\",\"index\":" << index
                      << ",\"id\":\"" << json_escape(conversation.id)
                      << "\",\"token_count\":" << batch.n_tokens
                      << ",\"scored\":" << conversation.positions.size()
                      << ",\"seconds\":" << seconds << ",\"kl\":[";
            for (size_t i = 0; i < kl.size(); ++i) {
                std::cout << (i ? "," : "") << kl[i];
            }
            std::cout << "],\"nll\":[";
            for (size_t i = 0; i < nll.size(); ++i) {
                std::cout << (i ? "," : "") << nll[i];
            }
            std::cout << "]}" << std::endl;
        }

        if (topk.is_open()) {
            topk.close();
            if (!topk) {
                throw std::runtime_error("failed to close the top-k output");
            }
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
        std::cerr << "gguf_chat_topk: " << error.what() << std::endl;
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
