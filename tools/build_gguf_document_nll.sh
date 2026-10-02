#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 LLAMA_CPP_DIR LLAMA_BUILD_DIR OUTPUT" >&2
    exit 2
fi

llama_cpp_dir=$(realpath "$1")
llama_build_dir=$(realpath "$2")
output=$(realpath -m "$3")
source_dir=$(realpath "$(dirname "$0")")

mkdir -p "$(dirname "$output")"
g++ -std=c++17 -O2 \
    -I "$llama_cpp_dir/include" \
    -I "$llama_cpp_dir/ggml/include" \
    "$source_dir/gguf_document_nll.cpp" \
    -L "$llama_build_dir/bin" \
    -Wl,-rpath,"$llama_build_dir/bin" \
    -l:libllama.so.0.4.1 -l:libggml.so.0.24.0 \
    -o "$output"
