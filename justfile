set positional-arguments

default:
    @just --list

# Run the CLI with the default Cargo feature set.
run *args:
    cargo run -- "$@"

# Run the CLI with both local CUDA backend feature flags enabled.
cuda *args:
    cargo run --features "asr-whisper-cuda llm-llama-cpp-cuda" -- "$@"

# Build the default local stack.
build:
    cargo build

# Build with both local CUDA backend feature flags enabled.
build-cuda:
    cargo build --features "asr-whisper-cuda llm-llama-cpp-cuda"

# Check the default local stack.
check:
    cargo check

# Check the CUDA feature path without building every default dependency.
check-cuda:
    cargo check --no-default-features --features "asr-whisper-cuda llm-llama-cpp-cuda tts-piper model-download"

# Run the test suite.
test:
    cargo test

# Remove Cargo build artifacts.
clean:
    cargo clean
