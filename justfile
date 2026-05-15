set positional-arguments

cuda-library-path := env_var_or_default("CUDA_LIBRARY_PATH", "/usr/lib/x86_64-linux-gnu")
rustflags := env_var_or_default("RUSTFLAGS", "")
cuda-rustflags := if rustflags == "" { "-L native=" + cuda-library-path } else { rustflags + " -L native=" + cuda-library-path }

default:
    @just --list

# Run the CLI with the default Cargo feature set.
run *args:
    cargo run -- "$@"

# Run the CLI with both local CUDA backend feature flags enabled.
cuda *args:
    CUDA_LIBRARY_PATH="{{cuda-library-path}}" RUSTFLAGS="{{cuda-rustflags}}" cargo run --features "asr-whisper-cuda llm-llama-cpp-cuda" -- "$@"

# Build the default local stack.
build:
    cargo build

# Build with both local CUDA backend feature flags enabled.
build-cuda:
    CUDA_LIBRARY_PATH="{{cuda-library-path}}" RUSTFLAGS="{{cuda-rustflags}}" cargo build --features "asr-whisper-cuda llm-llama-cpp-cuda"

# Check the default local stack.
check:
    cargo check

# Check the CUDA feature path without building every default dependency.
check-cuda:
    CUDA_LIBRARY_PATH="{{cuda-library-path}}" RUSTFLAGS="{{cuda-rustflags}}" cargo check --no-default-features --features "asr-whisper-cuda llm-llama-cpp-cuda tts-piper model-download"

# Run the test suite.
test:
    cargo test

# Remove Cargo build artifacts.
clean:
    cargo clean
