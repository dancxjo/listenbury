set positional-arguments

cuda-library-path := env_var_or_default("CUDA_LIBRARY_PATH", "/usr/lib/x86_64-linux-gnu")
rustflags := env_var_or_default("RUSTFLAGS", "")
cuda-rustflags := if rustflags == "" { "-L native=" + cuda-library-path } else { rustflags + " -L native=" + cuda-library-path }
cmudict-url := "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict"
cmudict-path := "data/cmudict.dict"
mbrola-voices-url := "https://raw.githubusercontent.com/numediart/MBROLA-voices/master/data"

default:
    @just --list

# Run the CLI with the default Cargo feature set.
run *args:
    cargo run -- "$@"

# Speak through the Riper MBROLA backend. Run `just fetch` first for the default us3/en1 voices.
say-mbrola text:
    cargo run -- say --riper --mbrola "{{text}}"

# Render the ragtime singing demo through the Riper MBROLA probe backend.
sing-mbrola:
    cargo run -- sing --riper --mbrola

# Start the live PETE listening loop.
listen *args:
    cargo run -- listen "$@"

# Download the full CMU Pronouncing Dictionary into data/cmudict.dict.
fetch:
    @mkdir -p "$(dirname "{{cmudict-path}}")"
    @tmp="$(mktemp "{{cmudict-path}}.XXXXXX")" && curl --fail --location --show-error --output "$tmp" "{{cmudict-url}}" && mv "$tmp" "{{cmudict-path}}"
    @for voice in us3 en1; do mkdir -p "data/mbrola/$voice"; tmp="$(mktemp "data/mbrola/$voice/$voice.XXXXXX")"; curl --fail --location --show-error --output "$tmp" "{{mbrola-voices-url}}/$voice/$voice"; mv "$tmp" "data/mbrola/$voice/$voice"; done

# Download the default model assets into LISTENBURY_HOME.
fetch-models *args:
    cargo run -- models fetch "$@"

# Run the CLI with both local CUDA backend feature flags enabled.
cuda *args:
    CUDA_LIBRARY_PATH="{{cuda-library-path}}" RUSTFLAGS="{{cuda-rustflags}}" cargo run --features "asr-whisper-cuda llm-llama-cpp-cuda" -- "$@"

# Build the default local stack.
build:
    cargo build

# Install local build and scan tooling.
setup:
    @cargo audit --version >/dev/null 2>&1 || cargo install --locked cargo-audit

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

# Run the fast CI mirror: formatting, lints, short smoke tests, and audit.
rescan:
    cargo fmt -- --check
    cargo clippy --all-targets -- -D warnings
    LISTENBURY_SHORT=1 cargo test --no-default-features --test pipeline_smoke --release -- --nocapture
    just audit

# Run the full local release integration suite.
rescan-full:
    cargo fmt -- --check
    cargo clippy --all-targets -- -D warnings
    cargo test --all --tests --release -- --nocapture
    just audit

# Run the security audit, installing cargo-audit if needed.
audit:
    @cargo audit --version >/dev/null 2>&1 || cargo install --locked cargo-audit
    cargo audit

# Remove Cargo build artifacts.
clean:
    cargo clean
