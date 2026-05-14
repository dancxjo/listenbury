pub struct ModelAsset {
    pub id: &'static str,
    pub filename: &'static str,
    pub relative_path: &'static str,
    pub url: &'static str,
    pub expected_size_hint: Option<u64>,
}

pub const DEFAULT_MODELS: &[ModelAsset] = &[
    ModelAsset {
        id: "whisper-tiny-en",
        filename: "ggml-tiny.en.bin",
        relative_path: "models/whisper/ggml-tiny.en.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin",
        expected_size_hint: None,
    },
    ModelAsset {
        id: "tinyllama-q4-k-m",
        filename: "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        relative_path: "models/llama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        url: "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        expected_size_hint: None,
    },
    ModelAsset {
        id: "piper-lessac-medium",
        filename: "en_US-lessac-medium.onnx",
        relative_path: "models/piper/en_US-lessac-medium.onnx",
        url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
        expected_size_hint: None,
    },
    ModelAsset {
        id: "piper-lessac-medium-config",
        filename: "en_US-lessac-medium.onnx.json",
        relative_path: "models/piper/en_US-lessac-medium.onnx.json",
        url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
        expected_size_hint: None,
    },
];
