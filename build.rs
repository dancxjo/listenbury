fn main() {
    let whisper_enabled = std::env::var_os("CARGO_FEATURE_ASR_WHISPER").is_some();
    let llama_enabled = std::env::var_os("CARGO_FEATURE_LLM_LLAMA_CPP").is_some();
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").ok();

    if whisper_enabled && llama_enabled && target_os.as_deref() == Some("linux") {
        println!("cargo:rustc-link-arg-bins=-Wl,--allow-multiple-definition");
        println!("cargo:rustc-link-lib=gomp");
    }
}
