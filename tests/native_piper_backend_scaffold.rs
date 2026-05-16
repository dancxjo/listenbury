#![cfg(feature = "tts-piper-native")]

use std::{env, fs, path::PathBuf};

use listenbury::mouth::piper_native::{NativePiperBackend, PiperVoiceConfig};

#[test]
#[ignore = "set LISTENBURY_TEST_PIPER_MODEL and LISTENBURY_TEST_PIPER_CONFIG to a local Piper model/config, and ensure an ONNX Runtime shared library is available"]
fn loads_real_local_piper_model_when_configured() {
    let model_path = PathBuf::from(
        env::var("LISTENBURY_TEST_PIPER_MODEL").expect("LISTENBURY_TEST_PIPER_MODEL"),
    );
    let config_path = PathBuf::from(
        env::var("LISTENBURY_TEST_PIPER_CONFIG").expect("LISTENBURY_TEST_PIPER_CONFIG"),
    );

    let config = PiperVoiceConfig::from_json_str(
        &fs::read_to_string(&config_path).expect("read Piper config JSON"),
    )
    .expect("parse Piper config JSON");

    let backend = NativePiperBackend::load(&model_path, config).expect("load local Piper model");
    let contract = backend
        .validate_model_contract()
        .expect("validate ONNX model contract");

    assert!(!contract.input_names.is_empty(), "expected model inputs");
    assert!(!contract.output_names.is_empty(), "expected model outputs");
}
