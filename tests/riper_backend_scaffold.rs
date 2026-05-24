#![cfg(feature = "piper-compat")]

use std::{env, fs, path::PathBuf};

use listenbury::mouth::riper::{PiperIdSequence, PiperVoiceConfig, RiperBackend};

#[test]
#[ignore = "set LISTENBURY_TEST_PIPER_MODEL, LISTENBURY_TEST_PIPER_CONFIG, and LISTENBURY_TEST_PIPER_IDS for a local Piper model/config; ensure ONNX Runtime shared library is available"]
fn synthesizes_real_local_piper_model_ids_when_configured() {
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

    let ids = PiperIdSequence {
        ids: env::var("LISTENBURY_TEST_PIPER_IDS")
            .expect("LISTENBURY_TEST_PIPER_IDS")
            .split(',')
            .map(|value| value.trim().parse::<i64>().expect("parse Piper ID"))
            .collect(),
    };

    let mut backend = RiperBackend::load(&model_path, config).expect("load local Piper model");
    let contract = backend
        .validate_model_contract()
        .expect("validate ONNX model contract");
    let pcm = backend
        .synthesize_ids(&ids)
        .expect("synthesize from explicit Piper IDs");

    assert!(!contract.input_names.is_empty(), "expected model inputs");
    assert!(!contract.output_names.is_empty(), "expected model outputs");
    assert_eq!(pcm.sample_rate_hz, backend.config().sample_rate_hz);
    assert!(
        !pcm.samples.is_empty(),
        "expected non-empty waveform samples"
    );
}
