
use super::*;
use listenbury::time::ExactTimestamp;

#[test]
fn say_args_treats_single_word_as_text() {
    let args = SayArgs::from_command(SayCommand {
        piper: false,
        riper: false,
        piper_bin: None,
        piper_voice: None,
        output_wav: None,
        dump_pipeline: false,
        dump_phonemes: false,
        dump_phone_plan: false,
        dump_piper_tensors: false,
        klatt: false,
        hifigan: false,
        speecht5: false,
        hifigan_model: None,
        skip_gan: false,
        rp: false,
        diphone: false,
        mbrola_voice: None,
        words: vec!["hello".to_string()],
    })
    .expect("single word should be text");

    assert!(args.piper_bin.is_none());
    assert!(args.piper_voice.is_none());
    assert_eq!(args.text, "hello");
}

#[test]
fn say_args_accepts_dump_pipeline_flag() {
    let args = SayArgs::from_command(SayCommand {
        piper: false,
        riper: false,
        piper_bin: None,
        piper_voice: None,
        output_wav: None,
        dump_pipeline: true,
        dump_phonemes: false,
        dump_phone_plan: false,
        dump_piper_tensors: false,
        klatt: false,
        hifigan: false,
        speecht5: false,
        hifigan_model: None,
        skip_gan: false,
        rp: false,
        diphone: false,
        mbrola_voice: None,
        words: vec!["hello".to_string()],
    })
    .expect("--dump-pipeline should parse without changing the default route");

    assert!(args.dump_pipeline);
    let dump = format_say_pipeline(&args);
    assert!(dump.contains("speech pipeline: piper-compat"));
    assert!(dump.contains("-> acoustic generator: piper-compatible ONNX/Riper"));
    assert!(dump.contains("-> vocoder: piper-compatible internal"));
}

#[test]
fn say_args_accepts_dump_phonemes_flag() {
    let args = SayArgs::from_command(SayCommand {
        piper: false,
        riper: false,
        piper_bin: None,
        piper_voice: None,
        output_wav: None,
        dump_pipeline: false,
        dump_phonemes: true,
        dump_phone_plan: false,
        dump_piper_tensors: false,
        klatt: false,
        hifigan: false,
        speecht5: false,
        hifigan_model: None,
        skip_gan: false,
        rp: false,
        diphone: false,
        mbrola_voice: None,
        words: vec!["hello".to_string()],
    })
    .expect("--dump-phonemes should parse without changing the default route");

    assert!(args.dump_phonemes);
    let dump = format_say_phonemes(&args).expect("phoneme dump should format");
    assert!(dump.contains("phoneme dump: piper-compat"));
    assert!(dump.contains("riper phonemes:"));
    assert!(dump.contains("acoustic phones:"));
    assert!(dump.contains("word phones:"));
}

#[test]
fn say_args_accepts_dump_phone_plan_flag() {
    let args = SayArgs::from_command(SayCommand {
        piper: false,
        riper: true,
        piper_bin: None,
        piper_voice: None,
        output_wav: None,
        dump_pipeline: false,
        dump_phonemes: false,
        dump_phone_plan: true,
        dump_piper_tensors: false,
        klatt: false,
        hifigan: false,
        speecht5: false,
        hifigan_model: None,
        skip_gan: false,
        rp: false,
        diphone: false,
        mbrola_voice: None,
        words: vec!["Don't be jealous.".to_string()],
    })
    .expect("--dump-phone-plan should parse without synthesis setup");

    assert!(args.dump_phone_plan);
    let plan = PhonePlan::from_text_with_riper_g2p(&args.text).expect("plan should build");
    assert_eq!(plan.words[0].phones, ["d", "ow", "n", "t"]);
}

#[test]
fn say_args_accepts_trailing_dump_phonemes_flag() {
    let args = SayArgs::from_command(SayCommand {
        piper: false,
        riper: false,
        piper_bin: None,
        piper_voice: None,
        output_wav: None,
        dump_pipeline: false,
        dump_phonemes: false,
        dump_phone_plan: false,
        dump_piper_tensors: false,
        klatt: false,
        hifigan: true,
        speecht5: false,
        hifigan_model: None,
        skip_gan: false,
        rp: false,
        diphone: false,
        mbrola_voice: None,
        words: vec!["hello".to_string(), "--dump-phonemes".to_string()],
    })
    .expect("trailing phoneme dump flag should be accepted");

    assert!(args.dump_phonemes);
    assert_eq!(args.text, "hello");
    let dump = format_say_phonemes(&args).expect("phoneme dump should format");
    assert!(dump.contains("phoneme dump: speecht5-hifigan"));
    assert!(dump.contains("SpeechT5"));
}

#[test]
fn say_args_accepts_trailing_trace_speech_pipeline_flag() {
    let args = SayArgs::from_command(SayCommand {
        piper: false,
        riper: false,
        piper_bin: None,
        piper_voice: None,
        output_wav: None,
        dump_pipeline: false,
        dump_phonemes: false,
        dump_phone_plan: false,
        dump_piper_tensors: false,
        klatt: true,
        hifigan: false,
        speecht5: false,
        hifigan_model: None,
        skip_gan: false,
        rp: false,
        diphone: false,
        mbrola_voice: None,
        words: vec!["hello".to_string(), "--trace-speech-pipeline".to_string()],
    })
    .expect("trailing pipeline trace flag should be accepted");

    assert!(args.dump_pipeline);
    assert_eq!(args.text, "hello");
    let dump = format_say_pipeline(&args);
    assert!(dump.contains("speech pipeline: klatt"));
    assert!(dump.contains("-> acoustic generator: klatt"));
    assert!(dump.contains("-> mel/features: disabled"));
    assert!(dump.contains("-> vocoder: disabled"));
}

#[test]
fn say_args_accepts_legacy_piper_bin_position() {
    let args = SayArgs::from_command(SayCommand {
        piper: true,
        riper: false,
        piper_bin: None,
        piper_voice: None,
        output_wav: None,
        dump_pipeline: false,
        dump_phonemes: false,
        dump_phone_plan: false,
        dump_piper_tensors: false,
        klatt: false,
        hifigan: false,
        speecht5: false,
        hifigan_model: None,
        skip_gan: false,
        rp: false,
        diphone: false,
        mbrola_voice: None,
        words: vec![
            "/snap/bin/piper-tts.piper-cli".to_string(),
            "hello".to_string(),
        ],
    })
    .expect("legacy Piper executable should be accepted when --piper is selected");

    assert_eq!(
        args.piper_bin,
        Some(PathBuf::from("/snap/bin/piper-tts.piper-cli"))
    );
    assert!(args.piper_voice.is_none());
    assert_eq!(args.text, "hello");
}

#[test]
fn say_args_accepts_legacy_voice_position() {
    let args = SayArgs::from_command(SayCommand {
        piper: true,
        riper: false,
        piper_bin: None,
        piper_voice: None,
        output_wav: None,
        dump_pipeline: false,
        dump_phonemes: false,
        dump_phone_plan: false,
        dump_piper_tensors: false,
        klatt: false,
        hifigan: false,
        speecht5: false,
        hifigan_model: None,
        skip_gan: false,
        rp: false,
        diphone: false,
        mbrola_voice: None,
        words: vec![
            "/snap/bin/piper-tts.piper-cli".to_string(),
            "voice.onnx".to_string(),
            "hello".to_string(),
        ],
    })
    .expect("legacy Piper executable and voice should be accepted");

    assert_eq!(
        args.piper_bin,
        Some(PathBuf::from("/snap/bin/piper-tts.piper-cli"))
    );
    assert_eq!(args.piper_voice, Some(PathBuf::from("voice.onnx")));
    assert_eq!(args.text, "hello");
}

#[test]
fn say_args_accepts_trailing_riper_flag() {
    let args = SayArgs::from_command(SayCommand {
        piper: false,
        riper: false,
        piper_bin: None,
        piper_voice: None,
        output_wav: None,
        dump_pipeline: false,
        dump_phonemes: false,
        dump_phone_plan: false,
        dump_piper_tensors: false,
        klatt: false,
        hifigan: false,
        speecht5: false,
        hifigan_model: None,
        skip_gan: false,
        rp: false,
        diphone: false,
        mbrola_voice: None,
        words: vec![
            "hello".to_string(),
            "there".to_string(),
            "--riper".to_string(),
        ],
    })
    .expect("--riper should be accepted as an explicit default route");
    assert_eq!(args.text, "hello there");
    assert!(!args.piper);
}

#[test]
fn say_args_accepts_trailing_klatt_flag() {
    let error = SayArgs::from_command(SayCommand {
        piper: false,
        riper: false,
        piper_bin: None,
        piper_voice: None,
        output_wav: None,
        dump_pipeline: false,
        dump_phonemes: false,
        dump_phone_plan: false,
        dump_piper_tensors: false,
        klatt: false,
        hifigan: false,
        speecht5: false,
        hifigan_model: None,
        skip_gan: false,
        rp: false,
        diphone: false,
        mbrola_voice: None,
        words: vec!["hello".to_string(), "my".to_string(), "--klatt".to_string()],
    })
    .expect("Klatt is a default Riper-path backend");
    assert!(error.klatt);
    assert_eq!(error.text, "hello my");
}

#[test]
fn say_args_accepts_klatt() {
    let args = SayArgs::from_command(SayCommand {
        piper: false,
        riper: false,
        piper_bin: None,
        piper_voice: None,
        output_wav: None,
        dump_pipeline: false,
        dump_phonemes: false,
        dump_phone_plan: false,
        dump_piper_tensors: false,
        klatt: true,
        hifigan: false,
        speecht5: false,
        hifigan_model: None,
        skip_gan: false,
        rp: false,
        diphone: false,
        mbrola_voice: None,
        words: vec!["hello".to_string()],
    })
    .expect("klatt should parse as a Riper backend alternative");
    assert!(args.klatt);
    assert!(should_use_klatt_backend(&args));
    assert_eq!(say_backend_graph(&args).id, "klatt");
    assert_eq!(say_speech_loom(&args).projection, "current-backend/klatt");
}

#[test]
fn say_args_accepts_hifigan() {
    let args = SayArgs::from_command(SayCommand {
        piper: false,
        riper: false,
        piper_bin: None,
        piper_voice: None,
        output_wav: None,
        dump_pipeline: false,
        dump_phonemes: false,
        dump_phone_plan: false,
        dump_piper_tensors: false,
        klatt: false,
        hifigan: true,
        speecht5: false,
        hifigan_model: None,
        skip_gan: false,
        rp: false,
        diphone: false,
        mbrola_voice: None,
        words: vec!["hello".to_string()],
    })
    .expect("hifigan should parse, selecting the SpeechT5 acoustic route");
    assert!(!args.klatt);
    assert!(args.hifigan);
    assert!(should_use_speecht5_backend(&args));
    assert!(!should_use_source_filter_hifigan_backend(&args));
    assert_eq!(say_backend_graph(&args).id, "speecht5-hifigan");
    assert_eq!(
        say_speech_loom(&args).projection,
        "current-backend/speecht5-hifigan"
    );
    let dump = format_say_pipeline(&args);
    assert!(dump.contains("speech pipeline: speecht5-hifigan"));
    assert!(dump.contains("-> tokenizer: SpeechT5 tokenizer"));
    assert!(dump.contains("-> acoustic generator: SpeechT5 encoder/decoder ONNX"));
    assert!(dump.contains("-> mel/features: SpeechT5 mel spectrogram"));
    assert!(dump.contains("-> vocoder: SpeechT5 HiFi-GAN ONNX"));
    assert!(!dump.contains("source-filter spectral proxy"));
}

#[test]
fn say_args_accepts_speecht5_as_native_acoustic_route() {
    let args = SayArgs::from_command(SayCommand {
        piper: false,
        riper: false,
        piper_bin: None,
        piper_voice: None,
        output_wav: None,
        dump_pipeline: false,
        dump_phonemes: false,
        dump_phone_plan: false,
        dump_piper_tensors: false,
        klatt: false,
        hifigan: false,
        speecht5: true,
        hifigan_model: None,
        skip_gan: false,
        rp: false,
        diphone: false,
        mbrola_voice: None,
        words: vec!["hello".to_string()],
    })
    .expect("speecht5 should parse as a native acoustic route");
    assert!(!args.hifigan);
    assert!(args.speecht5);
    assert!(should_use_speecht5_backend(&args));
    assert_eq!(say_backend_graph(&args).id, "speecht5-hifigan");
    assert_eq!(
        say_speech_loom(&args).projection,
        "current-backend/speecht5-hifigan"
    );
    let dump = format_say_pipeline(&args);
    assert!(dump.contains("speech pipeline: speecht5-hifigan"));
    assert!(dump.contains("-> tokenizer: SpeechT5 tokenizer"));
    assert!(dump.contains("-> acoustic generator: SpeechT5 encoder/decoder ONNX"));
    assert!(dump.contains("-> mel/features: SpeechT5 mel spectrogram"));
    assert!(dump.contains("-> vocoder: SpeechT5 HiFi-GAN ONNX"));
    assert!(!dump.contains("source-filter spectral proxy"));
}

#[test]
fn say_args_accepts_trailing_speecht5_flag() {
    let args = SayArgs::from_command(SayCommand {
        piper: false,
        riper: false,
        piper_bin: None,
        piper_voice: None,
        output_wav: None,
        dump_pipeline: false,
        dump_phonemes: false,
        dump_phone_plan: false,
        dump_piper_tensors: false,
        klatt: false,
        hifigan: false,
        speecht5: false,
        hifigan_model: None,
        skip_gan: false,
        rp: false,
        diphone: false,
        mbrola_voice: None,
        words: vec!["hello".to_string(), "--speecht5".to_string()],
    })
    .expect("trailing SpeechT5 flag should be accepted");
    assert!(args.speecht5);
    assert_eq!(args.text, "hello");
}

#[test]
fn say_args_rejects_hifigan_model_without_hifigan_or_speecht5() {
    let error = SayArgs::from_command(SayCommand {
        piper: false,
        riper: false,
        piper_bin: None,
        piper_voice: None,
        output_wav: None,
        dump_pipeline: false,
        dump_phonemes: false,
        dump_phone_plan: false,
        dump_piper_tensors: false,
        klatt: false,
        hifigan: false,
        speecht5: false,
        hifigan_model: Some(PathBuf::from("speecht5_hifigan.onnx")),
        skip_gan: false,
        rp: false,
        diphone: false,
        mbrola_voice: None,
        words: vec!["hello".to_string()],
    })
    .expect_err("--hifigan-model should require a HiFi-GAN route");
    assert!(
        error
            .to_string()
            .contains("--hifigan-model only applies when --hifigan or --speecht5"),
        "unexpected error: {error}"
    );
}

#[test]
fn say_args_accepts_trailing_hifigan_flag() {
    let args = SayArgs::from_command(SayCommand {
        piper: false,
        riper: false,
        piper_bin: None,
        piper_voice: None,
        output_wav: None,
        dump_pipeline: false,
        dump_phonemes: false,
        dump_phone_plan: false,
        dump_piper_tensors: false,
        klatt: false,
        hifigan: false,
        speecht5: false,
        hifigan_model: None,
        skip_gan: false,
        rp: false,
        diphone: false,
        mbrola_voice: None,
        words: vec!["hello".to_string(), "--hifigan".to_string()],
    })
    .expect("trailing HiFi-GAN flag should be accepted");
    assert!(args.hifigan);
    assert_eq!(args.text, "hello");
}

#[test]
fn say_args_accepts_skip_gan_as_hifigan_modifier() {
    let args = SayArgs::from_command(SayCommand {
        piper: false,
        riper: false,
        piper_bin: None,
        piper_voice: None,
        output_wav: None,
        dump_pipeline: false,
        dump_phonemes: false,
        dump_phone_plan: false,
        dump_piper_tensors: false,
        klatt: false,
        hifigan: true,
        speecht5: false,
        hifigan_model: None,
        skip_gan: true,
        rp: false,
        diphone: false,
        mbrola_voice: None,
        words: vec!["hello".to_string()],
    })
    .expect("--hifigan --skip-gan should select the source-filter mel debug route");
    assert!(args.hifigan);
    assert!(args.skip_gan);
    assert!(should_use_source_filter_hifigan_backend(&args));
    assert!(!should_use_speecht5_backend(&args));
    assert_eq!(say_backend_graph(&args).id, "source-filter-hifigan");
}

#[test]
fn say_args_accepts_trailing_hifigan_fallback_alias() {
    let args = SayArgs::from_command(SayCommand {
        piper: false,
        riper: false,
        piper_bin: None,
        piper_voice: None,
        output_wav: None,
        dump_pipeline: false,
        dump_phonemes: false,
        dump_phone_plan: false,
        dump_piper_tensors: false,
        klatt: false,
        hifigan: false,
        speecht5: false,
        hifigan_model: None,
        skip_gan: false,
        rp: false,
        diphone: false,
        mbrola_voice: None,
        words: vec![
            "hello".to_string(),
            "--hifigan".to_string(),
            "--hifigan-fallback".to_string(),
        ],
    })
    .expect("trailing --hifigan-fallback should select the mel debug route");
    assert!(args.hifigan);
    assert!(args.skip_gan);
    assert_eq!(args.text, "hello");
}

#[test]
fn say_args_rejects_skip_gan_without_hifigan() {
    let error = SayArgs::from_command(SayCommand {
        piper: false,
        riper: false,
        piper_bin: None,
        piper_voice: None,
        output_wav: None,
        dump_pipeline: false,
        dump_phonemes: false,
        dump_phone_plan: false,
        dump_piper_tensors: false,
        klatt: false,
        hifigan: false,
        speecht5: false,
        hifigan_model: None,
        skip_gan: false,
        rp: false,
        diphone: true,
        mbrola_voice: None,
        words: vec!["And sudd....".to_string(), "--skip-gan".to_string()],
    })
    .expect_err("--skip-gan should not select the mel debug route by itself");
    assert!(error.to_string().contains("--skip-gan only applies"));
}

#[test]
fn say_args_accepts_diphone_voice() {
    let args = SayArgs::from_command(SayCommand {
        piper: false,
        riper: false,
        piper_bin: None,
        piper_voice: None,
        output_wav: None,
        dump_pipeline: false,
        dump_phonemes: false,
        dump_phone_plan: false,
        dump_piper_tensors: false,
        klatt: false,
        hifigan: false,
        speecht5: false,
        hifigan_model: None,
        skip_gan: false,
        rp: false,
        diphone: true,
        mbrola_voice: Some(PathBuf::from("voices/us1")),
        words: vec!["hello".to_string()],
    })
    .expect("diphone should select the diphone voice backend");
    assert!(!args.klatt);
    assert!(should_use_mbrola_backend(&args));
    assert_eq!(args.mbrola_voice, Some(PathBuf::from("voices/us1")));
    assert_eq!(say_backend_graph(&args).id, "mbrola-diphone");
}

#[test]
fn say_args_accepts_diphone() {
    let args = SayArgs::from_command(SayCommand {
        piper: false,
        riper: false,
        piper_bin: None,
        piper_voice: None,
        output_wav: None,
        dump_pipeline: false,
        dump_phonemes: false,
        dump_phone_plan: false,
        dump_piper_tensors: false,
        klatt: false,
        hifigan: false,
        speecht5: false,
        hifigan_model: None,
        skip_gan: false,
        rp: false,
        diphone: true,
        mbrola_voice: None,
        words: vec!["hello".to_string()],
    })
    .expect("diphone should select the diphone voice backend");
    assert!(args.mbrola);
    assert_eq!(say_backend_graph(&args).id, "mbrola-diphone");
}

#[test]
fn say_backend_graph_defaults_to_piper_compat() {
    let args = SayArgs::from_command(SayCommand {
        piper: false,
        riper: false,
        piper_bin: None,
        piper_voice: None,
        output_wav: None,
        dump_pipeline: false,
        dump_phonemes: false,
        dump_phone_plan: false,
        dump_piper_tensors: false,
        klatt: false,
        hifigan: false,
        speecht5: false,
        hifigan_model: None,
        skip_gan: false,
        rp: false,
        diphone: false,
        mbrola_voice: None,
        words: vec!["hello".to_string()],
    })
    .expect("default say route should parse");
    let backend_graph = say_backend_graph(&args);
    let loom = say_speech_loom(&args);
    assert_eq!(backend_graph.id, "piper-compat");
    assert!(backend_graph.fused);
    assert_eq!(backend_graph.workers.len(), 1);
    assert_eq!(backend_graph.workers[0].id, "piper-compatible-onnx");
    assert_eq!(loom.projection, "current-backend/piper-compat");
}

#[test]
fn say_backend_graph_reports_external_piper_process() {
    let args = SayArgs::from_command(SayCommand {
        piper: true,
        riper: false,
        piper_bin: None,
        piper_voice: None,
        output_wav: None,
        dump_pipeline: false,
        dump_phonemes: false,
        dump_phone_plan: false,
        dump_piper_tensors: false,
        klatt: false,
        hifigan: false,
        speecht5: false,
        hifigan_model: None,
        skip_gan: false,
        rp: false,
        diphone: false,
        mbrola_voice: None,
        words: vec!["hello".to_string()],
    })
    .expect("external piper route should parse");
    let backend_graph = say_backend_graph(&args);
    assert_eq!(backend_graph.id, "piper-process");
    assert!(backend_graph.fused);
    assert_eq!(backend_graph.workers.len(), 1);
    assert_eq!(backend_graph.workers[0].id, "piper-process-backend");
}

#[test]
fn say_backend_graph_reports_klatt_worker_contract() {
    let args = SayArgs::from_command(SayCommand {
        piper: false,
        riper: false,
        piper_bin: None,
        piper_voice: None,
        output_wav: None,
        dump_pipeline: false,
        dump_phonemes: false,
        dump_phone_plan: false,
        dump_piper_tensors: false,
        klatt: true,
        hifigan: false,
        speecht5: false,
        hifigan_model: None,
        skip_gan: false,
        rp: false,
        diphone: false,
        mbrola_voice: None,
        words: vec!["hello".to_string()],
    })
    .expect("klatt route should parse");
    let backend_graph = say_backend_graph(&args);
    assert_eq!(backend_graph.id, "klatt");
    assert!(!backend_graph.fused);
    assert_eq!(backend_graph.workers.len(), 1);
    assert_eq!(backend_graph.workers[0].id, "klatt-formant-renderer");
}

#[test]
fn say_backend_graph_reports_mbrola_internal_workers() {
    let args = SayArgs::from_command(SayCommand {
        piper: false,
        riper: false,
        piper_bin: None,
        piper_voice: None,
        output_wav: None,
        dump_pipeline: false,
        dump_phonemes: false,
        dump_phone_plan: false,
        dump_piper_tensors: false,
        klatt: false,
        hifigan: false,
        speecht5: false,
        hifigan_model: None,
        skip_gan: false,
        rp: false,
        diphone: true,
        mbrola_voice: None,
        words: vec!["hello".to_string()],
    })
    .expect("diphone route should parse");
    let backend_graph = say_backend_graph(&args);
    assert_eq!(backend_graph.id, "mbrola-diphone");
    assert!(!backend_graph.fused);
    assert_eq!(backend_graph.workers.len(), 2);
    assert_eq!(backend_graph.workers[0].id, "mbrola-diphone-selection");
    assert_eq!(backend_graph.workers[1].id, "mbrola-diphone-renderer");
    let dump = format_say_pipeline(&args);
    assert!(dump.contains("-> acoustic generator: MBROLA-compatible diphone renderer"));
    assert!(dump.contains("-> diphone voice: default MBROLA-compatible voice"));
}

#[test]
fn say_backend_graph_reports_hifigan_speecht5_workers() {
    let args = SayArgs::from_command(SayCommand {
        piper: false,
        riper: false,
        piper_bin: None,
        piper_voice: None,
        output_wav: None,
        dump_pipeline: false,
        dump_phonemes: false,
        dump_phone_plan: false,
        dump_piper_tensors: false,
        klatt: false,
        hifigan: true,
        speecht5: false,
        hifigan_model: None,
        skip_gan: false,
        rp: false,
        diphone: false,
        mbrola_voice: None,
        words: vec!["hello".to_string()],
    })
    .expect("hifigan route should parse");
    let backend_graph = say_backend_graph(&args);
    let loom = say_speech_loom(&args);
    assert_eq!(backend_graph.id, "speecht5-hifigan");
    assert!(!backend_graph.fused);
    assert_eq!(backend_graph.workers.len(), 3);
    assert_eq!(backend_graph.workers[0].id, "speecht5-tokenizer");
    assert_eq!(
        backend_graph.workers[1].id,
        "speecht5-encoder-decoder-acoustic-generator"
    );
    assert_eq!(backend_graph.workers[2].id, "speecht5-hifigan-vocoder");
    assert_eq!(loom.projection, "current-backend/speecht5-hifigan");
}

#[test]
fn say_backend_graph_reports_hifigan_fallback_feature_bridge_workers() {
    let args = SayArgs::from_command(SayCommand {
        piper: false,
        riper: false,
        piper_bin: None,
        piper_voice: None,
        output_wav: None,
        dump_pipeline: false,
        dump_phonemes: false,
        dump_phone_plan: false,
        dump_piper_tensors: false,
        klatt: false,
        hifigan: true,
        speecht5: false,
        hifigan_model: None,
        skip_gan: true,
        rp: false,
        diphone: false,
        mbrola_voice: None,
        words: vec!["hello".to_string()],
    })
    .expect("hifigan fallback route should parse");
    let backend_graph = say_backend_graph(&args);
    let loom = say_speech_loom(&args);
    assert_eq!(backend_graph.id, "source-filter-hifigan");
    assert!(!backend_graph.fused);
    assert_eq!(backend_graph.workers.len(), 4);
    assert_eq!(
        backend_graph.workers[0].id,
        "source-filter-acoustic-generator"
    );
    assert_eq!(
        backend_graph.workers[1].id,
        "source-filter-temporal-smoother"
    );
    assert_eq!(
        backend_graph.workers[2].id,
        "source-filter-mel-compat-bridge"
    );
    assert_eq!(backend_graph.workers[3].id, "hifigan-vocoder");
    assert_eq!(loom.projection, "current-backend/source-filter-hifigan");
}

#[test]
fn say_args_rp_selects_en1_mbrola_voice() {
    let args = SayArgs::from_command(SayCommand {
        piper: false,
        riper: false,
        piper_bin: None,
        piper_voice: None,
        output_wav: None,
        dump_pipeline: false,
        dump_phonemes: false,
        dump_phone_plan: false,
        dump_piper_tensors: false,
        klatt: false,
        hifigan: false,
        speecht5: false,
        hifigan_model: None,
        skip_gan: false,
        rp: true,
        diphone: false,
        mbrola_voice: None,
        words: vec!["hello".to_string()],
    })
    .expect("RP shorthand should select the en1 MBROLA voice");
    assert!(args.mbrola);
    assert_eq!(
        args.mbrola_voice,
        Some(received_pronunciation_mbrola_voice())
    );
}

#[test]
fn say_args_accepts_trailing_rp_flag() {
    let args = SayArgs::from_command(SayCommand {
        piper: false,
        riper: false,
        piper_bin: None,
        piper_voice: None,
        output_wav: None,
        dump_pipeline: false,
        dump_phonemes: false,
        dump_phone_plan: false,
        dump_piper_tensors: false,
        klatt: false,
        hifigan: false,
        speecht5: false,
        hifigan_model: None,
        skip_gan: false,
        rp: false,
        diphone: false,
        mbrola_voice: None,
        words: vec!["hello".to_string(), "--rp".to_string()],
    })
    .expect("trailing RP shorthand should be accepted");
    assert!(args.mbrola);
    assert_eq!(args.text, "hello");
    assert_eq!(
        args.mbrola_voice,
        Some(received_pronunciation_mbrola_voice())
    );
}

#[test]
fn say_args_rejects_rp_with_klatt() {
    let error = SayArgs::from_command(SayCommand {
        piper: false,
        riper: false,
        piper_bin: None,
        piper_voice: None,
        output_wav: None,
        dump_pipeline: false,
        dump_phonemes: false,
        dump_phone_plan: false,
        dump_piper_tensors: false,
        klatt: false,
        hifigan: false,
        speecht5: false,
        hifigan_model: None,
        skip_gan: false,
        rp: true,
        diphone: false,
        mbrola_voice: None,
        words: vec!["hello".to_string(), "--klatt".to_string()],
    })
    .expect_err("RP shorthand should conflict with Klatt");
    assert!(
        error.to_string().contains("MBROLA/RP voice path"),
        "unexpected error: {error}"
    );
}

#[test]
fn say_args_uses_default_diphone_demo_text() {
    let args = SayArgs::from_command(SayCommand {
        piper: false,
        riper: false,
        piper_bin: None,
        piper_voice: None,
        output_wav: None,
        dump_pipeline: false,
        dump_phonemes: false,
        dump_phone_plan: false,
        dump_piper_tensors: false,
        klatt: false,
        hifigan: false,
        speecht5: false,
        hifigan_model: None,
        skip_gan: false,
        rp: false,
        diphone: true,
        mbrola_voice: None,
        words: Vec::new(),
    })
    .expect("diphone should have a default smoke utterance");
    assert_eq!(args.text, "Hello, my baby.");
}

#[test]
fn say_args_treats_dash_as_stdin_stream() {
    let args = SayArgs::from_command(SayCommand {
        piper: false,
        riper: false,
        piper_bin: None,
        piper_voice: None,
        output_wav: None,
        dump_pipeline: false,
        dump_phonemes: false,
        dump_phone_plan: false,
        dump_piper_tensors: false,
        klatt: true,
        hifigan: false,
        speecht5: false,
        hifigan_model: None,
        skip_gan: false,
        rp: false,
        diphone: false,
        mbrola_voice: None,
        words: vec!["-".to_string()],
    })
    .expect("dash should select stdin streaming");

    assert!(args.stdin_stream);
    assert!(args.klatt);
    assert!(args.text.is_empty());
}

#[test]
fn klatt_phrase_renders_non_empty_audio_and_wav_bytes() {
    let frames =
        synthesize_klatt_for_say("Hello, my baby. Hello, my darling. Hello, my ragtime gal.")
            .expect("klatt phrase should synthesize");
    assert_eq!(frames.len(), 1);
    assert!(!frames[0].samples.is_empty());
    let wav = write_wav_bytes(&frames).expect("frames should serialize as WAV");
    assert!(wav.len() > 44, "WAV payload should include audio data");
}

#[test]
fn klatt_phrase_unknown_word_reports_clear_error() {
    let error = synthesize_klatt_for_say("Hello 💥")
        .expect_err("unsupported text should produce a clear error");
    assert!(error.to_string().contains("could not phonemize"));
}

#[test]
#[cfg(feature = "piper-compat")]
fn klatt_uses_riper_pronunciation_for_mixed_prose() {
    let frames = synthesize_klatt_for_say(
            "MBROLA was created by Thierry Dutoit. It's a speech synthesizer based on the concatenation of diphones.",
        )
        .expect("Klatt should synthesize prose via Riper pronunciation machinery");
    assert_eq!(frames.len(), 1);
    assert!(!frames[0].samples.is_empty());
}

#[test]
#[cfg(feature = "piper-compat")]
fn klatt_riper_phone_bridge_splits_diphthongs_and_affricates() {
    let phone_string = klatt_phone_string_for_text("Okay, Charlie.")
        .expect("Riper phones should convert to Klatt render phones");
    let ipas = phone_string.ipa_segments();
    assert!(ipas.windows(2).any(|phones| phones == ["o", "ʊ"]));
    assert!(ipas.windows(2).any(|phones| phones == ["t", "ʃ"]));
}

#[test]
#[cfg(feature = "piper-compat")]
fn diphone_plan_uses_planned_durations_pitches_and_pause() {
    let plan = planned_phone_timed_plan_for_text(
        SimpleEnglishG2p::default(),
        "The red machine.",
        |symbol| Ok(symbol.to_string()),
        "test",
    )
    .expect("planned diphone plan");

    assert!(
        plan.phones.iter().any(|phone| phone.symbol != "_"
            && phone.duration_ms != 75
            && phone.duration_ms != 145),
        "planned durations should replace the old canned consonant/vowel defaults: {:?}",
        plan.phones
    );
    assert!(
        plan.phones.iter().any(|phone| phone
            .pitch_targets
            .iter()
            .any(|target| (target.hz - 135.0).abs() > 0.01)),
        "planned pitch shapes should replace the old canned vowel pitch triplet: {:?}",
        plan.phones
    );
    assert_eq!(
        plan.phones.last(),
        Some(&MbrolaPhone::new("_", 260)),
        "committed full-turn say should use the planner final pause"
    );
}

#[test]
#[cfg(feature = "piper-compat")]
fn frame_duration_ms_handles_zero_values() {
    let frame = AudioFrame {
        captured_at: ExactTimestamp::now(),
        sample_rate_hz: 0,
        channels: 1,
        samples: vec![0.0; 1600],
        voice_signatures: Vec::new(),
    };
    assert_eq!(frame_duration_ms(&frame), 0);

    let frame = AudioFrame {
        captured_at: ExactTimestamp::now(),
        sample_rate_hz: 16_000,
        channels: 0,
        samples: vec![0.0; 1600],
        voice_signatures: Vec::new(),
    };
    assert_eq!(frame_duration_ms(&frame), 0);
}

#[test]
#[cfg(feature = "piper-compat")]
fn frame_duration_ms_preserves_fractional_millisecond_precision() {
    let frame = AudioFrame {
        captured_at: ExactTimestamp::now(),
        sample_rate_hz: 16_000,
        channels: 2,
        samples: vec![0.0; 3_200],
        voice_signatures: Vec::new(),
    };

    assert_eq!(frame_duration_ms(&frame), 100);
}

#[test]
#[cfg(feature = "piper-compat")]
fn riper_compare_args_joins_words_into_text() {
    let args = RiperCompareArgs::from_command(RiperCompareCommand {
        piper_bin: None,
        piper_voice: None,
        riper_voice: None,
        riper_config: None,
        process_output_wav: None,
        riper_output_wav: None,
        phonemes: None,
        words: vec!["Okay.".to_string(), "Again.".to_string()],
    })
    .expect("words should parse");

    assert_eq!(args.text, "Okay. Again.");
}

#[test]
#[cfg(unix)]
fn snap_piper_copy_check_follows_symlink_to_hidden_directory() {
    let root = std::env::temp_dir().join(format!(
        "listenbury-piper-symlink-test-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    let hidden_dir = root.join(".models");
    let visible_dir = root.join("voices");
    std::fs::create_dir_all(&hidden_dir).expect("create hidden model directory");
    std::fs::create_dir_all(&visible_dir).expect("create visible voice directory");

    let hidden_model = hidden_dir.join("ryan.onnx");
    std::fs::write(&hidden_model, b"model").expect("write hidden model");
    let visible_model = visible_dir.join("ryan.onnx");
    std::os::unix::fs::symlink(&hidden_model, &visible_model).expect("create model symlink");

    assert!(piper_model_needs_snap_copy(
        Path::new("/snap/bin/piper-tts.piper-cli"),
        &visible_model,
    ));
    assert!(!piper_model_needs_snap_copy(
        Path::new("/usr/bin/piper"),
        &visible_model,
    ));

    std::fs::remove_dir_all(root).expect("remove test directory");
}

#[test]
#[cfg(feature = "piper-compat")]
fn espeak_compatible_ids_match_piper_debug_shape_for_okay() {
    let config = PiperVoiceConfig::from_json_str(
        r#"
            {
              "audio": { "sample_rate": 22050 },
              "phoneme_id_map": {
                "_": [0],
                "^": [1],
                "$": [2],
                ".": [10],
                "e": [18],
                "k": [23],
                "o": [27],
                "ɪ": [74],
                "ʊ": [100],
                "ˈ": [120]
              }
            }
            "#,
    )
    .expect("voice config should parse");
    let sequence = PiperPhonemeSequence {
        phonemes: ["OW", "K", "EY", "|"]
            .into_iter()
            .map(|symbol| PiperPhoneme(symbol.to_string()))
            .collect(),
    };

    let ids = sequence
        .to_piper_ids_compatible(&config)
        .expect("ARPAbet symbols should map to eSpeak Piper IDs");

    assert_eq!(
        ids,
        PiperIdSequence {
            ids: vec![1, 0, 27, 0, 100, 0, 23, 0, 18, 0, 74, 0, 10, 0, 2]
        }
    );
}

#[test]
#[cfg(feature = "piper-compat")]
fn espeak_compatible_ids_support_lollipop_guild_sentence_symbols() {
    let config = PiperVoiceConfig::from_json_str(
        r#"
            {
              "audio": { "sample_rate": 22050 },
              "phoneme_id_map": {
                "_": [0],
                "^": [1],
                "$": [2],
                " ": [3],
                ".": [10],
                "a": [11],
                "d": [12],
                "i": [13],
                "l": [14],
                "n": [15],
                "p": [16],
                "t": [17],
                "w": [18],
                "z": [19],
                "ð": [20],
                "ɡ": [21],
                "ɪ": [22],
                "ɛ": [23],
                "ɑ": [24],
                "ə": [25],
                "ɹ": [26]
              }
            }
            "#,
    )
    .expect("voice config should parse");
    let sequence = PiperPhonemeSequence {
        phonemes: [
            "W", "IY", " ", "R", "EH", "P", "R", "IH", "Z", "EH", "N", "T", " ", "DH", "AH0", " ",
            "L", "AA", "L", "IY", "P", "AA", "P", " ", "G", "IH", "L", "D", "|",
        ]
        .into_iter()
        .map(|symbol| PiperPhoneme(symbol.to_string()))
        .collect(),
    };

    sequence
        .to_piper_ids_compatible(&config)
        .expect("sentence ARPAbet symbols should map to eSpeak Piper IDs");
}

#[test]
#[cfg(feature = "piper-compat")]
fn espeak_compatible_ids_map_arpabet_flap_symbol() {
    let config = PiperVoiceConfig::from_json_str(
        r#"
            {
              "audio": { "sample_rate": 22050 },
              "phoneme_id_map": {
                "_": [0],
                "^": [1],
                "$": [2],
                "b": [10],
                "l": [11],
                "ɑ": [12],
                "ə": [13],
                "ɾ": [14]
              }
            }
            "#,
    )
    .expect("voice config should parse");
    let sequence = PiperPhonemeSequence {
        phonemes: ["B", "AA", "DX", "AH0", "L"]
            .into_iter()
            .map(|symbol| PiperPhoneme(symbol.to_string()))
            .collect(),
    };

    let ids = sequence
        .to_piper_ids_compatible(&config)
        .expect("flapped Riper sequence should map to eSpeak Piper IDs");

    assert_eq!(
        ids,
        PiperIdSequence {
            ids: vec![1, 0, 10, 0, 12, 0, 14, 0, 13, 0, 11, 0, 2]
        }
    );
}

#[test]
#[cfg(feature = "piper-compat")]
fn audio_stats_computes_duration_rms_and_peak() {
    let stats = AudioStats::from_frames(
        &[AudioFrame {
            captured_at: ExactTimestamp::now(),
            sample_rate_hz: 20,
            channels: 1,
            samples: vec![0.0, 0.5, -1.0, 0.5],
            voice_signatures: Vec::new(),
        }],
        "test",
    )
    .expect("stats should compute");

    assert_eq!(stats.sample_rate_hz, 20);
    assert_eq!(stats.channels, 1);
    assert_eq!(stats.sample_count, 4);
    assert!((stats.duration_ms - 200.0).abs() < 0.0001);
    assert!((stats.rms - 0.6123724).abs() < 0.0001);
    assert!((stats.peak_abs - 1.0).abs() < 0.0001);
}
