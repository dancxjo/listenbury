use super::*;
use clap::Parser;
use std::path::PathBuf;

#[test]
fn transcribe_accepts_input_and_default_model() {
    let cli = Cli::try_parse_from(["listenbury", "transcribe", "welcome.wav"])
        .expect("transcribe should parse an input path and optional model path");

    let Some(Command::Transcribe(command)) = cli.command else {
        panic!("expected transcribe command");
    };
    assert_eq!(command.input_wav, Some(PathBuf::from("welcome.wav")));
    assert!(command.whisper_model.is_none());
    assert_eq!(command.seconds, 30);
    assert!(!command.until_ctrl_c);
    assert_eq!(command.vad, VadBackendOption::WebRtc);
}

#[test]
fn debug_loop_trace_parses_proposed_command() {
    let cli = Cli::try_parse_from(["listenbury", "debug", "loop-trace", "--duration", "20"])
        .expect("debug loop-trace should parse");

    let Some(Command::Debug { command }) = cli.command else {
        panic!("expected debug command");
    };
    let DebugCommand::LoopTrace(command) = command else {
        panic!("expected loop-trace command");
    };
    assert_eq!(command.duration, 20);
    assert_eq!(command.profile, LoopTraceProfile::Mock);
    assert_eq!(command.effective_profile(), LoopTraceProfile::Mock);
    assert_eq!(command.write, PathBuf::from("out/loop-trace.jsonl"));
    assert!(!command.json);
    assert!(!command.no_self_hearing);
}

#[test]
fn debug_loop_trace_ear_profile_parses_without_audio_hardware() {
    let cli = Cli::try_parse_from([
        "listenbury",
        "debug",
        "loop-trace",
        "--profile",
        "ear",
        "--duration",
        "20",
        "--mock-llm",
        "--mock-mouth",
    ])
    .expect("debug loop-trace ear profile should parse");

    let Some(Command::Debug { command }) = cli.command else {
        panic!("expected debug command");
    };
    let DebugCommand::LoopTrace(command) = command else {
        panic!("expected loop-trace command");
    };
    assert_eq!(command.profile, LoopTraceProfile::Ear);
    assert_eq!(command.effective_profile(), LoopTraceProfile::Ear);
    assert!(command.mock_llm);
    assert!(command.mock_mouth);
}

#[test]
fn debug_loop_trace_real_flags_imply_ear_profile() {
    let cli = Cli::try_parse_from([
        "listenbury",
        "debug",
        "loop-trace",
        "--duration",
        "20",
        "--real-mic",
        "--real-asr",
        "--mock-llm",
        "--mock-mouth",
    ])
    .expect("debug loop-trace real flags should parse");

    let Some(Command::Debug { command }) = cli.command else {
        panic!("expected debug command");
    };
    let DebugCommand::LoopTrace(command) = command else {
        panic!("expected loop-trace command");
    };
    assert_eq!(command.profile, LoopTraceProfile::Mock);
    assert_eq!(command.effective_profile(), LoopTraceProfile::Ear);
    assert!(command.real_mic);
    assert!(command.real_asr);
}

#[test]
fn debug_ear_scope_parses_default_outputs() {
    let cli = Cli::try_parse_from(["listenbury", "debug", "ear-scope", "--duration", "10"])
        .expect("debug ear-scope should parse");

    let Some(Command::Debug { command }) = cli.command else {
        panic!("expected debug command");
    };
    let DebugCommand::EarScope(command) = command else {
        panic!("expected ear-scope command");
    };
    assert_eq!(command.duration, 10);
    assert_eq!(command.output_wav, PathBuf::from("out/ear-scope.wav"));
    assert_eq!(command.output_png, PathBuf::from("out/ear-scope.png"));
    assert_eq!(command.output_jsonl, PathBuf::from("out/ear-scope.jsonl"));
    assert_eq!(command.vad, VadBackendOption::WebRtc);
}

#[test]
fn debug_think_say_parses_prompt_and_mocks() {
    let cli = Cli::try_parse_from([
        "listenbury",
        "debug",
        "think-say",
        "--mock-llm",
        "--mock-mouth",
        "Say hello.",
    ])
    .expect("debug think-say should parse");

    let Some(Command::Debug { command }) = cli.command else {
        panic!("expected debug command");
    };
    let DebugCommand::ThinkSay(command) = command else {
        panic!("expected think-say command");
    };
    assert!(command.mock_llm);
    assert!(command.mock_mouth);
    assert_eq!(command.mouth, ThinkSayMouthOption::Current);
    assert_eq!(command.max_tokens, 80);
    assert_eq!(command.prompt, ["Say hello."]);
}

#[test]
fn debug_think_say_accepts_mouth_and_dump_options() {
    let cli = Cli::try_parse_from([
        "listenbury",
        "debug",
        "think-say",
        "--mouth",
        "klatt",
        "--dump",
        "out/think-say.jsonl",
        "--max-tokens",
        "12",
        "Say hello.",
    ])
    .expect("debug think-say options should parse");

    let Some(Command::Debug { command }) = cli.command else {
        panic!("expected debug command");
    };
    let DebugCommand::ThinkSay(command) = command else {
        panic!("expected think-say command");
    };
    assert_eq!(command.mouth, ThinkSayMouthOption::Klatt);
    assert_eq!(command.dump, Some(PathBuf::from("out/think-say.jsonl")));
    assert_eq!(command.max_tokens, 12);
}

#[test]
fn diphone_forge_parses_top_level_command() {
    let cli = Cli::try_parse_from([
        "listenbury",
        "diphone",
        "forge",
        "--model",
        "voice.onnx",
        "--config",
        "voice.json",
        "--left",
        "h",
        "--right",
        "@",
    ])
    .expect("diphone forge should parse");

    let Some(Command::Diphone { command }) = cli.command else {
        panic!("expected top-level diphone command");
    };
    let DiphoneCommand::Forge(command) = command else {
        panic!("expected diphone forge subcommand");
    };
    assert_eq!(command.model, PathBuf::from("voice.onnx"));
    assert_eq!(command.config, PathBuf::from("voice.json"));
    assert_eq!(command.left, "h");
    assert_eq!(command.right, "@");
}

#[test]
fn diphone_cache_build_parses_inventory_and_force() {
    let cli = Cli::try_parse_from([
        "listenbury",
        "diphone",
        "cache-build",
        "--model",
        "voice.onnx",
        "--config",
        "voice.json",
        "--inventory",
        "en-us-basic",
        "--force",
    ])
    .expect("diphone cache-build should parse");

    let Some(Command::Diphone { command }) = cli.command else {
        panic!("expected top-level diphone command");
    };
    let DiphoneCommand::CacheBuild(command) = command else {
        panic!("expected cache-build subcommand");
    };
    assert_eq!(command.inventory, "en-us-basic");
    assert!(command.force);
}

#[test]
fn diphone_mbrola_database_parses_pcm_dir_and_out() {
    let cli = Cli::try_parse_from([
        "listenbury",
        "diphone",
        "mbrola-database",
        "--pcm-dir",
        "diphone-cache",
        "--out",
        "voice/voice",
        "--sample-rate",
        "22050",
    ])
    .expect("diphone mbrola-database should parse");

    let Some(Command::Diphone { command }) = cli.command else {
        panic!("expected top-level diphone command");
    };
    let DiphoneCommand::MbrolaDatabase(command) = command else {
        panic!("expected mbrola-database subcommand");
    };
    assert_eq!(command.pcm_dir, PathBuf::from("diphone-cache"));
    assert_eq!(command.out, PathBuf::from("voice/voice"));
    assert_eq!(command.sample_rate, Some(22_050));
    assert_eq!(command.mbr_period, 80);
}

#[test]
fn diphone_wizard_parses_model_voice_and_defaults() {
    let cli = Cli::try_parse_from(["listenbury", "diphone", "wizard", "ryan.onnx", "ryan"])
        .expect("diphone wizard should parse model and voice name");

    let Some(Command::Diphone { command }) = cli.command else {
        panic!("expected top-level diphone command");
    };
    let DiphoneCommand::Wizard(command) = command else {
        panic!("expected wizard subcommand");
    };
    assert_eq!(command.model, PathBuf::from("ryan.onnx"));
    assert_eq!(command.voice, PathBuf::from("ryan"));
    assert!(command.config.is_none());
    assert_eq!(command.inventory, "en-us-basic");
}

#[test]
fn diphone_audit_plan_parses_optional_mbrola_voice() {
    let cli = Cli::try_parse_from([
        "listenbury",
        "diphone",
        "audit-plan",
        "--model",
        "voice.onnx",
        "--config",
        "voice.json",
        "--mbrola-voice",
        "data/mbrola/us3/us3",
        "--plan",
        "out/example.pho",
    ])
    .expect("diphone audit-plan should parse");

    let Some(Command::Diphone { command }) = cli.command else {
        panic!("expected top-level diphone command");
    };
    let DiphoneCommand::AuditPlan(command) = command else {
        panic!("expected audit-plan subcommand");
    };
    assert_eq!(
        command.mbrola_voice,
        Some(PathBuf::from("data/mbrola/us3/us3"))
    );
    assert_eq!(command.plan, PathBuf::from("out/example.pho"));
}

#[test]
fn vad_calibrate_room_parses_with_defaults() {
    let cli = Cli::try_parse_from(["listenbury", "vad", "calibrate-room"])
        .expect("vad calibrate-room should parse");

    let Some(Command::Vad {
        command: VadCommand::CalibrateRoom(command),
    }) = cli.command
    else {
        panic!("expected vad calibrate-room command");
    };
    assert!(command.audio.is_none());
    assert_eq!(command.seconds, 20);
    assert_eq!(command.vad, VadBackendOption::Energy);
}

#[test]
fn vad_compare_parses_backend_list_and_labels() {
    let cli = Cli::try_parse_from([
        "listenbury",
        "vad",
        "compare",
        "--audio",
        "samples/hello-16k-mono.wav",
        "--labels",
        "samples/room-labels.json",
        "--backends",
        "energy,webrtc",
    ])
    .expect("vad compare should parse backend list");

    let Some(Command::Vad {
        command: VadCommand::Compare(command),
    }) = cli.command
    else {
        panic!("expected vad compare command");
    };
    assert_eq!(command.audio, PathBuf::from("samples/hello-16k-mono.wav"));
    assert_eq!(
        command.labels,
        Some(PathBuf::from("samples/room-labels.json"))
    );
    assert_eq!(
        command.backends,
        vec![VadBackendOption::Energy, VadBackendOption::WebRtc]
    );
}

#[test]
fn transcribe_without_input_uses_mic_defaults() {
    let cli = Cli::try_parse_from(["listenbury", "transcribe"])
        .expect("transcribe should parse without input for mic capture");

    let Some(Command::Transcribe(command)) = cli.command else {
        panic!("expected transcribe command");
    };
    assert!(command.input_wav.is_none());
    assert!(command.whisper_model.is_none());
    assert_eq!(command.seconds, 30);
    assert!(!command.until_ctrl_c);
    assert_eq!(command.vad, VadBackendOption::WebRtc);
}

#[test]
fn transcribe_without_input_accepts_mic_options() {
    let cli = Cli::try_parse_from([
        "listenbury",
        "transcribe",
        "--seconds",
        "5",
        "--until-ctrl-c",
        "--model-path",
        "models/ggml-base.en.bin",
        "--vad",
        "webrtc",
    ])
    .expect("transcribe should parse mic capture options");

    let Some(Command::Transcribe(command)) = cli.command else {
        panic!("expected transcribe command");
    };
    assert!(command.input_wav.is_none());
    assert_eq!(command.seconds, 5);
    assert!(command.until_ctrl_c);
    assert_eq!(
        command.whisper_model,
        Some(PathBuf::from("models/ggml-base.en.bin"))
    );
    assert_eq!(command.vad, VadBackendOption::WebRtc);
}

#[test]
fn transcribe_web_accepts_refinement_options() {
    let cli = Cli::try_parse_from([
        "listenbury",
        "transcribe",
        "--web",
        "--web-port",
        "0",
        "--refine-whisper-model",
        "models/ggml-large-v3-turbo.bin",
        "--refine-window-seconds",
        "120",
        "--refine-interval-ms",
        "2000",
    ])
    .expect("transcribe web should parse refinement options");

    let Some(Command::Transcribe(command)) = cli.command else {
        panic!("expected transcribe command");
    };
    assert!(command.web);
    assert_eq!(command.web_port, 0);
    assert_eq!(
        command.refine_whisper_model,
        Some(PathBuf::from("models/ggml-large-v3-turbo.bin"))
    );
    assert_eq!(command.refine_window_seconds, 120);
    assert_eq!(command.refine_interval_ms, 2_000);
}

#[test]
fn say_accepts_text_and_default_piper_options() {
    let cli = Cli::try_parse_from(["listenbury", "say", "hello", "there"])
        .expect("say should parse text and optional Piper options");

    let Some(Command::Say(command)) = cli.command else {
        panic!("expected say command");
    };
    assert!(command.piper_bin.is_none());
    assert!(command.piper_voice.is_none());
    assert!(command.output_wav.is_none());
    assert!(!command.piper);
    assert!(!command.klatt);
    assert!(!command.diphone);
    assert!(command.mbrola_voice.is_none());
    assert_eq!(command.words, ["hello", "there"]);
}

#[test]
fn say_accepts_output_wav_override() {
    let cli = Cli::try_parse_from([
        "listenbury",
        "say",
        "--output-wav",
        "out/test.wav",
        "hello",
        "there",
    ])
    .expect("say should parse an optional output WAV path");

    let Some(Command::Say(command)) = cli.command else {
        panic!("expected say command");
    };
    assert_eq!(command.output_wav, Some(PathBuf::from("out/test.wav")));
    assert!(!command.piper);
    assert!(!command.klatt);
    assert!(!command.diphone);
    assert!(command.mbrola_voice.is_none());
    assert_eq!(command.words, ["hello", "there"]);
}

#[test]
fn say_accepts_riper_flag_before_text() {
    let cli = Cli::try_parse_from(["listenbury", "say", "--riper", "hello", "there"])
        .expect("--riper should be accepted as an explicit default route");

    let Some(Command::Say(command)) = cli.command else {
        panic!("expected say command");
    };
    assert!(command.riper);
    assert_eq!(command.words, ["hello", "there"]);
}

#[test]
fn say_keeps_trailing_riper_flag_for_runtime_normalization() {
    let cli = Cli::try_parse_from(["listenbury", "say", "hello", "there", "--riper"])
        .expect("say trailing text is collected before SayArgs normalization");

    let Some(Command::Say(command)) = cli.command else {
        panic!("expected say command");
    };
    assert!(!command.piper);
    assert!(!command.klatt);
    assert_eq!(command.words, ["hello", "there", "--riper"]);
}

#[test]
fn say_accepts_klatt() {
    let cli = Cli::try_parse_from(["listenbury", "say", "--klatt", "hello", "there"])
        .expect("say should parse Klatt as a default Riper-path backend");
    let Some(Command::Say(command)) = cli.command else {
        panic!("expected say command");
    };
    assert!(command.klatt);
}

#[test]
fn say_accepts_dump_phonemes() {
    let cli = Cli::try_parse_from(["listenbury", "say", "--dump-phonemes", "hello"])
        .expect("say should parse phoneme dump flag");

    let Some(Command::Say(command)) = cli.command else {
        panic!("expected say command");
    };
    assert!(command.dump_phonemes);
    assert_eq!(command.words, ["hello"]);
}

#[test]
fn say_accepts_dump_phone_plan() {
    let cli = Cli::try_parse_from(["listenbury", "say", "--dump-phone-plan", "hello"])
        .expect("say should parse phone plan dump flag");

    let Some(Command::Say(command)) = cli.command else {
        panic!("expected say command");
    };
    assert!(command.dump_phone_plan);
    assert_eq!(command.words, ["hello"]);
}

#[test]
fn say_accepts_dump_piper_tensors() {
    let cli = Cli::try_parse_from(["listenbury", "say", "--dump-piper-tensors", "hello"])
        .expect("say should parse Piper tensor dump flag");

    let Some(Command::Say(command)) = cli.command else {
        panic!("expected say command");
    };
    assert!(command.dump_piper_tensors);
    assert_eq!(command.words, ["hello"]);
}

#[test]
fn say_accepts_hifigan() {
    let cli = Cli::try_parse_from(["listenbury", "say", "--hifigan", "hello"])
        .expect("say should parse hifigan");

    let Some(Command::Say(command)) = cli.command else {
        panic!("expected say command");
    };
    assert!(command.hifigan);
    assert!(!command.klatt);
    assert!(!command.diphone);
    assert_eq!(command.words, ["hello"]);
}

#[test]
fn say_accepts_skip_gan() {
    let cli = Cli::try_parse_from(["listenbury", "say", "--hifigan", "--skip-gan", "hello"])
        .expect("say should parse skip-gan as a mel debug switch");

    let Some(Command::Say(command)) = cli.command else {
        panic!("expected say command");
    };
    assert!(command.hifigan);
    assert!(command.skip_gan);
    assert_eq!(command.words, ["hello"]);
}

#[test]
fn say_accepts_hifigan_fallback_alias() {
    let cli = Cli::try_parse_from([
        "listenbury",
        "say",
        "--hifigan",
        "--hifigan-fallback",
        "hello",
    ])
    .expect("say should parse hifigan-fallback as an explicit mel debug switch");

    let Some(Command::Say(command)) = cli.command else {
        panic!("expected say command");
    };
    assert!(command.hifigan);
    assert!(command.skip_gan);
    assert_eq!(command.words, ["hello"]);
}

#[test]
fn say_rejects_skip_gan_without_hifigan() {
    let error = Cli::try_parse_from(["listenbury", "say", "--skip-gan", "hello"])
        .expect_err("--skip-gan should require --hifigan");
    assert!(error.to_string().contains("--hifigan"));
}

#[test]
fn say_accepts_diphone_voice() {
    let cli = Cli::try_parse_from(["listenbury", "say", "--diphone", "hello"])
        .expect("say should parse diphone as a Riper backend voice");

    let Some(Command::Say(command)) = cli.command else {
        panic!("expected say command");
    };
    assert!(!command.klatt);
    assert!(command.diphone);
    assert!(command.mbrola_voice.is_none());
    assert_eq!(command.words, ["hello"]);
}

#[test]
fn say_rejects_mbrola_flag() {
    let error = Cli::try_parse_from(["listenbury", "say", "--mbrola", "hello"])
        .expect_err("--mbrola should be renamed");
    assert!(error.to_string().contains("--mbrola"));
}

#[test]
fn say_accepts_rp_shorthand() {
    let cli = Cli::try_parse_from(["listenbury", "say", "--rp", "hello"])
        .expect("say should parse RP shorthand");

    let Some(Command::Say(command)) = cli.command else {
        panic!("expected say command");
    };
    assert!(command.rp);
    assert!(!command.piper);
    assert!(!command.klatt);
    assert!(!command.diphone);
    assert!(command.mbrola_voice.is_none());
    assert_eq!(command.words, ["hello"]);
}

#[test]
fn say_rejects_rp_with_klatt() {
    let error = Cli::try_parse_from(["listenbury", "say", "--rp", "--klatt", "hello"])
        .expect_err("say should reject RP and Klatt together");
    assert!(
        error.to_string().contains("--rp"),
        "unexpected error: {error}"
    );
}

#[test]
fn say_rejects_rp_with_explicit_mbrola_voice() {
    let error = Cli::try_parse_from([
        "listenbury",
        "say",
        "--rp",
        "--diphone",
        "--diphone-voice",
        "data/mbrola/us3/us3",
        "hello",
    ])
    .expect_err("say should reject RP with a different explicit MBROLA voice");
    assert!(
        error.to_string().contains("--rp"),
        "unexpected error: {error}"
    );
}

#[test]
fn say_accepts_diphone_without_text_for_smoke_demo() {
    let cli = Cli::try_parse_from(["listenbury", "say", "--diphone"])
        .expect("say should parse diphone smoke command without text");

    let Some(Command::Say(command)) = cli.command else {
        panic!("expected say command");
    };
    assert!(command.diphone);
    assert!(command.words.is_empty());
}

#[test]
fn say_accepts_diphone_explicit_voice() {
    let cli = Cli::try_parse_from([
        "listenbury",
        "say",
        "--diphone",
        "--diphone-voice",
        "data/mbrola/en1/en1",
        "hello",
    ])
    .expect("say should parse explicit diphone voice");

    let Some(Command::Say(command)) = cli.command else {
        panic!("expected say command");
    };
    assert!(command.diphone);
    assert_eq!(
        command.mbrola_voice,
        Some(PathBuf::from("data/mbrola/en1/en1"))
    );
}

#[test]
fn mbrola_render_accepts_raw_pho_inputs() {
    let cli = Cli::try_parse_from([
        "listenbury",
        "mbrola-render",
        "--voice",
        "voices/us1",
        "--phones",
        "examples/mbrola/hello-us3.pho",
        "--out",
        "out/hello.wav",
    ])
    .expect("mbrola-render should parse voice, phones, and output paths");

    let Some(Command::MbrolaRender(command)) = cli.command else {
        panic!("expected mbrola-render command");
    };
    assert_eq!(command.voice, PathBuf::from("voices/us1"));
    assert_eq!(
        command.phones,
        PathBuf::from("examples/mbrola/hello-us3.pho")
    );
    assert_eq!(command.out, PathBuf::from("out/hello.wav"));
}

#[test]
fn riper_compare_parses_text_and_optional_outputs() {
    let cli = Cli::try_parse_from([
        "listenbury",
        "riper-compare",
        "--process-output-wav",
        "out/process.wav",
        "--riper-output-wav",
        "out/riper.wav",
        "I",
        "see.",
    ])
    .expect("riper-compare should parse text and optional output paths");

    let Some(Command::RiperCompare(command)) = cli.command else {
        panic!("expected riper-compare command");
    };
    assert_eq!(
        command.process_output_wav,
        Some(PathBuf::from("out/process.wav"))
    );
    assert_eq!(
        command.riper_output_wav,
        Some(PathBuf::from("out/riper.wav"))
    );
    assert_eq!(command.words, ["I", "see."]);
}

#[test]
fn riper_compare_accepts_default_utterance() {
    let cli = Cli::try_parse_from(["listenbury", "riper-compare"])
        .expect("riper-compare should allow its default utterance");

    let Some(Command::RiperCompare(command)) = cli.command else {
        panic!("expected riper-compare command");
    };
    assert!(command.words.is_empty());
}

#[test]
fn echo_accepts_wav_output_and_comparison_paths() {
    let cli = Cli::try_parse_from([
        "listenbury",
        "echo",
        "fixtures/in.wav",
        "--whisper-model",
        "models/ggml-base.en.bin",
        "--model-path",
        "voices/en_US.onnx",
        "--riper-config",
        "voices/en_US.onnx.json",
        "--output-wav",
        "out/echo.wav",
        "--comparison-json",
        "out/echo.json",
    ])
    .expect("echo should parse offline echo options");

    let Some(Command::Echo(command)) = cli.command else {
        panic!("expected echo command");
    };
    assert_eq!(command.input_wav, PathBuf::from("fixtures/in.wav"));
    assert_eq!(
        command.whisper_model,
        Some(PathBuf::from("models/ggml-base.en.bin"))
    );
    assert_eq!(
        command.piper_voice,
        Some(PathBuf::from("voices/en_US.onnx"))
    );
    assert_eq!(
        command.riper_config,
        Some(PathBuf::from("voices/en_US.onnx.json"))
    );
    assert_eq!(command.output_wav, Some(PathBuf::from("out/echo.wav")));
    assert_eq!(
        command.comparison_json,
        Some(PathBuf::from("out/echo.json"))
    );
}

#[test]
fn record_wav_parses_seconds_and_output_path() {
    let cli = Cli::try_parse_from([
        "listenbury",
        "dev",
        "record-wav",
        "out/mic.wav",
        "--seconds",
        "5",
    ])
    .expect("record-wav should parse");

    let Some(Command::Dev {
        command: DevCommand::RecordWav(command),
    }) = cli.command
    else {
        panic!("expected record-wav command");
    };
    assert_eq!(command.output_wav, PathBuf::from("out/mic.wav"));
    assert_eq!(command.seconds, 5);
}

#[test]
fn play_wav_parses_input_path() {
    let cli = Cli::try_parse_from([
        "listenbury",
        "dev",
        "play-wav",
        "out/listenbury-round-trip.wav",
    ])
    .expect("play-wav should parse");

    let Some(Command::Dev {
        command: DevCommand::PlayWav(command),
    }) = cli.command
    else {
        panic!("expected play-wav command");
    };
    assert_eq!(
        command.input_wav,
        PathBuf::from("out/listenbury-round-trip.wav")
    );
}

#[test]
fn vad_trace_parses_input_and_jsonl() {
    let cli = Cli::try_parse_from([
        "listenbury",
        "dev",
        "vad-trace",
        "samples/silence-16k-mono.wav",
        "--jsonl",
        "out/vad-trace.jsonl",
    ])
    .expect("vad-trace should parse");

    let Some(Command::Dev {
        command: DevCommand::VadTrace(command),
    }) = cli.command
    else {
        panic!("expected vad-trace command");
    };
    assert_eq!(
        command.input_wav,
        PathBuf::from("samples/silence-16k-mono.wav")
    );
    assert_eq!(command.jsonl, Some(PathBuf::from("out/vad-trace.jsonl")));
    assert_eq!(command.vad, VadBackendOption::WebRtc);
}

#[test]
fn breath_transcribe_parses_config() {
    let cli = Cli::try_parse_from([
        "listenbury",
        "dev",
        "breath-transcribe",
        "samples/hello-16k-mono.wav",
        "--pre-roll-ms",
        "60",
        "--trailing-pad-ms",
        "120",
        "--min-group-ms",
        "80",
        "--max-group-ms",
        "3000",
    ])
    .expect("breath-transcribe should parse");

    let Some(Command::Dev {
        command: DevCommand::BreathTranscribe(command),
    }) = cli.command
    else {
        panic!("expected breath-transcribe command");
    };
    assert_eq!(
        command.input_wav,
        PathBuf::from("samples/hello-16k-mono.wav")
    );
    assert_eq!(command.pre_roll_ms, 60);
    assert_eq!(command.trailing_pad_ms, 120);
    assert_eq!(command.min_group_ms, 80);
    assert_eq!(command.max_group_ms, 3000);
}

#[test]
fn mic_transcribe_parses_seconds_by_default() {
    let cli = Cli::try_parse_from(["listenbury", "dev", "mic-transcribe"])
        .expect("mic-transcribe should parse with defaults");

    let Some(Command::Dev {
        command: DevCommand::MicTranscribe(command),
    }) = cli.command
    else {
        panic!("expected mic-transcribe command");
    };
    assert_eq!(command.seconds, 30);
    assert!(!command.until_ctrl_c);
    assert!(command.whisper_model.is_none());
    assert_eq!(command.vad, VadBackendOption::WebRtc);
}

#[test]
fn mic_transcribe_parses_until_ctrl_c_and_model() {
    let cli = Cli::try_parse_from([
        "listenbury",
        "dev",
        "mic-transcribe",
        "--until-ctrl-c",
        "--model-path",
        "models/ggml-base.en.bin",
    ])
    .expect("mic-transcribe should parse until-ctrl-c and model path");

    let Some(Command::Dev {
        command: DevCommand::MicTranscribe(command),
    }) = cli.command
    else {
        panic!("expected mic-transcribe command");
    };
    assert!(command.until_ctrl_c);
    assert_eq!(
        command.whisper_model,
        Some(PathBuf::from("models/ggml-base.en.bin"))
    );
    assert_eq!(command.vad, VadBackendOption::WebRtc);
}

#[test]
fn live_half_duplex_parses_defaults() {
    let cli =
        Cli::try_parse_from(["listenbury", "listen"]).expect("listen should parse with defaults");

    let Some(Command::Listen(command)) = cli.command else {
        panic!("expected listen command");
    };
    assert_eq!(command.seconds, None);
    assert!(command.jsonl.is_none());
    assert_eq!(command.model_profile, ModelProfile::Tiny);
    assert!(!command.no_backchannels);
    assert_eq!(command.vad, VadBackendOption::WebRtc);
    assert_eq!(command.context_size, 8192);
    assert_eq!(command.reserved_generation_tokens, 512);
    assert!(!command.hifigan);
    assert!(command.hifigan_model.is_none());
    assert!(!command.skip_gan);
    assert!(!command.web);
    assert!(!command.native_video);
    assert_eq!(command.video_device, PathBuf::from("/dev/video0"));
    assert_eq!(command.video_width, 320);
    assert_eq!(command.video_height, 240);
    assert_eq!(command.video_fps, 2);
    assert!(!command.retain_video_images);
    assert!(!command.duplex);
    assert_eq!(command.web_host, "127.0.0.1");
    assert_eq!(command.web_port, 8787);
}

#[test]
fn listen_accepts_hifigan_voice_flags() {
    let cli = Cli::try_parse_from([
        "listenbury",
        "listen",
        "--web",
        "--hifigan",
        "--hifigan-model",
        "models/hifigan.onnx",
    ])
    .expect("listen should parse hifigan voice flags");

    let Some(Command::Listen(command)) = cli.command else {
        panic!("expected listen command");
    };
    assert!(command.web);
    assert!(command.hifigan);
    assert_eq!(
        command.hifigan_model,
        Some(PathBuf::from("models/hifigan.onnx"))
    );
    assert!(!command.skip_gan);
}

#[test]
fn listen_parses_web_flag() {
    let cli = Cli::try_parse_from([
        "listenbury",
        "listen",
        "--web",
        "--web-host",
        "0.0.0.0",
        "--web-port",
        "9000",
    ])
    .expect("listen should parse --web flags");

    let Some(Command::Listen(command)) = cli.command else {
        panic!("expected listen command");
    };
    assert!(command.web);
    assert_eq!(command.web_host, "0.0.0.0");
    assert_eq!(command.web_port, 9000);
}

#[test]
fn listen_parses_duplex_web_flag() {
    let cli = Cli::try_parse_from([
        "listenbury",
        "listen",
        "--web",
        "--duplex",
        "--jsonl",
        "out/duplex-live.jsonl",
        "--llm-model",
        "models/pete.gguf",
    ])
    .expect("listen should parse --web --duplex");

    let Some(Command::Listen(command)) = cli.command else {
        panic!("expected listen command");
    };
    assert!(command.web);
    assert!(command.duplex);
    assert_eq!(command.jsonl, Some(PathBuf::from("out/duplex-live.jsonl")));
    assert_eq!(command.llm_model, Some(PathBuf::from("models/pete.gguf")));
}

#[test]
fn listen_duplex_maps_to_continue_pipeline_options() {
    let cli = Cli::try_parse_from([
        "listenbury",
        "listen",
        "--web",
        "--duplex",
        "--web-host",
        "0.0.0.0",
        "--web-port",
        "9000",
        "--jsonl",
        "out/duplex-live.jsonl",
        "--whisper-model",
        "models/ggml-base.en.bin",
        "--piper-voice",
        "voices/pete.onnx",
        "--hifigan",
        "--skip-gan",
        "--vad",
        "energy",
        "--no-backchannels",
        "--reserved-generation-tokens",
        "768",
        "--native-video",
        "--video-device",
        "/dev/video2",
        "--video-width",
        "640",
        "--video-height",
        "480",
        "--video-fps",
        "5",
        "--retain-video-images",
    ])
    .expect("listen should parse duplex pipeline options");

    let Some(Command::Listen(command)) = cli.command else {
        panic!("expected listen command");
    };
    let command = live_session::continue_command_from_listen_command(command);
    assert!(command.web);
    assert_eq!(command.web_host, "0.0.0.0");
    assert_eq!(command.web_port, 9000);
    assert_eq!(command.jsonl, Some(PathBuf::from("out/duplex-live.jsonl")));
    assert_eq!(
        command.whisper_model,
        Some(PathBuf::from("models/ggml-base.en.bin"))
    );
    assert_eq!(command.piper_voice, Some(PathBuf::from("voices/pete.onnx")));
    assert!(command.hifigan);
    assert!(command.skip_gan);
    assert_eq!(command.vad, VadBackendOption::Energy);
    assert!(command.no_backchannels);
    assert_eq!(command.reserved_generation_tokens, 768);
    assert!(command.native_video);
    assert_eq!(command.video_device, PathBuf::from("/dev/video2"));
    assert_eq!(command.video_width, 640);
    assert_eq!(command.video_height, 480);
    assert_eq!(command.video_fps, 5);
    assert!(command.retain_video_images);
    assert!(command.duplex_trace_scenario.is_none());
}

#[test]
fn listen_duplex_maps_to_live_session_config() {
    let cli = Cli::try_parse_from([
        "listenbury",
        "listen",
        "--duplex",
        "--web",
        "--web-host",
        "0.0.0.0",
        "--web-port",
        "9000",
        "--jsonl",
        "out/duplex-live.jsonl",
        "--whisper-model",
        "models/ggml-base.en.bin",
        "--piper-voice",
        "voices/pete.onnx",
        "--vad",
        "energy",
    ])
    .expect("listen should parse duplex live session options");

    let Some(Command::Listen(command)) = cli.command else {
        panic!("expected listen command");
    };
    let config = live_session::LiveSessionConfig::from_listen_command(command);
    assert_eq!(config.mode, live_session::LiveSessionMode::Duplex);
    assert_eq!(config.input.vad, VadBackendOption::Energy);
    assert_eq!(
        config.asr.whisper_model,
        Some(PathBuf::from("models/ggml-base.en.bin"))
    );
    assert_eq!(
        config.mouth_playback.piper_voice,
        Some(PathBuf::from("voices/pete.onnx"))
    );
    assert!(!config.mouth_playback.hifigan);
    assert!(config.mouth_playback.hifigan_model.is_none());
    assert!(!config.mouth_playback.skip_gan);
    assert_eq!(config.web_bridge.host, "0.0.0.0");
    assert_eq!(config.web_bridge.port, 9000);
    assert_eq!(
        config.tracing.jsonl,
        Some(PathBuf::from("out/duplex-live.jsonl"))
    );
}

#[test]
fn live_half_duplex_parses_optional_flags() {
    let cli = Cli::try_parse_from([
        "listenbury",
        "listen",
        "--seconds",
        "12",
        "--jsonl",
        "out/live-trace.jsonl",
        "--model-profile",
        "tiny",
        "--no-backchannels",
        "--context-size",
        "16384",
        "--reserved-generation-tokens",
        "1024",
        "--native-video",
        "--video-device",
        "/dev/video2",
        "--video-width",
        "640",
        "--video-height",
        "480",
        "--video-fps",
        "5",
        "--retain-video-images",
    ])
    .expect("listen should parse optional flags");

    let Some(Command::Listen(command)) = cli.command else {
        panic!("expected listen command");
    };
    assert_eq!(command.seconds, Some(12));
    assert_eq!(command.jsonl, Some(PathBuf::from("out/live-trace.jsonl")));
    assert_eq!(command.model_profile, ModelProfile::Tiny);
    assert!(command.no_backchannels);
    assert_eq!(command.context_size, 16384);
    assert_eq!(command.reserved_generation_tokens, 1024);
    assert!(command.native_video);
    assert_eq!(command.video_device, PathBuf::from("/dev/video2"));
    assert_eq!(command.video_width, 640);
    assert_eq!(command.video_height, 480);
    assert_eq!(command.video_fps, 5);
    assert!(command.retain_video_images);
    assert_eq!(command.vad, VadBackendOption::WebRtc);
}

#[test]
fn vad_options_parse_for_vad_trace_mic_and_live_half_duplex() {
    let trace = Cli::try_parse_from([
        "listenbury",
        "dev",
        "vad-trace",
        "samples/hello-16k-mono.wav",
        "--vad",
        "webrtc",
    ])
    .expect("vad-trace should parse --vad webrtc");
    let Some(Command::Dev {
        command: DevCommand::VadTrace(trace_command),
    }) = trace.command
    else {
        panic!("expected vad-trace command");
    };
    assert_eq!(trace_command.vad, VadBackendOption::WebRtc);

    let mic = Cli::try_parse_from(["listenbury", "dev", "mic-transcribe", "--vad", "silero"])
        .expect("mic-transcribe should parse --vad silero");
    let Some(Command::Dev {
        command: DevCommand::MicTranscribe(mic_command),
    }) = mic.command
    else {
        panic!("expected mic-transcribe command");
    };
    assert_eq!(mic_command.vad, VadBackendOption::Silero);

    let live = Cli::try_parse_from(["listenbury", "listen", "--vad", "energy"])
        .expect("listen should parse --vad energy");
    let Some(Command::Listen(live_command)) = live.command else {
        panic!("expected listen command");
    };
    assert_eq!(live_command.vad, VadBackendOption::Energy);
}

#[test]
fn vad_profile_parses_for_live_mic_trace_and_continue() {
    let profile = PathBuf::from("out/vad-profile.toml");

    let live = Cli::try_parse_from([
        "listenbury",
        "listen",
        "--vad-profile",
        "out/vad-profile.toml",
    ])
    .expect("listen should parse --vad-profile");
    let Some(Command::Listen(live_command)) = live.command else {
        panic!("expected listen command");
    };
    assert_eq!(live_command.vad_profile, Some(profile.clone()));

    let mic = Cli::try_parse_from([
        "listenbury",
        "dev",
        "mic-transcribe",
        "--vad-profile",
        "out/vad-profile.toml",
    ])
    .expect("mic-transcribe should parse --vad-profile");
    let Some(Command::Dev {
        command: DevCommand::MicTranscribe(mic_command),
    }) = mic.command
    else {
        panic!("expected mic-transcribe command");
    };
    assert_eq!(mic_command.vad_profile, Some(profile.clone()));

    let trace = Cli::try_parse_from([
        "listenbury",
        "dev",
        "vad-trace",
        "samples/hello-16k-mono.wav",
        "--vad-profile",
        "out/vad-profile.toml",
    ])
    .expect("vad-trace should parse --vad-profile");
    let Some(Command::Dev {
        command: DevCommand::VadTrace(trace_command),
    }) = trace.command
    else {
        panic!("expected vad-trace command");
    };
    assert_eq!(trace_command.vad_profile, Some(profile.clone()));

    let continue_command = Cli::try_parse_from([
        "listenbury",
        "dev",
        "continue",
        "--vad-profile",
        "out/vad-profile.toml",
    ])
    .expect("dev continue should parse --vad-profile");
    let Some(Command::Dev {
        command: DevCommand::Continue(continue_command),
    }) = continue_command.command
    else {
        panic!("expected continue command");
    };
    assert_eq!(continue_command.vad_profile, Some(profile));
}

#[test]
fn dogfood_two_parses_defaults() {
    let cli = Cli::try_parse_from(["listenbury", "dev", "dogfood-two"])
        .expect("dogfood-two should parse with all defaults");

    let Some(Command::Dev {
        command: DevCommand::DogfoodTwo(command),
    }) = cli.command
    else {
        panic!("expected dogfood-two command");
    };
    assert_eq!(command.seed, "Hello.");
    assert_eq!(command.turns, 6);
    assert_eq!(command.max_tokens, 128);
    assert!(command.whisper_model.is_none());
    assert!(command.llm_model.is_none());
    assert!(command.piper_bin.is_none());
    assert!(command.piper_voice_a.is_none());
    assert!(command.piper_voice_b.is_none());
    assert!(command.jsonl.is_none());
    assert!(command.save_audio_dir.is_none());
}

#[test]
fn dev_continue_parses_defaults_and_prompt() {
    let cli = Cli::try_parse_from(["listenbury", "dev", "continue", "seed", "prompt"])
        .expect("dev continue should parse with a seed prompt");

    let Some(Command::Dev {
        command: DevCommand::Continue(command),
    }) = cli.command
    else {
        panic!("expected continue command");
    };
    assert!(command.llm_model.is_none());
    assert!(command.piper_bin.is_none());
    assert!(command.piper_voice.is_none());
    assert!(command.whisper_model.is_none());
    assert_eq!(command.vad, VadBackendOption::WebRtc);
    assert_eq!(command.mode, PromptMode::Raw);
    assert_eq!(command.max_tokens, None);
    assert_eq!(command.context_size, 8192);
    assert_eq!(command.verbatim_turns, 8);
    assert_eq!(command.tts_vad_pause_ms, 250);
    assert_eq!(command.tts_vad_listen_ms, 700);
    assert!(!command.web);
    assert_eq!(command.web_host, "127.0.0.1");
    assert_eq!(command.web_port, 8787);
    assert!(command.duplex_trace_scenario.is_none());
    assert!(command.jsonl.is_none());
    assert_eq!(command.prompt, ["seed", "prompt"]);
}

#[test]
fn dev_continue_accepts_optional_token_cap() {
    let cli = Cli::try_parse_from(["listenbury", "dev", "continue", "--max-tokens", "64"])
        .expect("dev continue should parse an optional token cap");

    let Some(Command::Dev {
        command: DevCommand::Continue(command),
    }) = cli.command
    else {
        panic!("expected continue command");
    };
    assert_eq!(command.max_tokens, Some(64));
}

#[test]
fn dev_continue_accepts_web_flags() {
    let cli = Cli::try_parse_from([
        "listenbury",
        "dev",
        "continue",
        "--web",
        "--web-host",
        "0.0.0.0",
        "--web-port",
        "9000",
    ])
    .expect("dev continue should parse web flags");

    let Some(Command::Dev {
        command: DevCommand::Continue(command),
    }) = cli.command
    else {
        panic!("expected continue command");
    };
    assert!(command.web);
    assert_eq!(command.web_host, "0.0.0.0");
    assert_eq!(command.web_port, 9000);
}

#[test]
fn dev_continue_accepts_mouth_overrides() {
    let cli = Cli::try_parse_from([
        "listenbury",
        "dev",
        "continue",
        "--piper-bin",
        "/usr/bin/piper",
        "--piper-voice",
        "voices/pete.onnx",
    ])
    .expect("dev continue should parse optional mouth overrides");

    let Some(Command::Dev {
        command: DevCommand::Continue(command),
    }) = cli.command
    else {
        panic!("expected continue command");
    };
    assert_eq!(command.piper_bin, Some(PathBuf::from("/usr/bin/piper")));
    assert_eq!(command.piper_voice, Some(PathBuf::from("voices/pete.onnx")));
}

#[test]
fn dev_continue_accepts_ear_overrides() {
    let cli = Cli::try_parse_from([
        "listenbury",
        "dev",
        "continue",
        "--whisper-model",
        "models/ggml-base.en.bin",
        "--vad",
        "energy",
        "--tts-vad-pause-ms",
        "180",
        "--tts-vad-listen-ms",
        "900",
    ])
    .expect("dev continue should parse optional ear overrides");

    let Some(Command::Dev {
        command: DevCommand::Continue(command),
    }) = cli.command
    else {
        panic!("expected continue command");
    };
    assert_eq!(
        command.whisper_model,
        Some(PathBuf::from("models/ggml-base.en.bin"))
    );
    assert_eq!(command.vad, VadBackendOption::Energy);
    assert_eq!(command.tts_vad_pause_ms, 180);
    assert_eq!(command.tts_vad_listen_ms, 900);
}

#[test]
fn dev_continue_accepts_duplex_trace_scenario_and_jsonl() {
    let cli = Cli::try_parse_from([
        "listenbury",
        "dev",
        "continue",
        "--duplex-trace-scenario",
        "overlap-yield",
        "--jsonl",
        "out/duplex-trace.jsonl",
    ])
    .expect("dev continue should parse duplex trace scenario options");

    let Some(Command::Dev {
        command: DevCommand::Continue(command),
    }) = cli.command
    else {
        panic!("expected continue command");
    };
    assert_eq!(
        command.duplex_trace_scenario,
        Some(DuplexTraceScenarioOption::OverlapYield)
    );
    assert_eq!(command.jsonl, Some(PathBuf::from("out/duplex-trace.jsonl")));
}

#[test]
fn dev_trace_viewer_export_parses_input_and_output_paths() {
    let cli = Cli::try_parse_from([
        "listenbury",
        "dev",
        "trace-viewer-export",
        "out/live-trace.jsonl",
        "examples/browser-transcript-player/fixtures/live-trace.sample.viewer.json",
    ])
    .expect("trace-viewer-export should parse");

    let Some(Command::Dev {
        command: DevCommand::TraceViewerExport(command),
    }) = cli.command
    else {
        panic!("expected trace-viewer-export command");
    };
    assert_eq!(command.input_jsonl, PathBuf::from("out/live-trace.jsonl"));
    assert_eq!(
        command.output_json,
        PathBuf::from("examples/browser-transcript-player/fixtures/live-trace.sample.viewer.json")
    );
}

#[test]
fn dev_prosody_plan_parses_alignment_praat_and_outputs() {
    let cli = Cli::try_parse_from([
        "listenbury",
        "dev",
        "prosody-plan",
        "out/alignment.json",
        "out/praat.json",
        "out/prosody-plan.json",
        "--ssml-output",
        "out/prosody.ssml",
    ])
    .expect("prosody-plan should parse");

    let Some(Command::Dev {
        command: DevCommand::ProsodyPlan(command),
    }) = cli.command
    else {
        panic!("expected prosody-plan command");
    };
    assert_eq!(command.alignment_json, PathBuf::from("out/alignment.json"));
    assert_eq!(command.praat_json, PathBuf::from("out/praat.json"));
    assert_eq!(command.output_json, PathBuf::from("out/prosody-plan.json"));
    assert_eq!(command.ssml_output, Some(PathBuf::from("out/prosody.ssml")));
}

#[test]
fn dev_sing_demo_parses_backend_and_output() {
    let cli = Cli::try_parse_from([
        "listenbury",
        "dev",
        "sing-demo",
        "--backend",
        "riper",
        "--output-wav",
        "out/hello-ragtime-riper.wav",
        "--piper-voice",
        "voices/demo.onnx",
    ])
    .expect("sing-demo should parse backend and optional output path");

    let Some(Command::Dev {
        command: DevCommand::SingDemo(command),
    }) = cli.command
    else {
        panic!("expected sing-demo command");
    };
    assert_eq!(command.backend, Some(SingDemoBackendOption::Riper));
    assert_eq!(command.selected_backend(), SingDemoBackendOption::Riper);
    assert_eq!(
        command.output_wav,
        Some(PathBuf::from("out/hello-ragtime-riper.wav"))
    );
    assert_eq!(command.piper_voice, Some(PathBuf::from("voices/demo.onnx")));
}

#[test]
fn dev_sing_demo_defaults_to_klatt_backend() {
    let cli = Cli::try_parse_from(["listenbury", "dev", "sing-demo"])
        .expect("sing-demo should parse defaults");

    let Some(Command::Dev {
        command: DevCommand::SingDemo(command),
    }) = cli.command
    else {
        panic!("expected sing-demo command");
    };
    assert_eq!(command.backend, None);
    assert_eq!(command.selected_backend(), SingDemoBackendOption::Riper);
    assert!(command.output_wav.is_none());
}

#[test]
fn dev_sing_demo_accepts_klatt_flag() {
    let cli = Cli::try_parse_from(["listenbury", "dev", "sing-demo", "--klatt"])
        .expect("sing-demo should parse klatt flag");

    let Some(Command::Dev {
        command: DevCommand::SingDemo(command),
    }) = cli.command
    else {
        panic!("expected sing-demo command");
    };
    assert!(command.klatt);
    assert_eq!(command.selected_backend(), SingDemoBackendOption::Klatt);
}

#[test]
fn dev_sing_demo_rejects_riper_flag() {
    let error = Cli::try_parse_from(["listenbury", "dev", "sing-demo", "--riper"])
        .expect_err("--riper should be removed");
    assert!(error.to_string().contains("--riper"));
}

#[test]
fn dev_sing_demo_defaults_to_riper() {
    let cli = Cli::try_parse_from(["listenbury", "dev", "sing-demo"])
        .expect("sing-demo should parse default Riper backend");

    let Some(Command::Dev {
        command: DevCommand::SingDemo(command),
    }) = cli.command
    else {
        panic!("expected sing-demo command");
    };
    assert!(!command.klatt);
    assert_eq!(command.selected_backend(), SingDemoBackendOption::Riper);
}

#[test]
fn top_level_sing_accepts_diphone() {
    let cli = Cli::try_parse_from(["listenbury", "sing", "--diphone"])
        .expect("top-level sing should parse diphone as a Riper backend voice");

    let Some(Command::Sing(command)) = cli.command else {
        panic!("expected top-level sing command");
    };
    assert!(command.diphone);
    assert_eq!(command.selected_backend(), SingDemoBackendOption::Mbrola);
}

#[test]
fn top_level_sing_accepts_hifigan() {
    let cli = Cli::try_parse_from(["listenbury", "sing", "--hifigan"])
        .expect("top-level sing should parse hifigan as SpeechT5 acoustic plus HiFi-GAN");

    let Some(Command::Sing(command)) = cli.command else {
        panic!("expected top-level sing command");
    };
    assert!(command.hifigan);
    assert_eq!(command.selected_backend(), SingDemoBackendOption::Speecht5);
}

#[test]
fn top_level_sing_accepts_speecht5() {
    let cli = Cli::try_parse_from(["listenbury", "sing", "--speecht5"])
        .expect("top-level sing should parse speecht5 as a native acoustic backend");

    let Some(Command::Sing(command)) = cli.command else {
        panic!("expected top-level sing command");
    };
    assert!(command.speecht5);
    assert_eq!(command.selected_backend(), SingDemoBackendOption::Speecht5);
}

#[test]
fn top_level_sing_accepts_hifigan_backend_option() {
    let cli = Cli::try_parse_from(["listenbury", "sing", "--backend", "hifigan"])
        .expect("top-level sing should parse hifigan backend option");

    let Some(Command::Sing(command)) = cli.command else {
        panic!("expected top-level sing command");
    };
    assert_eq!(command.backend, Some(SingDemoBackendOption::Hifigan));
    assert_eq!(command.selected_backend(), SingDemoBackendOption::Hifigan);
}

#[test]
fn top_level_sing_accepts_speecht5_backend_option() {
    let cli = Cli::try_parse_from(["listenbury", "sing", "--backend", "speecht5"])
        .expect("top-level sing should parse speecht5 backend option");

    let Some(Command::Sing(command)) = cli.command else {
        panic!("expected top-level sing command");
    };
    assert_eq!(command.backend, Some(SingDemoBackendOption::Speecht5));
    assert_eq!(command.selected_backend(), SingDemoBackendOption::Speecht5);
}

#[test]
fn top_level_sing_accepts_skip_gan_with_hifigan_backend() {
    let cli = Cli::try_parse_from(["listenbury", "sing", "--backend", "hifigan", "--skip-gan"])
        .expect("top-level sing should parse skip-gan as a mel debug modifier");

    let Some(Command::Sing(command)) = cli.command else {
        panic!("expected top-level sing command");
    };
    assert_eq!(command.selected_backend(), SingDemoBackendOption::Hifigan);
    assert!(command.skip_gan);
}

#[test]
fn top_level_sing_rejects_mbrola() {
    let error = Cli::try_parse_from(["listenbury", "sing", "--mbrola"])
        .expect_err("--mbrola should be renamed");
    assert!(error.to_string().contains("--mbrola"));
}

#[test]
fn sing_routes_to_shared_sing_demo_command() {
    let cli = Cli::try_parse_from(["listenbury", "sing", "--output-wav", "out/sing.wav"])
        .expect("top-level sing should parse sing-demo flags");

    let Some(Command::Sing(command)) = cli.command else {
        panic!("expected top-level sing command");
    };
    assert_eq!(command.selected_backend(), SingDemoBackendOption::Riper);
    assert_eq!(command.output_wav, Some(PathBuf::from("out/sing.wav")));
}

#[test]
fn web_command_parses_defaults() {
    let cli = Cli::try_parse_from(["listenbury", "web"]).expect("web should parse defaults");

    let Some(Command::Web(command)) = cli.command else {
        panic!("expected web command");
    };
    assert_eq!(command.host, "127.0.0.1");
    assert_eq!(command.port, 8787);
    assert!(command.payload.is_none());
    assert!(command.trace.is_none());
    assert!(!command.open);
}

#[test]
fn web_command_parses_all_flags() {
    let cli = Cli::try_parse_from([
        "listenbury",
        "web",
        "--host",
        "0.0.0.0",
        "--port",
        "9090",
        "--payload",
        "out/viewer-payload.json",
        "--trace",
        "out/live.jsonl",
        "--open",
    ])
    .expect("web should parse optional flags");

    let Some(Command::Web(command)) = cli.command else {
        panic!("expected web command");
    };
    assert_eq!(command.host, "0.0.0.0");
    assert_eq!(command.port, 9090);
    assert_eq!(
        command.payload,
        Some(PathBuf::from("out/viewer-payload.json"))
    );
    assert_eq!(command.trace, Some(PathBuf::from("out/live.jsonl")));
    assert!(command.open);
}

#[test]
fn dogfood_two_parses_all_flags() {
    let cli = Cli::try_parse_from([
        "listenbury",
        "dev",
        "dogfood-two",
        "--seed",
        "Hi there.",
        "--turns",
        "4",
        "--max-tokens",
        "64",
        "--whisper-model",
        "models/ggml-tiny.bin",
        "--llm-model",
        "models/tiny.gguf",
        "--piper-bin",
        "/usr/bin/piper",
        "--piper-voice-a",
        "voices/a.onnx",
        "--piper-voice-b",
        "voices/b.onnx",
        "--jsonl",
        "out/dogfood-two.jsonl",
        "--save-audio-dir",
        "out/dogfood-two-audio",
    ])
    .expect("dogfood-two should parse all flags");

    let Some(Command::Dev {
        command: DevCommand::DogfoodTwo(command),
    }) = cli.command
    else {
        panic!("expected dogfood-two command");
    };
    assert_eq!(command.seed, "Hi there.");
    assert_eq!(command.turns, 4);
    assert_eq!(command.max_tokens, 64);
    assert_eq!(
        command.whisper_model,
        Some(PathBuf::from("models/ggml-tiny.bin"))
    );
    assert_eq!(command.llm_model, Some(PathBuf::from("models/tiny.gguf")));
    assert_eq!(command.piper_bin, Some(PathBuf::from("/usr/bin/piper")));
    assert_eq!(command.piper_voice_a, Some(PathBuf::from("voices/a.onnx")));
    assert_eq!(command.piper_voice_b, Some(PathBuf::from("voices/b.onnx")));
    assert_eq!(command.jsonl, Some(PathBuf::from("out/dogfood-two.jsonl")));
    assert_eq!(
        command.save_audio_dir,
        Some(PathBuf::from("out/dogfood-two-audio"))
    );
}

#[test]
fn harmony_go_parses_go_options() {
    let cli = Cli::try_parse_from([
        "listenbury",
        "harmony-go",
        "--llm-model",
        "models/pete.gguf",
        "--max-tokens",
        "64",
        "hello",
        "Pete",
    ])
    .expect("harmony-go should parse go-compatible options");

    let Some(Command::HarmonyGo(command)) = cli.command else {
        panic!("expected harmony-go command");
    };
    assert_eq!(command.llm_model, Some(PathBuf::from("models/pete.gguf")));
    assert_eq!(command.max_tokens, Some(64));
    assert_eq!(command.prompt, vec!["hello", "Pete"]);
}

#[test]
fn draft_pete_line_parses_go_llm_options() {
    let cli = Cli::try_parse_from([
        "listenbury",
        "draft",
        "pete-line",
        "--llm-model",
        "models/pete.gguf",
        "--llm-gpu-layers",
        "12",
        "--max-tokens",
        "64",
        "--context-size",
        "4096",
        "--mock-mouth",
    ])
    .expect("draft pete-line should parse");

    let Some(Command::Draft {
        command: DraftCommand::PeteLine(command),
    }) = cli.command
    else {
        panic!("expected draft pete-line command");
    };
    assert_eq!(command.llm_model, Some(PathBuf::from("models/pete.gguf")));
    assert_eq!(command.llm_gpu_layers, Some(12));
    assert_eq!(command.max_tokens, Some(64));
    assert_eq!(command.context_size, 4096);
    assert!(command.mock_mouth);
}

#[test]
fn top_level_help_keeps_diagnostics_hidden() {
    let mut command = Cli::command();
    let help = command.render_help().to_string();

    for visible in [
        "transcribe",
        "say",
        "listen",
        "ask",
        "reply",
        "web",
        "models",
    ] {
        assert!(
            help.contains(visible),
            "missing {visible} from help:\n{help}"
        );
    }

    for hidden in [
        "fake-turn",
        "demo-vad",
        "vad-trace",
        "breath-transcribe",
        "mic-transcribe",
        "record-wav",
        "play-wav",
        "llama-turn",
        "round-trip-wav",
        "live-half-duplex",
        "dogfood-two",
        "speech-cache",
        "dev",
    ] {
        assert!(!help.contains(hidden), "leaked {hidden} into help:\n{help}");
    }
}

#[test]
fn models_fetch_parses_jobs() {
    let cli = Cli::try_parse_from(["listenbury", "models", "fetch", "--jobs", "4"])
        .expect("models fetch should parse jobs");

    let Some(Command::Models {
        command: Some(ModelsCommand::Fetch(command)),
    }) = cli.command
    else {
        panic!("expected models fetch command");
    };
    assert_eq!(command.jobs, 4);
    assert!(!command.verify);
}

#[test]
fn models_fetch_parses_verify_flag() {
    let cli = Cli::try_parse_from(["listenbury", "models", "fetch", "--verify"])
        .expect("models fetch should parse verify");

    let Some(Command::Models {
        command: Some(ModelsCommand::Fetch(command)),
    }) = cli.command
    else {
        panic!("expected models fetch command");
    };
    assert!(command.verify);
}

#[test]
fn models_status_parses_verify_flag() {
    let cli = Cli::try_parse_from(["listenbury", "models", "status", "--verify"])
        .expect("models status should parse verify");

    let Some(Command::Models {
        command: Some(ModelsCommand::Status(command)),
    }) = cli.command
    else {
        panic!("expected models status command");
    };
    assert!(command.verify);
}

#[test]
fn models_repair_parses_jobs() {
    let cli = Cli::try_parse_from(["listenbury", "models", "repair", "--jobs", "3"])
        .expect("models repair should parse jobs");

    let Some(Command::Models {
        command: Some(ModelsCommand::Repair(command)),
    }) = cli.command
    else {
        panic!("expected models repair command");
    };
    assert_eq!(command.jobs, 3);
    assert!(!command.verify);
}

#[test]
fn models_accepts_bare_menu() {
    let cli = Cli::try_parse_from(["listenbury", "models"])
        .expect("bare models should parse as the interactive menu");

    let Some(Command::Models { command: None }) = cli.command else {
        panic!("expected bare models command");
    };
}

#[test]
fn models_menu_parses_explicitly() {
    let cli =
        Cli::try_parse_from(["listenbury", "models", "menu"]).expect("models menu should parse");

    let Some(Command::Models {
        command: Some(ModelsCommand::Menu),
    }) = cli.command
    else {
        panic!("expected models menu command");
    };
}

#[test]
fn models_verify_parses() {
    let cli = Cli::try_parse_from(["listenbury", "models", "verify"])
        .expect("models verify should parse");

    let Some(Command::Models {
        command: Some(ModelsCommand::Verify(command)),
    }) = cli.command
    else {
        panic!("expected models verify command");
    };
    assert!(command.model.is_none());
}

#[test]
fn models_verify_parses_model_name() {
    let cli = Cli::try_parse_from(["listenbury", "models", "verify", "whisper-tiny"])
        .expect("models verify with model name should parse");

    let Some(Command::Models {
        command: Some(ModelsCommand::Verify(command)),
    }) = cli.command
    else {
        panic!("expected models verify command");
    };
    assert_eq!(command.model.as_deref(), Some("whisper-tiny"));
}

#[test]
fn models_use_parses_text_embedding_kind() {
    let cli = Cli::try_parse_from([
        "listenbury",
        "models",
        "use",
        "text-embedding",
        "embeddinggemma",
    ])
    .expect("models use text-embedding should parse");

    let Some(Command::Models {
        command: Some(ModelsCommand::Use(command)),
    }) = cli.command
    else {
        panic!("expected models use command");
    };
    assert_eq!(command.kind, ModelsUseKind::TextEmbedding);
    assert_eq!(command.model, "embeddinggemma");
}
