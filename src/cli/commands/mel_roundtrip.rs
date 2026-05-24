use crate::cli::MelRoundtripCommand;
use anyhow::{Context, Result};
use listenbury::audio::{read_wav_as_whisper_frames, write_wav};
use listenbury::vocoder::{
    HifiganBackend, SpeechSynthesizer, VocoderInput, extract_speecht5_log_mel,
};

pub(crate) fn run_mel_roundtrip(command: MelRoundtripCommand) -> Result<()> {
    run_mel_roundtrip_impl(command)
}

#[cfg(feature = "piper-compat")]
fn run_mel_roundtrip_impl(command: MelRoundtripCommand) -> Result<()> {
    let frames = read_wav_as_whisper_frames(&command.input_wav, 1_600).with_context(|| {
        format!(
            "failed to read and normalize reference WAV {}",
            command.input_wav.display()
        )
    })?;
    let samples = frames
        .iter()
        .flat_map(|frame| frame.samples.iter().copied())
        .collect::<Vec<_>>();
    let extraction = extract_speecht5_log_mel(&samples)?;
    HifiganBackend::validate_acoustic_contract(extraction.sample_rate_hz, extraction.hop_samples)?;

    if let Some(path) = command.mel_dump.as_ref() {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent).with_context(|| {
                format!("failed to create mel dump directory {}", parent.display())
            })?;
        }
        std::fs::write(path, format_mel_dump(&command, &extraction.mel))
            .with_context(|| format!("failed to write SpeechT5 mel dump {}", path.display()))?;
    }

    let mut hifigan = HifiganBackend::load(&command.hifigan_model)?;
    let reconstructed = hifigan
        .render(VocoderInput::Mel(&extraction.mel))
        .context("failed to reconstruct WAV from SpeechT5 mel with HiFi-GAN")?;

    if let Some(parent) = command
        .output_wav
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create output directory {}", parent.display()))?;
    }
    write_wav(&command.output_wav, &reconstructed)
        .with_context(|| format!("failed to write {}", command.output_wav.display()))?;

    println!(
        "mel_roundtrip input={} frames={} bins={} output={}",
        command.input_wav.display(),
        extraction.mel.len(),
        extraction
            .mel
            .first()
            .map(|frame| frame.bins.len())
            .unwrap_or(0),
        command.output_wav.display()
    );
    Ok(())
}

#[cfg(not(feature = "piper-compat"))]
fn run_mel_roundtrip_impl(_command: MelRoundtripCommand) -> Result<()> {
    anyhow::bail!("mel roundtrip requires the `piper-compat` feature for HiFi-GAN ONNX")
}

#[cfg(feature = "piper-compat")]
fn format_mel_dump(command: &MelRoundtripCommand, mel: &[listenbury::MelFrame]) -> String {
    let mut dump = String::new();
    dump.push_str(&format!("input_wav={}\n", command.input_wav.display()));
    dump.push_str("contract=speecht5-feature-extractor-log10-slaney\n");
    dump.push_str(&format!(
        "sample_rate_hz=16000 hop_samples=256 win_length=1024 n_fft=1024 frame_count={} mel_bins={}\n",
        mel.len(),
        mel.first().map(|frame| frame.bins.len()).unwrap_or(0)
    ));
    for frame in mel {
        let row = frame
            .bins
            .iter()
            .map(|bin| format!("{bin:.6}"))
            .collect::<Vec<_>>()
            .join(" ");
        dump.push_str(&row);
        dump.push('\n');
    }
    dump
}
