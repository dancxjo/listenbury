use crate::cli::VadTraceCommand;
use anyhow::{Context, Result};
use listenbury::audio::read_wav_as_audio_frames;
use listenbury::AudioFrame;
use listenbury::event::HearingEvent;
use listenbury::hearing::breath::BreathGroupSegmenter;
use listenbury::hearing::vad::{EnergyVad, VoiceActivityDetector};
use serde::Serialize;
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

const VAD_TRACE_FRAME_SAMPLES: usize = 160;

#[derive(Debug, Clone, PartialEq)]
enum VadTraceEvent {
    VadFrame {
        t_ms: u64,
        speech: bool,
        rms: f32,
    },
    BreathGroupStart {
        t_ms: u64,
    },
    BreathGroupEnd {
        start_ms: u64,
        t_ms: u64,
        duration_ms: u64,
    },
}

pub(crate) fn run_vad_trace(command: VadTraceCommand) -> Result<()> {
    let frames = read_wav_as_audio_frames(&command.input_wav, VAD_TRACE_FRAME_SAMPLES)
        .with_context(|| format!("failed to read WAV {}", command.input_wav.display()))?;
    let events = collect_vad_trace_events(&frames)?;

    for event in &events {
        match event {
            VadTraceEvent::VadFrame { t_ms, speech, rms } => {
                println!("{t_ms:04}ms speech={speech:<5} rms={rms:.3}");
            }
            VadTraceEvent::BreathGroupStart { t_ms } => {
                println!("breath-group start={t_ms}ms");
            }
            VadTraceEvent::BreathGroupEnd {
                start_ms,
                t_ms,
                duration_ms,
            } => {
                println!("breath-group start={start_ms}ms end={t_ms}ms duration={duration_ms}ms");
            }
        }
    }

    if let Some(path) = command.jsonl {
        write_events_jsonl(&path, &events)?;
    }

    Ok(())
}

fn collect_vad_trace_events(frames: &[AudioFrame]) -> Result<Vec<VadTraceEvent>> {
    let mut vad = EnergyVad::default();
    let mut segmenter = BreathGroupSegmenter::default();
    let mut events = Vec::new();
    let mut t_ms: u64 = 0;
    let mut group_start_ms = HashMap::new();

    for frame in frames {
        let rms = rms(&frame.samples);
        let vad_result = vad.process_frame(frame)?;
        events.push(VadTraceEvent::VadFrame {
            t_ms,
            speech: vad_result.is_speech,
            rms,
        });

        for hearing_event in segmenter.process(vad_result) {
            match hearing_event {
                HearingEvent::BreathGroupOpened { id } => {
                    group_start_ms.insert(id, t_ms);
                    events.push(VadTraceEvent::BreathGroupStart { t_ms });
                }
                HearingEvent::BreathGroupClosed { id, .. } => {
                    let start_ms = group_start_ms.remove(&id).unwrap_or(t_ms);
                    events.push(VadTraceEvent::BreathGroupEnd {
                        start_ms,
                        t_ms,
                        duration_ms: t_ms.saturating_sub(start_ms),
                    });
                }
                HearingEvent::SpeechStarted
                | HearingEvent::SpeechContinued { .. }
                | HearingEvent::PauseStarted => {}
            }
        }

        t_ms = t_ms.saturating_add(frame_duration_ms(frame));
    }

    Ok(events)
}

fn write_events_jsonl(path: &Path, events: &[VadTraceEvent]) -> Result<()> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty());
    if let Some(parent) = parent {
        std::fs::create_dir_all(parent).with_context(|| {
            format!(
                "failed to create JSONL output directory {}",
                parent.to_string_lossy()
            )
        })?;
    }

    let file = std::fs::File::create(path)
        .with_context(|| format!("failed to create {}", path.display()))?;
    let mut writer = std::io::BufWriter::new(file);

    for event in events {
        match event {
            VadTraceEvent::VadFrame { t_ms, speech, rms } => {
                serde_json::to_writer(
                    &mut writer,
                    &VadFrameJson {
                        kind: "vad_frame",
                        t_ms: *t_ms,
                        speech: *speech,
                        rms: *rms,
                    },
                )?;
            }
            VadTraceEvent::BreathGroupStart { t_ms } => {
                serde_json::to_writer(
                    &mut writer,
                    &BreathGroupStartJson {
                        kind: "breath_group_start",
                        t_ms: *t_ms,
                    },
                )?;
            }
            VadTraceEvent::BreathGroupEnd {
                t_ms, duration_ms, ..
            } => {
                serde_json::to_writer(
                    &mut writer,
                    &BreathGroupEndJson {
                        kind: "breath_group_end",
                        t_ms: *t_ms,
                        duration_ms: *duration_ms,
                    },
                )?;
            }
        }
        writer.write_all(b"\n")?;
    }
    writer.flush()?;

    Ok(())
}

fn frame_duration_ms(frame: &AudioFrame) -> u64 {
    if frame.sample_rate_hz == 0 || frame.channels == 0 {
        return 0;
    }
    let samples_per_channel = frame.samples.len() as f64 / f64::from(frame.channels);
    ((samples_per_channel / f64::from(frame.sample_rate_hz)) * 1000.0).round() as u64
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|sample| sample * sample).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

#[derive(Serialize)]
struct VadFrameJson {
    kind: &'static str,
    t_ms: u64,
    speech: bool,
    rms: f32,
}

#[derive(Serialize)]
struct BreathGroupStartJson {
    kind: &'static str,
    t_ms: u64,
}

#[derive(Serialize)]
struct BreathGroupEndJson {
    kind: &'static str,
    t_ms: u64,
    duration_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use std::path::PathBuf;

    #[test]
    fn silence_fixture_has_no_speech_groups() {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let frames =
            read_wav_as_audio_frames(&repo_root.join("samples/silence-16k-mono.wav"), 160).unwrap();
        let events = collect_vad_trace_events(&frames).unwrap();

        assert!(
            !events
                .iter()
                .any(|event| matches!(event, VadTraceEvent::BreathGroupStart { .. }))
        );
    }

    #[test]
    fn hello_fixture_contains_speech() {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let frames =
            read_wav_as_audio_frames(&repo_root.join("samples/hello-16k-mono.wav"), 160).unwrap();
        let events = collect_vad_trace_events(&frames).unwrap();

        assert!(
            events
                .iter()
                .any(|event| { matches!(event, VadTraceEvent::VadFrame { speech: true, .. }) })
        );
        assert!(
            events
                .iter()
                .any(|event| matches!(event, VadTraceEvent::BreathGroupStart { .. }))
        );
    }

    #[test]
    fn writes_jsonl_events() {
        let path =
            std::env::temp_dir().join(format!("listenbury-vad-trace-{}.jsonl", std::process::id()));
        let events = vec![
            VadTraceEvent::VadFrame {
                t_ms: 20,
                speech: true,
                rms: 0.081,
            },
            VadTraceEvent::BreathGroupStart { t_ms: 430 },
            VadTraceEvent::BreathGroupEnd {
                start_ms: 430,
                t_ms: 1820,
                duration_ms: 1390,
            },
        ];

        write_events_jsonl(&path, &events).unwrap();
        let content = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<Value> = content
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect();

        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0]["kind"], "vad_frame");
        assert_eq!(lines[1]["kind"], "breath_group_start");
        assert_eq!(lines[2]["kind"], "breath_group_end");

        std::fs::remove_file(path).unwrap();
    }
}
