use std::f32::consts::TAU;

use anyhow::{Result, bail, ensure};

use crate::audio::frame::AudioFrame;
use crate::time::ExactTimestamp;
use crate::vocoder::{
    BackendCapabilities, BackendFamily, MelConfig, MelFrame, SPEECHT5_HIFIGAN_MEL_CONFIG,
    SpeechSynthesizer, VocoderDescriptor, VocoderInput,
};

const MIN_F0_HZ: f32 = 55.0;
const MAX_F0_HZ: f32 = 1_200.0;
const NOISE_GAIN: f32 = 0.018;
const LOG_MEL_MIN: f32 = -8.0;
const LOG_MEL_MAX: f32 = 2.0;
const MIN_NORMALIZABLE_PEAK: f32 = 1.0e-4;

pub struct MelDebugRendererBackend {
    mel_config: MelConfig,
}

impl MelDebugRendererBackend {
    pub fn new() -> Self {
        Self {
            mel_config: SPEECHT5_HIFIGAN_MEL_CONFIG,
        }
    }

    pub fn descriptor() -> VocoderDescriptor {
        VocoderDescriptor {
            id: "mel-debug-renderer",
            family: BackendFamily::FormantSourceFilter,
            capabilities: BackendCapabilities {
                accepts_phone_timed: false,
                accepts_partial_prosody: false,
                accepts_coarse_text: false,
                accepts_mel: true,
                accepts_mel_f0: true,
                honors_explicit_duration: false,
                honors_explicit_f0: true,
                honors_vibrato: false,
                streaming_safe: false,
            },
            sample_rate_hz: SPEECHT5_HIFIGAN_MEL_CONFIG.sample_rate_hz,
            backend_kind: None,
            detail: None,
            notes: &[
                "Debug-only renderer for acoustic mel/F0 tracks; it is not a neural vocoder and does not run HiFi-GAN.",
                "Uses a deterministic harmonic/noise source shaped by mel energy so mel-path tests can emit inspectable audio.",
            ],
        }
    }

    fn render_mel(
        &self,
        mel: &[MelFrame],
        f0_hz: Option<&[f32]>,
        voiced: Option<&[bool]>,
    ) -> Result<Vec<AudioFrame>> {
        validate_mel_f0_tracks(mel, f0_hz, voiced)?;
        self.mel_config.validate_mel(mel)?;

        let mut phase = 0.0f32;
        let mut noise_state = 0x4d59_4446u32;
        let mut samples = Vec::with_capacity(mel.len() * self.mel_config.hop_length);

        for (frame_index, frame) in mel.iter().enumerate() {
            let next_frame = mel.get(frame_index + 1).unwrap_or(frame);
            let f0_start = f0_for_frame(frame, f0_hz.map(|values| values[frame_index]));
            let f0_end = f0_for_frame(
                next_frame,
                f0_hz.map(|values| {
                    values
                        .get(frame_index + 1)
                        .copied()
                        .unwrap_or(values[frame_index])
                }),
            );
            let voiced_start = voiced.map(|values| values[frame_index]).unwrap_or(true);
            let voiced_end = voiced
                .map(|values| values.get(frame_index + 1).copied().unwrap_or(voiced_start))
                .unwrap_or(voiced_start);
            let amp_start = amplitude_for_frame(frame);
            let amp_end = amplitude_for_frame(next_frame);
            let brightness_start = brightness_for_frame(frame);
            let brightness_end = brightness_for_frame(next_frame);

            for sample_index in 0..self.mel_config.hop_length {
                let t = sample_index as f32 / self.mel_config.hop_length as f32;
                let amp = lerp(amp_start, amp_end, t);
                let brightness = lerp(brightness_start, brightness_end, t);
                let frame_f0 = lerp(f0_start, f0_end, t);
                let is_voiced = if t < 0.5 { voiced_start } else { voiced_end };

                let value = if is_voiced {
                    phase = (phase + TAU * frame_f0 / self.mel_config.sample_rate_hz as f32) % TAU;
                    let harmonic_mix = 0.18 + brightness * 0.32;
                    let source = phase.sin()
                        + harmonic_mix * (phase * 2.0).sin()
                        + (harmonic_mix * 0.45) * (phase * 3.0).sin();
                    source * amp
                } else {
                    (next_noise_sample(&mut noise_state) * 2.0 - 1.0) * amp * NOISE_GAIN
                };
                samples.push(value.clamp(-1.0, 1.0));
            }
        }

        ensure!(!samples.is_empty(), "mel debug renderer produced no audio");
        normalize_loudness(&mut samples, 0.075, 0.92);

        let frames = vec![AudioFrame {
            captured_at: ExactTimestamp::now(),
            sample_rate_hz: self.mel_config.sample_rate_hz,
            channels: 1,
            samples,
            voice_signatures: Vec::new(),
        }];
        tracing::debug!(
            frame_count = frames.len(),
            sample_rate_hz = self.mel_config.sample_rate_hz,
            "mel debug renderer waveform summary"
        );
        Ok(frames)
    }
}

impl Default for MelDebugRendererBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl SpeechSynthesizer for MelDebugRendererBackend {
    fn id(&self) -> &'static str {
        Self::descriptor().id
    }

    fn descriptor(&self) -> VocoderDescriptor {
        Self::descriptor()
    }

    fn render(&mut self, input: VocoderInput<'_>) -> Result<Vec<AudioFrame>> {
        match input {
            VocoderInput::Mel(mel) => self.render_mel(mel, None, None),
            VocoderInput::MelF0 { mel, f0_hz, voiced } => {
                self.render_mel(mel, Some(f0_hz), Some(voiced))
            }
            _ => bail!("mel debug renderer requires Mel or MelF0 input from an acoustic model"),
        }
    }
}

fn validate_mel_f0_tracks(
    mel: &[MelFrame],
    f0_hz: Option<&[f32]>,
    voiced: Option<&[bool]>,
) -> Result<()> {
    ensure!(
        !mel.is_empty(),
        "mel debug renderer received empty mel input"
    );
    if let Some(f0_hz) = f0_hz {
        ensure!(
            f0_hz.len() == mel.len(),
            "mel debug renderer received {} F0 values for {} mel frames",
            f0_hz.len(),
            mel.len()
        );
    }
    if let Some(voiced) = voiced {
        ensure!(
            voiced.len() == mel.len(),
            "mel debug renderer received {} voiced flags for {} mel frames",
            voiced.len(),
            mel.len()
        );
    }
    Ok(())
}

fn amplitude_for_frame(frame: &MelFrame) -> f32 {
    if frame.bins.is_empty() {
        return 0.0;
    }
    let level = frame
        .bins
        .iter()
        .map(|bin| mel_bin_energy(*bin))
        .sum::<f32>()
        / frame.bins.len() as f32;
    level.sqrt().clamp(0.0, 0.35)
}

fn brightness_for_frame(frame: &MelFrame) -> f32 {
    if frame.bins.is_empty() {
        return 0.0;
    }
    let mut weighted = 0.0f32;
    let mut total = 0.0f32;
    let max_index = (frame.bins.len() - 1).max(1) as f32;
    for (index, bin) in frame.bins.iter().enumerate() {
        let energy = mel_bin_energy(*bin);
        weighted += energy * (index as f32 / max_index);
        total += energy;
    }
    if total <= f32::EPSILON {
        0.0
    } else {
        (weighted / total).clamp(0.0, 1.0)
    }
}

fn mel_bin_energy(bin: f32) -> f32 {
    if (LOG_MEL_MIN..=LOG_MEL_MAX).contains(&bin) {
        bin.exp()
    } else {
        bin.max(0.0)
    }
}

fn f0_for_frame(frame: &MelFrame, explicit_f0: Option<f32>) -> f32 {
    explicit_f0
        .filter(|hz| hz.is_finite() && *hz > 0.0)
        .unwrap_or_else(|| 90.0 + brightness_for_frame(frame).powf(1.4) * 410.0)
        .clamp(MIN_F0_HZ, MAX_F0_HZ)
}

fn lerp(start: f32, end: f32, t: f32) -> f32 {
    start + (end - start) * t.clamp(0.0, 1.0)
}

fn next_noise_sample(state: &mut u32) -> f32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    ((*state >> 8) as f32) / ((u32::MAX >> 8) as f32)
}

fn normalize_loudness(samples: &mut [f32], target_rms: f32, ceiling: f32) {
    if samples.is_empty() || !target_rms.is_finite() || !ceiling.is_finite() {
        return;
    }

    let rms =
        (samples.iter().map(|sample| sample * sample).sum::<f32>() / samples.len() as f32).sqrt();
    if rms >= MIN_NORMALIZABLE_PEAK && rms.is_finite() {
        let gain = (target_rms / rms).clamp(0.25, 16.0);
        for sample in samples.iter_mut() {
            *sample *= gain;
        }
    }

    let limit = ceiling.abs().max(MIN_NORMALIZABLE_PEAK);
    let knee = limit * 0.86;
    for sample in samples.iter_mut() {
        *sample = soft_limit(*sample, knee, limit);
    }
}

fn soft_limit(sample: f32, knee: f32, limit: f32) -> f32 {
    let sign = sample.signum();
    let magnitude = sample.abs();
    if magnitude <= knee {
        return sample;
    }

    let headroom = (limit - knee).max(MIN_NORMALIZABLE_PEAK);
    let curved = knee + (1.0 - (-(magnitude - knee) / headroom).exp()) * headroom;
    sign * curved.min(limit)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_mel_frames() -> Vec<MelFrame> {
        (0..6)
            .map(|frame_index| MelFrame {
                bins: (0..SPEECHT5_HIFIGAN_MEL_CONFIG.n_mels)
                    .map(|bin_index| {
                        let envelope =
                            1.0 - (bin_index as f32 / SPEECHT5_HIFIGAN_MEL_CONFIG.n_mels as f32);
                        ((0.12 + frame_index as f32 * 0.01) * envelope.max(0.05)).ln()
                    })
                    .collect(),
            })
            .collect()
    }

    #[test]
    fn renders_acoustic_mel_f0_track_for_debugging() {
        let mel = synthetic_mel_frames();
        let f0_hz = vec![220.0, 225.0, 230.0, 235.0, 240.0, 245.0];
        let voiced = vec![true, true, true, true, true, true];
        let mut backend = MelDebugRendererBackend::new();

        let frames = backend
            .render(VocoderInput::MelF0 {
                mel: &mel,
                f0_hz: &f0_hz,
                voiced: &voiced,
            })
            .expect("debug renderer should render mel/F0");

        assert_eq!(backend.id(), "mel-debug-renderer");
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].sample_rate_hz, 16_000);
        assert_eq!(frames[0].channels, 1);
        assert_eq!(
            frames[0].samples.len(),
            mel.len() * SPEECHT5_HIFIGAN_MEL_CONFIG.hop_length
        );
        assert!(frames[0].samples.iter().any(|sample| sample.abs() > 0.0));
    }

    #[test]
    fn rejects_mismatched_f0_tracks() {
        let mel = synthetic_mel_frames();
        let f0_hz = vec![220.0];
        let voiced = vec![true; mel.len()];
        let mut backend = MelDebugRendererBackend::new();

        let err = backend
            .render(VocoderInput::MelF0 {
                mel: &mel,
                f0_hz: &f0_hz,
                voiced: &voiced,
            })
            .expect_err("debug renderer should validate F0 length");

        assert!(err.to_string().contains("F0 values"));
    }
}
