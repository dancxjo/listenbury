use std::collections::VecDeque;

use crate::audio::frame::AudioFrame;
use crate::hearing::sound_classify::{SoundFeatures, describe_sound};

const MAX_CAPTURED_ACTIVE_SOUND_MS: u64 = 5_000;

#[derive(Debug, Clone)]
pub struct EnvironmentalSound {
    pub label: Option<String>,
    pub description: String,
    pub confidence: f32,
    pub salience: f32,
}

#[derive(Debug, Clone)]
pub struct EnvironmentalSoundClip {
    pub sound: EnvironmentalSound,
    pub frames: Vec<AudioFrame>,
}

#[derive(Debug, Clone, Copy)]
pub struct EnvironmentalSoundConfig {
    pub min_observation_interval_ms: u64,
    pub silence_observation_interval_ms: u64,
    pub silence_min_duration_ms: u64,
    pub min_observation_salience: f32,
    pub min_duration_ms: u64,
    pub baseline_silence_rms: f32,
}

impl Default for EnvironmentalSoundConfig {
    fn default() -> Self {
        Self {
            min_observation_interval_ms: 900,
            silence_observation_interval_ms: 4_000,
            silence_min_duration_ms: 1_500,
            min_observation_salience: 0.15,
            min_duration_ms: 10,
            baseline_silence_rms: 0.003,
        }
    }
}

#[derive(Debug, Clone)]
struct ActiveSound {
    started_at_ms: u64,
    duration_ms: u64,
    frame_count: u64,
    sum_centroid_hz: f32,
    sum_density: f32,
    peak_rms: f32,
    frames: Vec<AudioFrame>,
}

impl ActiveSound {
    fn new(
        frame: &AudioFrame,
        started_at_ms: u64,
        rms: f32,
        centroid_hz: f32,
        density: f32,
        duration_ms: u64,
    ) -> Self {
        Self {
            started_at_ms,
            duration_ms,
            frame_count: 1,
            sum_centroid_hz: centroid_hz,
            sum_density: density,
            peak_rms: rms,
            frames: vec![frame.clone()],
        }
    }

    fn update(
        &mut self,
        frame: &AudioFrame,
        rms: f32,
        centroid_hz: f32,
        density: f32,
        duration_ms: u64,
    ) {
        self.duration_ms = self.duration_ms.saturating_add(duration_ms);
        self.frame_count = self.frame_count.saturating_add(1);
        self.sum_centroid_hz += centroid_hz;
        self.sum_density += density;
        self.peak_rms = self.peak_rms.max(rms);
        if self.duration_ms <= MAX_CAPTURED_ACTIVE_SOUND_MS {
            self.frames.push(frame.clone());
        }
    }

    fn avg_centroid_hz(&self) -> f32 {
        (self.sum_centroid_hz / self.frame_count.max(1) as f32).max(0.0)
    }

    fn avg_density(&self) -> f32 {
        (self.sum_density / self.frame_count.max(1) as f32).max(0.0)
    }
}

#[derive(Debug, Clone)]
pub struct EnvironmentalSoundObserver {
    config: EnvironmentalSoundConfig,
    frame_time_ms: u64,
    noise_floor_rms: f32,
    silence_started_at_ms: Option<u64>,
    last_observation_ms: Option<u64>,
    last_silence_observation_ms: Option<u64>,
    active_sound: Option<ActiveSound>,
    transient_timestamps_ms: VecDeque<u64>,
}

impl Default for EnvironmentalSoundObserver {
    fn default() -> Self {
        Self::new(EnvironmentalSoundConfig::default())
    }
}

impl EnvironmentalSoundObserver {
    pub fn new(config: EnvironmentalSoundConfig) -> Self {
        Self {
            config,
            frame_time_ms: 0,
            noise_floor_rms: config.baseline_silence_rms,
            silence_started_at_ms: Some(0),
            last_observation_ms: None,
            last_silence_observation_ms: None,
            active_sound: None,
            transient_timestamps_ms: VecDeque::with_capacity(16),
        }
    }

    pub fn noise_floor_rms(&self) -> f32 {
        self.noise_floor_rms
    }

    pub fn observe_frame(
        &mut self,
        frame: &AudioFrame,
        speech_active: bool,
    ) -> Option<EnvironmentalSound> {
        self.observe_frame_with_audio(frame, speech_active)
            .map(|clip| clip.sound)
    }

    pub fn observe_frame_with_audio(
        &mut self,
        frame: &AudioFrame,
        speech_active: bool,
    ) -> Option<EnvironmentalSoundClip> {
        let duration_ms = frame_duration_ms(frame);
        let frame_started_at_ms = self.frame_time_ms;
        self.frame_time_ms = self.frame_time_ms.saturating_add(duration_ms);
        if duration_ms == 0 {
            return None;
        }

        let rms = frame_rms(frame);
        let centroid_hz = spectral_centroid_hz(frame);
        let temporal_density = temporal_density(frame);

        if speech_active {
            self.finalize_active_sound(frame_started_at_ms, true)
        } else {
            let silence_threshold = self
                .config
                .baseline_silence_rms
                .max(self.noise_floor_rms * 1.35);
            let is_silence = rms <= silence_threshold;
            if is_silence {
                self.update_noise_floor(rms, true);
                let observation = self.finalize_active_sound(frame_started_at_ms, false);
                if let Some(existing) = observation {
                    return Some(existing);
                }
                self.maybe_emit_silence_observation()
            } else {
                self.silence_started_at_ms = None;
                self.update_noise_floor(rms, false);
                self.upsert_active_sound(
                    frame,
                    frame_started_at_ms,
                    duration_ms,
                    rms,
                    centroid_hz,
                    temporal_density,
                );
                None
            }
        }
    }

    fn update_noise_floor(&mut self, rms: f32, is_silence: bool) {
        let alpha = if is_silence { 0.25 } else { 0.01 };
        self.noise_floor_rms = (self.noise_floor_rms * (1.0 - alpha) + rms * alpha)
            .max(self.config.baseline_silence_rms);
    }

    fn upsert_active_sound(
        &mut self,
        frame: &AudioFrame,
        started_at_ms: u64,
        duration_ms: u64,
        rms: f32,
        centroid_hz: f32,
        temporal_density: f32,
    ) {
        match &mut self.active_sound {
            Some(active) => active.update(frame, rms, centroid_hz, temporal_density, duration_ms),
            None => {
                self.active_sound = Some(ActiveSound::new(
                    frame,
                    started_at_ms,
                    rms,
                    centroid_hz,
                    temporal_density,
                    duration_ms,
                ));
            }
        }
    }

    fn finalize_active_sound(
        &mut self,
        now_ms: u64,
        suppress_emit_for_speech: bool,
    ) -> Option<EnvironmentalSoundClip> {
        let active = self.active_sound.take()?;
        if suppress_emit_for_speech || active.duration_ms < self.config.min_duration_ms {
            return None;
        }
        if !rate_limit_elapsed(
            self.last_observation_ms,
            now_ms,
            self.config.min_observation_interval_ms,
        ) {
            return None;
        }

        let salience = ((active.peak_rms - self.noise_floor_rms)
            / self.noise_floor_rms.max(self.config.baseline_silence_rms))
        .clamp(0.0, 1.0);
        if salience < self.config.min_observation_salience {
            return None;
        }

        let repetition_hz = self.repetition_hz();
        let features = SoundFeatures {
            duration_ms: active.duration_ms,
            centroid_hz: active.avg_centroid_hz(),
            repetition_hz,
            salience,
            temporal_density: active.avg_density(),
        };
        let classification = describe_sound(features);
        if matches!(classification.label.as_deref(), Some("impact" | "chirp")) {
            self.transient_timestamps_ms.push_back(active.started_at_ms);
            while self.transient_timestamps_ms.len() > 8 {
                let _ = self.transient_timestamps_ms.pop_front();
            }
        }
        self.last_observation_ms = Some(now_ms);

        Some(EnvironmentalSoundClip {
            sound: EnvironmentalSound {
                label: classification.label,
                description: classification.description,
                confidence: classification.confidence,
                salience,
            },
            frames: active.frames,
        })
    }

    fn repetition_hz(&mut self) -> f32 {
        while let Some(first) = self.transient_timestamps_ms.front().copied() {
            if self.frame_time_ms.saturating_sub(first) > 2_000 {
                let _ = self.transient_timestamps_ms.pop_front();
            } else {
                break;
            }
        }
        if self.transient_timestamps_ms.len() < 3 {
            return 0.0;
        }
        let times = self
            .transient_timestamps_ms
            .iter()
            .copied()
            .collect::<Vec<_>>();
        let mut intervals = Vec::with_capacity(times.len().saturating_sub(1));
        for pair in times.windows(2) {
            intervals.push(pair[1].saturating_sub(pair[0]) as f32);
        }
        if intervals.is_empty() {
            return 0.0;
        }
        let mean = intervals.iter().sum::<f32>() / intervals.len() as f32;
        if !(55.0..=340.0).contains(&mean) {
            return 0.0;
        }
        1_000.0 / mean.max(1.0)
    }

    fn maybe_emit_silence_observation(&mut self) -> Option<EnvironmentalSoundClip> {
        let started = *self.silence_started_at_ms.get_or_insert(self.frame_time_ms);
        let silence_duration_ms = self.frame_time_ms.saturating_sub(started);
        if silence_duration_ms < self.config.silence_min_duration_ms {
            return None;
        }
        if !rate_limit_elapsed(
            self.last_silence_observation_ms,
            self.frame_time_ms,
            self.config.silence_observation_interval_ms,
        ) {
            return None;
        }
        self.last_silence_observation_ms = Some(self.frame_time_ms);
        Some(EnvironmentalSoundClip {
            sound: EnvironmentalSound {
                label: Some("silence".to_string()),
                description: format!(
                    "I heard ongoing silence for {}.",
                    format_seconds_duration(silence_duration_ms)
                ),
                confidence: 0.95,
                salience: 0.0,
            },
            frames: Vec::new(),
        })
    }
}

fn frame_duration_ms(frame: &AudioFrame) -> u64 {
    if frame.sample_rate_hz == 0 || frame.channels == 0 {
        return 0;
    }
    let samples_per_channel = frame.samples.len() as f64 / f64::from(frame.channels);
    ((samples_per_channel / f64::from(frame.sample_rate_hz)) * 1000.0).round() as u64
}

fn frame_rms(frame: &AudioFrame) -> f32 {
    if frame.samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = frame.samples.iter().map(|sample| sample * sample).sum();
    (sum_sq / frame.samples.len() as f32).sqrt()
}

fn spectral_centroid_hz(frame: &AudioFrame) -> f32 {
    if frame.sample_rate_hz == 0 || frame.channels == 0 || frame.samples.len() < 2 {
        return 0.0;
    }
    let channel_count = usize::from(frame.channels).max(1);
    let mono = frame
        .samples
        .chunks(channel_count)
        .map(|chunk| chunk.iter().copied().sum::<f32>() / channel_count as f32)
        .collect::<Vec<_>>();
    if mono.len() < 2 {
        return 0.0;
    }
    let zero_crossings = mono
        .windows(2)
        .filter(|pair| pair[0].signum() != pair[1].signum())
        .count() as f32;
    let duration_s = mono.len() as f32 / frame.sample_rate_hz as f32;
    if duration_s <= f32::EPSILON {
        0.0
    } else {
        (zero_crossings / (2.0 * duration_s)).max(0.0)
    }
}

fn temporal_density(frame: &AudioFrame) -> f32 {
    if frame.channels == 0 || frame.samples.len() < 2 {
        return 0.0;
    }
    let channel_count = usize::from(frame.channels).max(1);
    let mono = frame
        .samples
        .chunks(channel_count)
        .map(|chunk| chunk.iter().copied().sum::<f32>() / channel_count as f32)
        .collect::<Vec<_>>();
    if mono.len() < 2 {
        return 0.0;
    }
    let transitions = mono
        .windows(2)
        .filter(|pair| pair[0].signum() != pair[1].signum())
        .count();
    transitions as f32 / mono.len() as f32
}

fn format_seconds_duration(duration_ms: u64) -> String {
    if duration_ms < 1_000 {
        format!("{duration_ms} ms")
    } else if duration_ms < 10_000 {
        format!("{:.1} s", duration_ms as f64 / 1_000.0)
    } else {
        format!("{} s", duration_ms / 1_000)
    }
}

fn rate_limit_elapsed(last_at_ms: Option<u64>, now_ms: u64, interval_ms: u64) -> bool {
    last_at_ms
        .map(|last| now_ms.saturating_sub(last) >= interval_ms)
        .unwrap_or(true)
}

#[cfg(test)]
mod tests {
    use super::{EnvironmentalSoundObserver, frame_duration_ms};
    use crate::AudioFrame;
    use crate::time::ExactTimestamp;

    fn frame_from_samples(sample_rate_hz: u32, samples: Vec<f32>) -> AudioFrame {
        AudioFrame {
            captured_at: ExactTimestamp::now(),
            sample_rate_hz,
            channels: 1,
            samples,
            voice_signatures: Vec::new(),
        }
    }

    fn silence_frame() -> AudioFrame {
        frame_from_samples(16_000, vec![0.0; 160])
    }

    fn low_frequency_hum_frame() -> AudioFrame {
        let sample_rate = 16_000.0f32;
        let freq_hz = 120.0f32;
        let samples = (0..160)
            .map(|index| {
                let t = index as f32 / sample_rate;
                (2.0 * std::f32::consts::PI * freq_hz * t).sin() * 0.03
            })
            .collect();
        frame_from_samples(16_000, samples)
    }

    fn impact_frame() -> AudioFrame {
        let mut samples = vec![0.0; 160];
        samples[0] = 0.9;
        frame_from_samples(16_000, samples)
    }

    fn chirp_frame() -> AudioFrame {
        let sample_rate = 16_000.0f32;
        let freq_hz = 3_200.0f32;
        let samples = (0..160)
            .map(|index| {
                let t = index as f32 / sample_rate;
                (2.0 * std::f32::consts::PI * freq_hz * t).sin() * 0.20
            })
            .collect();
        frame_from_samples(16_000, samples)
    }

    #[test]
    fn tracks_and_reports_silence_state() {
        let mut observer = EnvironmentalSoundObserver::default();
        let mut output = Vec::new();
        for _ in 0..220 {
            if let Some(event) = observer.observe_frame(&silence_frame(), false) {
                output.push(event);
            }
        }

        assert!(observer.noise_floor_rms() >= 0.0);
        assert!(
            output
                .iter()
                .any(|event| event.description.contains("ongoing silence"))
        );
    }

    #[test]
    fn reports_sustained_low_frequency_noise() {
        let mut observer = EnvironmentalSoundObserver::default();
        let mut output = Vec::new();
        for _ in 0..120 {
            if let Some(event) = observer.observe_frame(&silence_frame(), false) {
                output.push(event);
            }
        }
        for _ in 0..120 {
            if let Some(event) = observer.observe_frame(&low_frequency_hum_frame(), false) {
                output.push(event);
            }
        }
        for _ in 0..20 {
            if let Some(event) = observer.observe_frame(&silence_frame(), false) {
                output.push(event);
            }
        }

        assert!(
            output
                .iter()
                .any(|event| event.description.contains("sustained fan-like noise"))
        );
    }

    #[test]
    fn reports_transient_impact_sound() {
        let mut observer = EnvironmentalSoundObserver::default();
        let mut output = Vec::new();
        for _ in 0..100 {
            let _ = observer.observe_frame(&silence_frame(), false);
        }
        let _ = observer.observe_frame(&impact_frame(), false);
        for _ in 0..25 {
            if let Some(event) = observer.observe_frame(&silence_frame(), false) {
                output.push(event);
            }
        }

        assert!(
            output
                .iter()
                .any(|event| event.description.contains("sharp impact sound"))
        );
    }

    #[test]
    fn sound_clip_includes_finalized_environmental_audio() {
        let mut observer = EnvironmentalSoundObserver::default();
        for _ in 0..100 {
            let _ = observer.observe_frame_with_audio(&silence_frame(), false);
        }
        let impact = impact_frame();
        let _ = observer.observe_frame_with_audio(&impact, false);

        let mut output = Vec::new();
        for _ in 0..25 {
            if let Some(event) = observer.observe_frame_with_audio(&silence_frame(), false) {
                output.push(event);
            }
        }

        let clip = output
            .iter()
            .find(|event| event.sound.description.contains("sharp impact sound"))
            .expect("impact sound should emit an audio clip");
        assert_eq!(clip.frames.len(), 1);
        assert_eq!(clip.frames[0].samples, impact.samples);
    }

    #[test]
    fn reports_short_chirp_like_signal() {
        let mut observer = EnvironmentalSoundObserver::default();
        let mut output = Vec::new();
        for _ in 0..100 {
            let _ = observer.observe_frame(&silence_frame(), false);
        }
        for _ in 0..5 {
            let _ = observer.observe_frame(&chirp_frame(), false);
        }
        for _ in 0..20 {
            if let Some(event) = observer.observe_frame(&silence_frame(), false) {
                output.push(event);
            }
        }

        assert!(
            output
                .iter()
                .any(|event| event.description.contains("short high-pitched chirp"))
        );
    }

    #[test]
    fn frame_duration_for_default_test_frames_is_ten_ms() {
        assert_eq!(frame_duration_ms(&silence_frame()), 10);
    }
}
