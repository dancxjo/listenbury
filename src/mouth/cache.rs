use std::collections::{HashMap, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::audio::frame::AudioFrame;
#[cfg(feature = "tts-piper")]
use crate::mouth::piper::PiperConfig;
use crate::mouth::planner::{SpeechPlan, SpeechUnit};
use crate::mouth::tts::TextToSpeech;

pub trait SpeechCache {
    fn get(&mut self, unit: &SpeechUnit) -> Result<Option<Vec<AudioFrame>>>;
    fn put(&mut self, unit: &SpeechUnit, frames: &[AudioFrame]) -> Result<()>;
}

#[derive(Debug, Default)]
pub struct MemorySpeechCache {
    map: HashMap<String, Vec<AudioFrame>>,
}

impl MemorySpeechCache {
    fn key(unit: &SpeechUnit) -> String {
        format!("{unit:?}")
    }
}

impl SpeechCache for MemorySpeechCache {
    fn get(&mut self, unit: &SpeechUnit) -> Result<Option<Vec<AudioFrame>>> {
        Ok(self.map.get(&Self::key(unit)).cloned())
    }

    fn put(&mut self, unit: &SpeechUnit, frames: &[AudioFrame]) -> Result<()> {
        self.map.insert(Self::key(unit), frames.to_vec());
        Ok(())
    }
}

pub struct CachedTextToSpeech<T, C> {
    inner: T,
    cache: C,
    pending_cached_audio: VecDeque<AudioFrame>,
    pending_cache_unit: Option<SpeechUnit>,
    pending_cache_frames: Vec<AudioFrame>,
}

impl<T, C> CachedTextToSpeech<T, C> {
    pub fn new(inner: T, cache: C) -> Self {
        Self {
            inner,
            cache,
            pending_cached_audio: VecDeque::new(),
            pending_cache_unit: None,
            pending_cache_frames: Vec::new(),
        }
    }
}

impl<T, C> TextToSpeech for CachedTextToSpeech<T, C>
where
    T: TextToSpeech,
    C: SpeechCache,
{
    fn enqueue(&mut self, plan: SpeechPlan) -> Result<()> {
        if should_cache_unit(plan.unit()) {
            if let Some(cached) = self.cache.get(plan.unit())? {
                self.pending_cached_audio.extend(cached);
                self.pending_cache_unit = None;
                self.pending_cache_frames.clear();
                return Ok(());
            }
            self.pending_cache_unit = Some(plan.unit().clone());
            self.pending_cache_frames.clear();
        } else {
            self.pending_cache_unit = None;
            self.pending_cache_frames.clear();
        }

        self.inner.enqueue(plan)
    }

    fn poll_audio(&mut self) -> Result<Vec<AudioFrame>> {
        if !self.pending_cached_audio.is_empty() {
            return Ok(self.pending_cached_audio.drain(..).collect());
        }

        let frames = self.inner.poll_audio()?;
        if let Some(unit) = self.pending_cache_unit.clone() {
            if frames.is_empty() {
                if !self.pending_cache_frames.is_empty() {
                    self.cache.put(&unit, &self.pending_cache_frames)?;
                    self.pending_cache_unit = None;
                    self.pending_cache_frames.clear();
                }
            } else {
                self.pending_cache_frames.extend(frames.iter().cloned());
            }
        }

        Ok(frames)
    }

    fn stop(&mut self) -> Result<()> {
        self.pending_cached_audio.clear();
        self.pending_cache_unit = None;
        self.pending_cache_frames.clear();
        self.inner.stop()
    }
}

fn should_cache_unit(unit: &SpeechUnit) -> bool {
    matches!(
        unit,
        SpeechUnit::Backchannel(_) | SpeechUnit::DiscourseMarker(_)
    )
}

pub struct FileSpeechCache {
    root: PathBuf,
    voice_id: String,
    config_identity: Option<String>,
}

impl FileSpeechCache {
    pub fn new(
        root: impl Into<PathBuf>,
        voice_id: impl Into<String>,
        config_identity: Option<String>,
    ) -> Self {
        Self {
            root: root.into(),
            voice_id: voice_id.into(),
            config_identity,
        }
    }

    #[cfg(feature = "tts-piper")]
    pub fn for_piper(listenbury_home: impl AsRef<Path>, config: &PiperConfig) -> Self {
        let voice_id = config
            .model_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("unknown-voice")
            .to_string();
        let config_identity = config
            .config_path
            .as_ref()
            .and_then(|path| path.file_name())
            .and_then(|name| name.to_str())
            .map(ToString::to_string)
            .or_else(|| Some(format!("sr{}", config.sample_rate_hz)));
        let root = listenbury_home.as_ref().join("cache").join("speech");

        Self::new(root, voice_id, config_identity)
    }

    fn unit_path(&self, unit: &SpeechUnit) -> Option<PathBuf> {
        if !should_cache_unit(unit) {
            return None;
        }

        let voice_dir = sanitize_segment(&self.voice_id);
        let config_dir = sanitize_segment(self.config_identity.as_deref().unwrap_or("default"));
        let unit_slug = slugify(unit_text(unit));
        Some(
            self.root
                .join(voice_dir)
                .join(config_dir)
                .join(format!("{unit_slug}.wav")),
        )
    }
}

impl SpeechCache for FileSpeechCache {
    fn get(&mut self, unit: &SpeechUnit) -> Result<Option<Vec<AudioFrame>>> {
        let Some(path) = self.unit_path(unit) else {
            return Ok(None);
        };
        if !path.exists() {
            return Ok(None);
        }
        let frames = read_wav_frames(&path)?;
        Ok(Some(frames))
    }

    fn put(&mut self, unit: &SpeechUnit, frames: &[AudioFrame]) -> Result<()> {
        let Some(path) = self.unit_path(unit) else {
            return Ok(());
        };
        let parent = path
            .parent()
            .context("cache path missing parent directory")?;
        fs::create_dir_all(parent).with_context(|| {
            format!(
                "failed to create speech cache directory {}",
                parent.display()
            )
        })?;
        write_wav_frames(&path, frames)
    }
}

fn sanitize_segment(segment: &str) -> String {
    let mut cleaned = segment
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '-' })
        .collect::<String>();
    while cleaned.contains("--") {
        cleaned = cleaned.replace("--", "-");
    }
    cleaned.trim_matches('-').to_ascii_lowercase()
}

fn slugify(text: &str) -> String {
    let normalized = normalize_text(text);
    let mut slug = normalized
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '-' })
        .collect::<String>();
    while slug.contains("--") {
        slug = slug.replace("--", "-");
    }
    let slug = slug.trim_matches('-').to_string();
    if slug.is_empty() {
        "speech-unit".to_string()
    } else {
        slug
    }
}

fn normalize_text(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase()
}

fn unit_text(unit: &SpeechUnit) -> &str {
    match unit {
        SpeechUnit::Backchannel(text)
        | SpeechUnit::DiscourseMarker(text)
        | SpeechUnit::CompleteClause(text)
        | SpeechUnit::CompleteSentence(text)
        | SpeechUnit::FullTurn(text) => text,
    }
}

fn write_wav_frames(path: &Path, frames: &[AudioFrame]) -> Result<()> {
    let Some(first_frame) = frames.first() else {
        anyhow::bail!("cannot cache empty audio frame buffer");
    };
    let spec = hound::WavSpec {
        channels: first_frame.channels,
        sample_rate: first_frame.sample_rate_hz,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .with_context(|| format!("failed to create cached WAV {}", path.display()))?;
    for frame in frames {
        anyhow::ensure!(
            frame.channels == first_frame.channels,
            "speech cache frame channels changed from {} to {}",
            first_frame.channels,
            frame.channels
        );
        anyhow::ensure!(
            frame.sample_rate_hz == first_frame.sample_rate_hz,
            "speech cache frame sample rate changed from {} to {}",
            first_frame.sample_rate_hz,
            frame.sample_rate_hz
        );
        for sample in &frame.samples {
            writer.write_sample(f32_to_i16(*sample))?;
        }
    }
    writer.finalize()?;
    Ok(())
}

fn read_wav_frames(path: &Path) -> Result<Vec<AudioFrame>> {
    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("failed to open cached WAV {}", path.display()))?;
    let spec = reader.spec();
    let samples = reader
        .samples::<i16>()
        .collect::<std::result::Result<Vec<_>, _>>()
        .with_context(|| format!("failed to read cached WAV samples {}", path.display()))?;
    Ok(vec![AudioFrame {
        captured_at: crate::time::ExactTimestamp::now(),
        sample_rate_hz: spec.sample_rate,
        channels: spec.channels,
        samples: samples
            .into_iter()
            .map(|sample| sample as f32 / i16::MAX as f32)
            .collect(),
    }])
}

fn f32_to_i16(sample: f32) -> i16 {
    let sample = sample.clamp(-1.0, 1.0);
    (sample * i16::MAX as f32) as i16
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    #[derive(Default)]
    struct FakeTts {
        enqueues: usize,
        queued_audio: VecDeque<Vec<AudioFrame>>,
    }

    impl TextToSpeech for FakeTts {
        fn enqueue(&mut self, _plan: SpeechPlan) -> Result<()> {
            self.enqueues += 1;
            self.queued_audio.push_back(vec![AudioFrame {
                captured_at: crate::time::ExactTimestamp::now(),
                sample_rate_hz: 22_050,
                channels: 1,
                samples: vec![0.2, 0.1],
            }]);
            Ok(())
        }

        fn poll_audio(&mut self) -> Result<Vec<AudioFrame>> {
            Ok(self.queued_audio.pop_front().unwrap_or_default())
        }

        fn stop(&mut self) -> Result<()> {
            self.queued_audio.clear();
            Ok(())
        }
    }

    #[test]
    fn cached_backchannel_hit_skips_tts_enqueue() {
        let mut cache = MemorySpeechCache::default();
        let unit = SpeechUnit::Backchannel("Okay.".to_string());
        let warmed_frame = AudioFrame {
            captured_at: crate::time::ExactTimestamp::now(),
            sample_rate_hz: 22_050,
            channels: 1,
            samples: vec![0.3, 0.2],
        };
        cache
            .put(&unit, &[warmed_frame.clone()])
            .expect("cache put");

        let inner = FakeTts::default();
        let mut tts = CachedTextToSpeech::new(inner, cache);
        tts.enqueue(SpeechPlan::from(unit)).expect("enqueue");

        let frames = tts.poll_audio().expect("poll");
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].samples, warmed_frame.samples);
        assert_eq!(tts.inner.enqueues, 0);
    }

    #[test]
    fn backchannel_cache_miss_uses_tts_then_warms_cache() {
        let cache = MemorySpeechCache::default();
        let inner = FakeTts::default();
        let mut tts = CachedTextToSpeech::new(inner, cache);
        let unit = SpeechUnit::Backchannel("Okay.".to_string());

        tts.enqueue(SpeechPlan::from(unit.clone()))
            .expect("enqueue");
        assert_eq!(tts.inner.enqueues, 1);

        let first = tts
            .poll_audio()
            .expect("poll should return synthesized audio");
        assert_eq!(first.len(), 1);
        let second = tts.poll_audio().expect("second poll should finalize cache");
        assert!(second.is_empty());

        let cached = tts.cache.get(&unit).expect("cache get");
        assert!(cached.is_some());
    }

    #[test]
    fn complete_sentence_uses_tts_path() {
        let cache = MemorySpeechCache::default();
        let inner = FakeTts::default();
        let mut tts = CachedTextToSpeech::new(inner, cache);
        let unit = SpeechUnit::CompleteSentence("I think that works.".to_string());

        tts.enqueue(SpeechPlan::from(unit)).expect("enqueue");
        assert_eq!(tts.inner.enqueues, 1);
    }

    #[test]
    fn file_cache_round_trip_for_backchannel() {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("listenbury-cache-test-{ts}"));
        let mut cache = FileSpeechCache::new(&root, "voice.onnx", Some("cfg".to_string()));
        let unit = SpeechUnit::Backchannel("Okay.".to_string());
        let frame = AudioFrame {
            captured_at: crate::time::ExactTimestamp::now(),
            sample_rate_hz: 22_050,
            channels: 1,
            samples: vec![0.25, -0.25],
        };
        cache
            .put(&unit, &[frame])
            .expect("cache put should succeed");
        let got = cache.get(&unit).expect("cache get should succeed");
        assert!(got.is_some());

        std::fs::remove_dir_all(root).expect("temporary cache dir should be removed");
    }
}
