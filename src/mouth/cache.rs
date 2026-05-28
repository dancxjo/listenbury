use std::collections::{HashMap, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::audio::frame::AudioFrame;
use crate::audio::write_wav;
use crate::mouth::piper::PiperConfig;
use crate::mouth::planner::{MouthSyntheticPlan, SyntheticUnit};
use crate::mouth::tts::TextToSpeech;

pub trait SyntheticCache {
    fn get(&mut self, unit: &SyntheticUnit) -> Result<Option<Vec<AudioFrame>>>;
    fn put(&mut self, unit: &SyntheticUnit, frames: &[AudioFrame]) -> Result<()>;
}

#[derive(Debug, Default)]
pub struct MemorySyntheticCache {
    map: HashMap<String, Vec<AudioFrame>>,
}

impl MemorySyntheticCache {
    fn key(unit: &SyntheticUnit) -> String {
        format!("{}:{}", unit_kind(unit), normalize_text(unit_text(unit)))
    }
}

impl SyntheticCache for MemorySyntheticCache {
    fn get(&mut self, unit: &SyntheticUnit) -> Result<Option<Vec<AudioFrame>>> {
        Ok(self.map.get(&Self::key(unit)).cloned())
    }

    fn put(&mut self, unit: &SyntheticUnit, frames: &[AudioFrame]) -> Result<()> {
        self.map.insert(Self::key(unit), frames.to_vec());
        Ok(())
    }
}

pub struct CachedTextToSpeech<T, C> {
    inner: T,
    cache: C,
    pending_cached_audio: VecDeque<AudioFrame>,
    pending_cache_unit: Option<SyntheticUnit>,
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
    C: SyntheticCache,
{
    fn enqueue(&mut self, plan: MouthSyntheticPlan) -> Result<()> {
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

fn should_cache_unit(unit: &SyntheticUnit) -> bool {
    matches!(
        unit,
        SyntheticUnit::Backchannel(_) | SyntheticUnit::DiscourseMarker(_)
    )
}

pub struct FileSyntheticCache {
    root: PathBuf,
    voice_id: String,
    config_identity: Option<String>,
}

impl FileSyntheticCache {
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

    fn unit_path(&self, unit: &SyntheticUnit) -> Option<PathBuf> {
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

impl SyntheticCache for FileSyntheticCache {
    fn get(&mut self, unit: &SyntheticUnit) -> Result<Option<Vec<AudioFrame>>> {
        let Some(path) = self.unit_path(unit) else {
            return Ok(None);
        };
        if !path.exists() {
            return Ok(None);
        }
        let frames = read_wav_frames(&path)?;
        Ok(Some(frames))
    }

    fn put(&mut self, unit: &SyntheticUnit, frames: &[AudioFrame]) -> Result<()> {
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
    collapse_repeated_dashes(
        segment
            .chars()
            .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '-' })
            .collect::<String>()
            .trim_matches('-'),
    )
    .to_ascii_lowercase()
}

fn slugify(text: &str) -> String {
    let normalized = normalize_text(text);
    let slug = collapse_repeated_dashes(
        normalized
            .chars()
            .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '-' })
            .collect::<String>()
            .trim_matches('-'),
    );
    if slug.is_empty() {
        "synthetic-unit".to_string()
    } else {
        slug
    }
}

fn collapse_repeated_dashes(input: &str) -> String {
    let mut output = String::with_capacity(input.len());
    let mut previous_dash = false;
    for ch in input.chars() {
        if ch == '-' {
            if !previous_dash {
                output.push(ch);
            }
            previous_dash = true;
        } else {
            output.push(ch);
            previous_dash = false;
        }
    }
    output
}

fn normalize_text(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase()
}

fn unit_text(unit: &SyntheticUnit) -> &str {
    match unit {
        SyntheticUnit::Backchannel(text)
        | SyntheticUnit::DiscourseMarker(text)
        | SyntheticUnit::CompleteClause(text)
        | SyntheticUnit::CompleteSentence(text)
        | SyntheticUnit::FullTurn(text) => text,
    }
}

fn unit_kind(unit: &SyntheticUnit) -> &'static str {
    match unit {
        SyntheticUnit::Backchannel(_) => "backchannel",
        SyntheticUnit::DiscourseMarker(_) => "discourse-marker",
        SyntheticUnit::CompleteClause(_) => "complete-clause",
        SyntheticUnit::CompleteSentence(_) => "complete-sentence",
        SyntheticUnit::FullTurn(_) => "full-turn",
    }
}

fn write_wav_frames(path: &Path, frames: &[AudioFrame]) -> Result<()> {
    anyhow::ensure!(!frames.is_empty(), "cannot cache empty audio frame buffer");
    write_wav(path, frames)
        .with_context(|| format!("failed to write cached WAV {}", path.display()))
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
            .map(|sample| sample as f32 / 32768.0)
            .collect(),
        voice_signatures: Vec::new(),
    }])
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
        fn enqueue(&mut self, _plan: MouthSyntheticPlan) -> Result<()> {
            self.enqueues += 1;
            self.queued_audio.push_back(vec![AudioFrame {
                captured_at: crate::time::ExactTimestamp::now(),
                sample_rate_hz: 22_050,
                channels: 1,
                samples: vec![0.2, 0.1],
                voice_signatures: Vec::new(),
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
        let mut cache = MemorySyntheticCache::default();
        let unit = SyntheticUnit::Backchannel("Okay.".to_string());
        let warmed_frame = AudioFrame {
            captured_at: crate::time::ExactTimestamp::now(),
            sample_rate_hz: 22_050,
            channels: 1,
            samples: vec![0.3, 0.2],
            voice_signatures: Vec::new(),
        };
        cache
            .put(&unit, std::slice::from_ref(&warmed_frame))
            .expect("cache put");

        let inner = FakeTts::default();
        let mut tts = CachedTextToSpeech::new(inner, cache);
        tts.enqueue(MouthSyntheticPlan::from(unit))
            .expect("enqueue");

        let frames = tts.poll_audio().expect("poll");
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].samples, warmed_frame.samples);
        assert_eq!(tts.inner.enqueues, 0);
    }

    #[test]
    fn backchannel_cache_miss_uses_tts_then_warms_cache() {
        let cache = MemorySyntheticCache::default();
        let inner = FakeTts::default();
        let mut tts = CachedTextToSpeech::new(inner, cache);
        let unit = SyntheticUnit::Backchannel("Okay.".to_string());

        tts.enqueue(MouthSyntheticPlan::from(unit.clone()))
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
        let cache = MemorySyntheticCache::default();
        let inner = FakeTts::default();
        let mut tts = CachedTextToSpeech::new(inner, cache);
        let unit = SyntheticUnit::CompleteSentence("I think that works.".to_string());

        tts.enqueue(MouthSyntheticPlan::from(unit))
            .expect("enqueue");
        assert_eq!(tts.inner.enqueues, 1);
    }

    #[test]
    fn file_cache_round_trip_for_backchannel() {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("listenbury-cache-test-{ts}"));
        let mut cache = FileSyntheticCache::new(&root, "voice.onnx", Some("cfg".to_string()));
        let unit = SyntheticUnit::Backchannel("Okay.".to_string());
        let frame = AudioFrame {
            captured_at: crate::time::ExactTimestamp::now(),
            sample_rate_hz: 22_050,
            channels: 1,
            samples: vec![0.25, -0.25],
            voice_signatures: Vec::new(),
        };
        cache
            .put(&unit, &[frame])
            .expect("cache put should succeed");
        let got = cache.get(&unit).expect("cache get should succeed");
        assert!(got.is_some());

        std::fs::remove_dir_all(root).expect("temporary cache dir should be removed");
    }
}
