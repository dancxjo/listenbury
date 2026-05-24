use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};

use crate::audio::{AudioFrame, AudioOutput};
use crate::mind::llm::{GenerationRequest, LlmEngine, LlmEvent, MockLlmEngine};
use crate::mouth::planner::{ExpressiveUnit, MouthSyntheticPlan, SyntheticPlanner};
use crate::mouth::player::{PlaybackEvent, Player, SequentialPlayer};
use crate::mouth::tts::TextToSpeech;
use crate::speech::recognizer::SpeechRecognizer;
use crate::speech::transcript::TranscriptChunk;
use crate::time::{Clock, ExactTimestamp};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaybackCheckEventKind {
    AsrStarted,
    AsrFinished,
    LlmToken,
    PlannerSyntheticReady,
    TtsQueued,
    PlaybackStarted,
    PlaybackFinished,
    DeviceFramePushed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlaybackCheckEvent {
    pub kind: PlaybackCheckEventKind,
    pub at: ExactTimestamp,
    pub text: Option<String>,
}

impl PlaybackCheckEvent {
    fn new(kind: PlaybackCheckEventKind, at: ExactTimestamp, text: Option<String>) -> Self {
        Self { kind, at, text }
    }
}

pub fn run_playback_check(
    clock: Arc<dyn Clock>,
    mut tick: impl FnMut(Duration),
) -> Result<Vec<PlaybackCheckEvent>> {
    let mut events = Vec::new();
    let mut asr = StubSpeechRecognizer::new("hello pete");

    events.push(PlaybackCheckEvent::new(
        PlaybackCheckEventKind::AsrStarted,
        clock.now(),
        None,
    ));
    asr.push_frame(&AudioFrame {
        captured_at: clock.now(),
        sample_rate_hz: 16_000,
        channels: 1,
        samples: vec![0.0; 160],
        voice_signatures: Vec::new(),
    })?;
    tick(Duration::from_millis(10));

    let chunks = asr.poll_chunks()?;
    let transcript = chunks
        .into_iter()
        .find(|chunk| chunk.is_final)
        .context("stub ASR did not produce a final transcript")?
        .text;
    events.push(PlaybackCheckEvent::new(
        PlaybackCheckEventKind::AsrFinished,
        clock.now(),
        Some(transcript.clone()),
    ));
    tick(Duration::from_millis(10));

    let mut llm = MockLlmEngine::default();
    let generation = llm.start(GenerationRequest {
        prompt: transcript,
        max_tokens: Some(16),
        stop: Vec::new(),
    })?;
    let mut planner = SyntheticPlanner::default();
    let mut planned_units = Vec::new();

    loop {
        let llm_events = llm.poll(generation)?;
        for event in &llm_events {
            if let LlmEvent::Token { text } = event {
                events.push(PlaybackCheckEvent::new(
                    PlaybackCheckEventKind::LlmToken,
                    clock.now(),
                    Some(text.clone()),
                ));
                tick(Duration::from_millis(10));
            }
        }
        planned_units.extend(planner.ingest(&llm_events));
        if llm_events
            .iter()
            .any(|event| matches!(event, LlmEvent::Completed))
        {
            break;
        }
    }

    for unit in &planned_units {
        if let ExpressiveUnit::Synthetic(plan) = unit {
            events.push(PlaybackCheckEvent::new(
                PlaybackCheckEventKind::PlannerSyntheticReady,
                clock.now(),
                Some(plan.text().to_string()),
            ));
        }
    }

    let tts = StubTextToSpeech::new(clock.clone());
    let mut player = SequentialPlayer::with_clock(tts, clock.clone());
    let mut device = StubAudioOutput::default();

    for unit in planned_units {
        player.enqueue(unit)?;
    }

    for event in player.poll()? {
        if let PlaybackEvent::SyntheticQueued { text, .. } = event {
            events.push(PlaybackCheckEvent::new(
                PlaybackCheckEventKind::TtsQueued,
                clock.now(),
                Some(text),
            ));
        }
    }
    tick(Duration::from_millis(10));

    for event in player.poll()? {
        match event {
            PlaybackEvent::SyntheticStarted { text, at, .. } => {
                events.push(PlaybackCheckEvent::new(
                    PlaybackCheckEventKind::PlaybackStarted,
                    at,
                    Some(text),
                ));
            }
            PlaybackEvent::SyntheticFinished { at, .. } => {
                events.push(PlaybackCheckEvent::new(
                    PlaybackCheckEventKind::PlaybackFinished,
                    at,
                    None,
                ));
            }
            _ => {}
        }
    }

    for frame in player.poll_audio()? {
        device.push_frame(frame)?;
        events.push(PlaybackCheckEvent::new(
            PlaybackCheckEventKind::DeviceFramePushed,
            clock.now(),
            None,
        ));
    }

    Ok(events)
}

struct StubSpeechRecognizer {
    transcript: Option<String>,
}

impl StubSpeechRecognizer {
    fn new(transcript: impl Into<String>) -> Self {
        Self {
            transcript: Some(transcript.into()),
        }
    }
}

impl SpeechRecognizer for StubSpeechRecognizer {
    fn push_frame(&mut self, _frame: &AudioFrame) -> Result<()> {
        Ok(())
    }

    fn poll_chunks(&mut self) -> Result<Vec<TranscriptChunk>> {
        Ok(self
            .transcript
            .take()
            .map(|text| TranscriptChunk {
                text,
                is_final: true,
            })
            .into_iter()
            .collect())
    }
}

struct StubTextToSpeech {
    clock: Arc<dyn Clock>,
    ready_frames: Vec<AudioFrame>,
}

impl StubTextToSpeech {
    fn new(clock: Arc<dyn Clock>) -> Self {
        Self {
            clock,
            ready_frames: Vec::new(),
        }
    }
}

impl TextToSpeech for StubTextToSpeech {
    fn enqueue(&mut self, _plan: MouthSyntheticPlan) -> Result<()> {
        self.ready_frames.push(AudioFrame {
            captured_at: self.clock.now(),
            sample_rate_hz: 22_050,
            channels: 1,
            samples: vec![0.0; 220],
            voice_signatures: Vec::new(),
        });
        Ok(())
    }

    fn poll_audio(&mut self) -> Result<Vec<AudioFrame>> {
        Ok(std::mem::take(&mut self.ready_frames))
    }

    fn stop(&mut self) -> Result<()> {
        self.ready_frames.clear();
        Ok(())
    }
}

#[derive(Default)]
struct StubAudioOutput {
    frames: Vec<AudioFrame>,
}

impl AudioOutput for StubAudioOutput {
    fn push_frame(&mut self, frame: AudioFrame) -> Result<()> {
        self.frames.push(frame);
        Ok(())
    }
}
