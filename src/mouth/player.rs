use std::collections::VecDeque;

use anyhow::Result;

use crate::audio::frame::AudioFrame;
use crate::mouth::planner::{ExpressiveUnit, FaceCommand, MouthCommand, SpeechPlan};
use crate::mouth::tts::TextToSpeech;
use crate::time::ExactTimestamp;

/// A unique identifier for a speech playback unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaybackUnitId(pub u64);

/// An event emitted by the [`Player`] during execution.
///
/// Events carry timing information so that upstream consumers (e.g.
/// `ConversationController`) can record precise playback state without needing
/// to own the TTS or audio device.
#[derive(Debug, Clone)]
pub enum PlaybackEvent {
    /// A speech unit has been accepted into the synthesis queue.
    SpeechQueued { id: PlaybackUnitId, text: String },
    /// Audio for the speech unit has begun playback.
    SpeechStarted {
        id: PlaybackUnitId,
        text: String,
        at: ExactTimestamp,
    },
    /// Audio for the speech unit has finished playback.
    SpeechFinished {
        id: PlaybackUnitId,
        at: ExactTimestamp,
    },
    /// Pete's facial expression has changed at playback time.
    FaceChanged { emoji: String, at: ExactTimestamp },
    /// Playback was gracefully faded (or treated as a stop when the audio
    /// backend does not support fading).
    PlaybackFaded { millis: u64, at: ExactTimestamp },
    /// Playback was stopped immediately.
    PlaybackStopped { at: ExactTimestamp },
    /// An error occurred during playback.
    Error { message: String },
}

/// Abstraction for an ordered executor of [`ExpressiveUnit`]s.
///
/// The player consumes ordered units from a [`SpeechPlanner`] and turns them
/// into synthesized audio (via [`TextToSpeech`]) plus timed [`PlaybackEvent`]s.
/// Face commands are held back until just before the next speech unit starts,
/// keeping them aligned with actual playback rather than planning time.
///
/// [`SpeechPlanner`]: crate::mouth::planner::SpeechPlanner
pub trait Player {
    /// Accept a new expressive unit (speech or face) into the player's queue.
    fn enqueue(&mut self, unit: ExpressiveUnit) -> Result<()>;

    /// Apply a mouth command, potentially interrupting or modifying playback.
    fn handle_command(&mut self, command: MouthCommand) -> Result<()>;

    /// Advance the player state and return any events that have become ready.
    ///
    /// Callers should drain synthesized audio frames with [`poll_audio`] after
    /// each `poll` call.
    ///
    /// [`poll_audio`]: Player::poll_audio
    fn poll(&mut self) -> Result<Vec<PlaybackEvent>>;

    /// Return any synthesized [`AudioFrame`]s that are ready for playback.
    fn poll_audio(&mut self) -> Result<Vec<AudioFrame>>;
}

/// Internal state for a speech unit that is currently being synthesized.
struct PendingSynthesis {
    id: PlaybackUnitId,
    text: String,
}

/// A sequential, non-overlapping [`Player`] implementation.
///
/// `SequentialPlayer` processes [`ExpressiveUnit`]s one at a time:
///
/// - Speech units are submitted to the wrapped [`TextToSpeech`] implementation.
///   `SpeechStarted` and `SpeechFinished` events are emitted when audio frames
///   become available (i.e. when synthesis completes), which is the earliest
///   we can establish a playback-time anchor without sample-accurate hooks.
///
/// - Face commands are buffered and emitted just before the next speech unit
///   starts, so that the face change is aligned with the audio rather than the
///   planning stage.  If there is no pending speech when a face command is
///   processed, it is emitted immediately.
///
/// - `MouthCommand::StopNow` clears all queued speech and face commands and
///   stops the TTS backend.
///
/// - `MouthCommand::FadeOut` is treated as `StopNow` for now (the CPAL backend
///   does not support gradual fading).  A `PlaybackFaded` event is emitted and
///   the limitation is logged.
pub struct SequentialPlayer<T: TextToSpeech> {
    tts: T,
    queue: VecDeque<ExpressiveUnit>,
    pending_faces: Vec<FaceCommand>,
    synthesis: Option<PendingSynthesis>,
    audio_buffer: Vec<AudioFrame>,
    /// Events enqueued by `handle_command` and drained on the next `poll`.
    command_events: Vec<PlaybackEvent>,
    next_id: u64,
}

impl<T: TextToSpeech> SequentialPlayer<T> {
    /// Create a new player backed by the given [`TextToSpeech`] implementation.
    pub fn new(tts: T) -> Self {
        Self {
            tts,
            queue: VecDeque::new(),
            pending_faces: Vec::new(),
            synthesis: None,
            audio_buffer: Vec::new(),
            command_events: Vec::new(),
            next_id: 0,
        }
    }

    fn alloc_id(&mut self) -> PlaybackUnitId {
        let id = PlaybackUnitId(self.next_id);
        self.next_id += 1;
        id
    }

    fn emit_face(face_cmd: FaceCommand, at: ExactTimestamp) -> PlaybackEvent {
        let emoji = match face_cmd {
            FaceCommand::SetEmoji(e) => e,
            FaceCommand::Clear => String::new(),
        };
        PlaybackEvent::FaceChanged { emoji, at }
    }
}

impl<T: TextToSpeech> Player for SequentialPlayer<T> {
    fn enqueue(&mut self, unit: ExpressiveUnit) -> Result<()> {
        self.queue.push_back(unit);
        Ok(())
    }

    fn handle_command(&mut self, command: MouthCommand) -> Result<()> {
        match command {
            MouthCommand::Speak(plan) => {
                self.enqueue(ExpressiveUnit::Speech(plan))?;
            }
            MouthCommand::StopNow => {
                self.queue.clear();
                self.pending_faces.clear();
                self.synthesis = None;
                self.audio_buffer.clear();
                self.tts.stop()?;
                self.command_events.push(PlaybackEvent::PlaybackStopped {
                    at: ExactTimestamp::now(),
                });
            }
            MouthCommand::FadeOut { millis } => {
                // First-pass fallback: the CPAL backend does not support gradual
                // volume fading, so we stop immediately and emit PlaybackFaded.
                tracing::warn!(
                    millis,
                    "FadeOut is not supported by the audio backend; stopping immediately"
                );
                self.queue.clear();
                self.pending_faces.clear();
                self.synthesis = None;
                self.audio_buffer.clear();
                self.tts.stop()?;
                self.command_events.push(PlaybackEvent::PlaybackFaded {
                    millis,
                    at: ExactTimestamp::now(),
                });
            }
        }
        Ok(())
    }

    fn poll(&mut self) -> Result<Vec<PlaybackEvent>> {
        // Drain any events produced by handle_command first.
        let mut events: Vec<PlaybackEvent> = std::mem::take(&mut self.command_events);

        // Check if the current in-flight synthesis has completed.
        if self.synthesis.is_some() {
            let new_frames = self.tts.poll_audio()?;
            if !new_frames.is_empty() {
                // All frames for this unit arrived; synthesis is complete.
                let synth = self.synthesis.take().unwrap();
                let at = ExactTimestamp::now();
                events.push(PlaybackEvent::SpeechStarted {
                    id: synth.id,
                    text: synth.text,
                    at,
                });
                self.audio_buffer.extend(new_frames);
                events.push(PlaybackEvent::SpeechFinished { id: synth.id, at });
            }
            // If synthesis is still in flight (frames not yet ready), return
            // what we have so far and wait for the next poll.
            if self.synthesis.is_some() {
                return Ok(events);
            }
        }

        // Advance the queue: drain leading face commands, then start the next
        // speech unit.
        while let Some(unit) = self.queue.pop_front() {
            match unit {
                ExpressiveUnit::Face(cmd) => {
                    // Accumulate face commands; flush them just before the next
                    // speech unit starts.
                    self.pending_faces.push(cmd);
                }
                ExpressiveUnit::Speech(plan) => {
                    // Emit all buffered face commands right before this speech
                    // unit starts, aligning them with actual playback time.
                    let at = ExactTimestamp::now();
                    for face_cmd in self.pending_faces.drain(..) {
                        events.push(Self::emit_face(face_cmd, at));
                    }

                    let id = self.alloc_id();
                    let text = plan.text().to_string();
                    events.push(PlaybackEvent::SpeechQueued {
                        id,
                        text: text.clone(),
                    });
                    self.tts.enqueue(plan)?;
                    self.synthesis = Some(PendingSynthesis { id, text });
                    // Only process one speech unit per poll pass so that the
                    // caller has a chance to drain audio frames between units.
                    break;
                }
            }
        }

        // If the queue is empty and there are still pending face commands with
        // no speech to follow, emit them immediately.
        if self.synthesis.is_none() && self.queue.is_empty() && !self.pending_faces.is_empty() {
            let at = ExactTimestamp::now();
            for face_cmd in self.pending_faces.drain(..) {
                events.push(Self::emit_face(face_cmd, at));
            }
        }

        Ok(events)
    }

    fn poll_audio(&mut self) -> Result<Vec<AudioFrame>> {
        Ok(std::mem::take(&mut self.audio_buffer))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mouth::planner::{SpeechPlan, SpeechUnit};

    // ---------------------------------------------------------------------------
    // Test helpers
    // ---------------------------------------------------------------------------

    /// A [`TextToSpeech`] mock that returns a single dummy [`AudioFrame`] the
    /// first time `poll_audio` is called after each `enqueue`.
    struct MockTts {
        pub enqueued_texts: Vec<String>,
        pub stop_count: usize,
        frames_ready: Vec<AudioFrame>,
    }

    impl MockTts {
        fn new() -> Self {
            Self {
                enqueued_texts: Vec::new(),
                stop_count: 0,
                frames_ready: Vec::new(),
            }
        }
    }

    impl TextToSpeech for MockTts {
        fn enqueue(&mut self, plan: SpeechPlan) -> Result<()> {
            self.enqueued_texts.push(plan.text().to_string());
            // Pre-load a frame so the next poll_audio call "completes" synthesis.
            self.frames_ready.push(AudioFrame {
                captured_at: ExactTimestamp::now(),
                sample_rate_hz: 22_050,
                channels: 1,
                samples: vec![0.1, 0.2, 0.3],
            });
            Ok(())
        }

        fn poll_audio(&mut self) -> Result<Vec<AudioFrame>> {
            Ok(std::mem::take(&mut self.frames_ready))
        }

        fn stop(&mut self) -> Result<()> {
            self.stop_count += 1;
            self.enqueued_texts.clear();
            self.frames_ready.clear();
            Ok(())
        }
    }

    fn sentence(text: &str) -> ExpressiveUnit {
        ExpressiveUnit::Speech(SpeechPlan::new(SpeechUnit::CompleteSentence(
            text.to_string(),
        )))
    }

    fn face(emoji: &str) -> ExpressiveUnit {
        ExpressiveUnit::Face(FaceCommand::SetEmoji(emoji.to_string()))
    }

    fn is_speech_queued(ev: &PlaybackEvent, expected_text: &str) -> bool {
        matches!(ev, PlaybackEvent::SpeechQueued { text, .. } if text == expected_text)
    }

    fn is_speech_started(ev: &PlaybackEvent, expected_text: &str) -> bool {
        matches!(ev, PlaybackEvent::SpeechStarted { text, .. } if text == expected_text)
    }

    fn is_speech_finished(ev: &PlaybackEvent) -> bool {
        matches!(ev, PlaybackEvent::SpeechFinished { .. })
    }

    fn is_face_changed(ev: &PlaybackEvent, expected_emoji: &str) -> bool {
        matches!(ev, PlaybackEvent::FaceChanged { emoji, .. } if emoji == expected_emoji)
    }

    // ---------------------------------------------------------------------------
    // Tests
    // ---------------------------------------------------------------------------

    /// Enqueueing a speech unit should produce SpeechQueued, then on the next
    /// poll SpeechStarted and SpeechFinished.
    #[test]
    fn speech_unit_produces_started_and_finished_events() {
        let mut player = SequentialPlayer::new(MockTts::new());
        player.enqueue(sentence("Hello.")).unwrap();

        // First poll: unit is dequeued, enqueued to TTS, SpeechQueued emitted.
        let ev1 = player.poll().unwrap();
        assert!(
            ev1.iter().any(|e| is_speech_queued(e, "Hello.")),
            "expected SpeechQueued; got {ev1:?}"
        );
        assert!(
            !ev1.iter().any(is_speech_finished),
            "SpeechFinished must not appear on first poll"
        );

        // Second poll: frames are available -> SpeechStarted + SpeechFinished.
        let ev2 = player.poll().unwrap();
        assert!(
            ev2.iter().any(|e| is_speech_started(e, "Hello.")),
            "expected SpeechStarted; got {ev2:?}"
        );
        assert!(
            ev2.iter().any(is_speech_finished),
            "expected SpeechFinished; got {ev2:?}"
        );
    }

    /// Audio frames are exposed through poll_audio after synthesis completes.
    #[test]
    fn audio_frames_available_after_synthesis() {
        let mut player = SequentialPlayer::new(MockTts::new());
        player.enqueue(sentence("Hello.")).unwrap();

        player.poll().unwrap(); // enqueues to TTS
        player.poll().unwrap(); // synthesis completes, buffers frames

        let frames = player.poll_audio().unwrap();
        assert!(!frames.is_empty(), "expected audio frames after synthesis");
    }

    /// A face command emitted when no speech is pending should be emitted
    /// immediately on the next poll.
    #[test]
    fn face_command_emitted_immediately_when_no_speech_pending() {
        let mut player = SequentialPlayer::new(MockTts::new());
        player.enqueue(face("🙂")).unwrap();

        let events = player.poll().unwrap();
        assert!(
            events.iter().any(|e| is_face_changed(e, "🙂")),
            "expected FaceChanged immediately; got {events:?}"
        );
    }

    /// A face command that appears between two speech units must be emitted
    /// *after* the first speech unit finishes and *before* the second starts.
    #[test]
    fn face_command_between_speech_units_emitted_at_playback_time() {
        let mut player = SequentialPlayer::new(MockTts::new());
        player.enqueue(sentence("Okay.")).unwrap();
        player.enqueue(face("🙂")).unwrap();
        player.enqueue(sentence("I see.")).unwrap();

        // Poll 1: "Okay." queued to TTS.
        let ev1 = player.poll().unwrap();
        assert!(ev1.iter().any(|e| is_speech_queued(e, "Okay.")));
        assert!(
            !ev1.iter().any(|e| is_face_changed(e, "🙂")),
            "face must not appear before first speech starts"
        );

        // Poll 2: "Okay." synthesis done; face change and "I see." queued.
        let ev2 = player.poll().unwrap();
        assert!(ev2.iter().any(|e| is_speech_started(e, "Okay.")));
        assert!(ev2.iter().any(is_speech_finished));
        assert!(
            ev2.iter().any(|e| is_face_changed(e, "🙂")),
            "face must appear after first speech finishes; got {ev2:?}"
        );
        assert!(ev2.iter().any(|e| is_speech_queued(e, "I see.")));

        // Ordering within poll 2: Finished → FaceChanged → Queued
        let finished_pos = ev2
            .iter()
            .position(is_speech_finished)
            .expect("SpeechFinished");
        let face_pos = ev2
            .iter()
            .position(|e| is_face_changed(e, "🙂"))
            .expect("FaceChanged");
        let queued_pos = ev2
            .iter()
            .position(|e| is_speech_queued(e, "I see."))
            .expect("SpeechQueued");
        assert!(
            finished_pos < face_pos,
            "FaceChanged must come after SpeechFinished"
        );
        assert!(
            face_pos < queued_pos,
            "FaceChanged must come before SpeechQueued for next unit"
        );
    }

    /// StopNow clears queued speech and face commands; subsequent polls return
    /// no speech or face events.
    #[test]
    fn stop_clears_queued_speech_and_faces() {
        let mut player = SequentialPlayer::new(MockTts::new());
        player.enqueue(sentence("Okay.")).unwrap();
        player.enqueue(face("🙂")).unwrap();
        player.enqueue(sentence("I see.")).unwrap();

        player.handle_command(MouthCommand::StopNow).unwrap();

        // First poll after stop returns PlaybackStopped only.
        let ev = player.poll().unwrap();
        assert!(
            ev.iter()
                .any(|e| matches!(e, PlaybackEvent::PlaybackStopped { .. })),
            "expected PlaybackStopped; got {ev:?}"
        );
        assert!(
            !ev.iter().any(|e| matches!(
                e,
                PlaybackEvent::SpeechQueued { .. }
                    | PlaybackEvent::SpeechStarted { .. }
                    | PlaybackEvent::FaceChanged { .. }
            )),
            "no speech or face events expected after stop; got {ev:?}"
        );

        // Subsequent poll returns nothing.
        let ev2 = player.poll().unwrap();
        assert!(ev2.is_empty(), "expected empty after stop; got {ev2:?}");
    }

    /// FadeOut emits a PlaybackFaded event and leaves no stale queued state.
    #[test]
    fn fade_emits_event_and_clears_queue() {
        let mut player = SequentialPlayer::new(MockTts::new());
        player.enqueue(sentence("Hello.")).unwrap();
        player.enqueue(face("😊")).unwrap();
        player.enqueue(sentence("World.")).unwrap();

        // Start synthesizing the first unit.
        player.poll().unwrap();

        player
            .handle_command(MouthCommand::FadeOut { millis: 300 })
            .unwrap();

        let ev = player.poll().unwrap();
        assert!(
            ev.iter().any(
                |e| matches!(e, PlaybackEvent::PlaybackFaded { millis, .. } if *millis == 300)
            ),
            "expected PlaybackFaded(300); got {ev:?}"
        );
        assert!(
            !ev.iter().any(|e| matches!(
                e,
                PlaybackEvent::SpeechQueued { .. }
                    | PlaybackEvent::SpeechStarted { .. }
                    | PlaybackEvent::FaceChanged { .. }
            )),
            "no stale speech/face events after fade; got {ev:?}"
        );

        // No more events.
        let ev2 = player.poll().unwrap();
        assert!(ev2.is_empty(), "expected empty after fade; got {ev2:?}");
    }

    /// MouthCommand::Speak enqueues a speech plan directly.
    #[test]
    fn speak_command_enqueues_speech() {
        let mut player = SequentialPlayer::new(MockTts::new());
        let plan = SpeechPlan::new(SpeechUnit::CompleteSentence("Via command.".to_string()));
        player.handle_command(MouthCommand::Speak(plan)).unwrap();

        let ev = player.poll().unwrap();
        assert!(
            ev.iter().any(|e| is_speech_queued(e, "Via command.")),
            "expected SpeechQueued via Speak command; got {ev:?}"
        );
    }

    /// FaceCommand::Clear emits a FaceChanged event with an empty emoji string.
    #[test]
    fn face_clear_emits_empty_emoji() {
        let mut player = SequentialPlayer::new(MockTts::new());
        player
            .enqueue(ExpressiveUnit::Face(FaceCommand::Clear))
            .unwrap();

        let ev = player.poll().unwrap();
        assert!(
            ev.iter().any(|e| is_face_changed(e, "")),
            "expected FaceChanged(\"\") for Clear; got {ev:?}"
        );
    }
}
