use std::sync::Arc;

use listenbury::hearing::VadResult;
use listenbury::mouth::player::{PlaybackEvent, PlaybackUnitId};
use listenbury::playback_check::{PlaybackCheckEvent, PlaybackCheckEventKind, run_playback_check};
use listenbury::soundscape::{SoundEventKind, SoundscapePipelineAdapter};
use listenbury::speech::transcript::TranscriptChunk;
use listenbury::{Clock, FakeClock};

fn index_of(events: &[PlaybackCheckEvent], kind: PlaybackCheckEventKind) -> usize {
    events
        .iter()
        .position(|event| event.kind == kind)
        .unwrap_or_else(|| panic!("missing {kind:?} in {events:?}"))
}

#[test]
fn playback_check_orders_stubbed_asr_planner_tts_and_device_events() {
    let fake_clock = FakeClock::from_unix_nanos(1_000_000_000);
    let clock: Arc<dyn Clock> = Arc::new(fake_clock.clone());

    let events = run_playback_check(clock, |duration| {
        fake_clock.advance(duration);
    })
    .expect("playback check should run");

    let asr_started = index_of(&events, PlaybackCheckEventKind::AsrStarted);
    let asr_finished = index_of(&events, PlaybackCheckEventKind::AsrFinished);
    let llm_token = index_of(&events, PlaybackCheckEventKind::LlmToken);
    let planner_ready = index_of(&events, PlaybackCheckEventKind::PlannerSpeechReady);
    let tts_queued = index_of(&events, PlaybackCheckEventKind::TtsQueued);
    let playback_started = index_of(&events, PlaybackCheckEventKind::PlaybackStarted);
    let playback_finished = index_of(&events, PlaybackCheckEventKind::PlaybackFinished);
    let device_pushed = index_of(&events, PlaybackCheckEventKind::DeviceFramePushed);

    assert!(asr_started < asr_finished);
    assert!(asr_finished < llm_token);
    assert!(llm_token < planner_ready);
    assert!(planner_ready < tts_queued);
    assert!(tts_queued < playback_started);
    assert!(playback_started < playback_finished);
    assert!(playback_finished < device_pushed);

    for pair in events.windows(2) {
        assert!(
            pair[0].at <= pair[1].at,
            "events should be timestamp-ordered: {pair:?}"
        );
    }

    assert_eq!(
        events[asr_started].at.unix_nanos, 1_000_000_000,
        "the fake clock should make the first edge deterministic"
    );
    assert_eq!(
        events[asr_finished].at.unix_nanos, 1_010_000_000,
        "ASR finish should be the first synthetic 10ms tick"
    );
}

#[test]
fn soundscape_adapter_converts_playback_and_microphone_events() {
    let fake_clock = FakeClock::from_unix_nanos(3_000_000_000);
    let clock: Arc<dyn Clock> = Arc::new(fake_clock.clone());
    let events = run_playback_check(clock, |duration| {
        fake_clock.advance(duration);
    })
    .expect("playback check should run");

    let asr_finished = events
        .iter()
        .find(|event| event.kind == PlaybackCheckEventKind::AsrFinished)
        .expect("ASR finished event");
    let playback_started = events
        .iter()
        .find(|event| event.kind == PlaybackCheckEventKind::PlaybackStarted)
        .expect("playback started event");
    let playback_started_text = playback_started
        .text
        .clone()
        .expect("playback started event should include text");

    let adapter = SoundscapePipelineAdapter::default();
    let mic_frame = listenbury::AudioFrame {
        captured_at: asr_finished.at,
        sample_rate_hz: 16_000,
        channels: 1,
        samples: vec![0.15; 160],
        voice_signatures: Vec::new(),
    };
    let rendered_playback = listenbury::AudioFrame {
        captured_at: playback_started.at,
        sample_rate_hz: 22_050,
        channels: 1,
        samples: vec![0.08; 220],
        voice_signatures: Vec::new(),
    };
    let vad = VadResult {
        speech_prob: 0.93,
        is_speech: true,
    };
    let transcript = TranscriptChunk {
        text: asr_finished
            .text
            .clone()
            .expect("ASR finished event should carry transcript text"),
        is_final: true,
    };
    let playback_event = PlaybackEvent::SpeechStarted {
        id: PlaybackUnitId(0),
        text: playback_started_text,
        at: playback_started.at,
    };

    let observed = adapter.observed_from_audio_vad_asr(&mic_frame, Some(vad), Some(&transcript));
    let expected = adapter
        .expected_from_playback_event(&playback_event, Some(&rendered_playback))
        .expect("expected playback sound");
    let frame = adapter.emit_frame(&observed, Some(&expected), Some(vad));
    let attributed = adapter.source_attributed_transcript(&observed, &transcript, Some(vad));

    assert!(
        frame
            .events
            .iter()
            .any(|event| event.kind == SoundEventKind::PlaybackActivity),
        "frame should include playback activity"
    );
    assert!(
        frame
            .events
            .iter()
            .any(|event| event.kind == SoundEventKind::VoiceActivity),
        "frame should include microphone voice activity"
    );
    assert_eq!(attributed.text, transcript.text);
    assert_eq!(observed.transcript_hypotheses.len(), 1);
}
