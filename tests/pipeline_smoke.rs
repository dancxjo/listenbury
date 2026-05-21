use std::sync::Arc;

use listenbury::playback_check::{PlaybackCheckEvent, PlaybackCheckEventKind, run_playback_check};
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
