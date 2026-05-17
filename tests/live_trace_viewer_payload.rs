use listenbury::trace::viewer_payload::live_trace_jsonl_to_viewer_payload;
use serde_json::Value;

#[test]
fn sample_live_trace_jsonl_converts_to_mixed_word_and_event_lanes() {
    let payload = live_trace_jsonl_to_viewer_payload(include_str!(
        "../web/browser-transcript-player/live-trace.sample.jsonl"
    ))
    .expect("sample live trace JSONL should convert");

    assert!(
        payload.streams.len() >= 2,
        "expected transcript + intended speech lanes"
    );
    assert!(
        payload
            .streams
            .iter()
            .any(|lane| lane.label == "User transcript" && !lane.stream.words.is_empty()),
        "expected user transcript words"
    );
    assert!(
        payload
            .streams
            .iter()
            .any(|lane| lane.label == "Pete intended speech" && !lane.stream.words.is_empty()),
        "expected intended speech words"
    );

    assert!(
        payload
            .events
            .iter()
            .any(|event| event.lane == "Overlap" && event.kind == "overlap"),
        "overlap span should be represented as an event lane item"
    );
    assert!(
        payload
            .events
            .iter()
            .any(|event| event.lane == "Interruption" && event.kind == "yield"),
        "yield span should be represented as an interruption event"
    );

    assert!(
        payload
            .markers
            .iter()
            .any(|marker| marker.lane == "Interruption" && marker.kind == "interruption_detected"),
        "interruption marker should remain visible"
    );
    assert!(
        payload
            .markers
            .iter()
            .any(|marker| marker.lane == "Latency" && marker.kind == "first_llm_token"),
        "latency marker should be preserved"
    );
    assert!(
        payload
            .markers
            .iter()
            .any(|marker| marker.lane == "Latency"
                && marker.kind == "first_tts_audio_frame_available"),
        "tts timing marker should be preserved"
    );
    assert!(
        payload
            .markers
            .iter()
            .any(|marker| marker.lane == "Latency" && marker.kind == "playback_started"),
        "playback timing marker should be preserved"
    );
    assert!(
        payload
            .events
            .iter()
            .any(|event| event.kind == "overlap" && event.audio_ref.is_some()),
        "overlap event should expose clip audio references when artifacts provide them"
    );
    assert!(
        payload
            .markers
            .iter()
            .any(|marker| marker.kind == "playback_started" && marker.audio_ref.is_some()),
        "playback marker should expose clip audio references when artifacts provide them"
    );
}

#[test]
fn sample_live_trace_viewer_json_has_word_and_event_lanes() {
    let payload: Value = serde_json::from_str(include_str!(
        "../web/browser-transcript-player/live-trace.sample.viewer.json"
    ))
    .expect("sample viewer payload JSON should parse");

    let streams = payload["streams"]
        .as_array()
        .expect("streams should be present");
    let events = payload["events"]
        .as_array()
        .expect("events should be present");
    let markers = payload["markers"]
        .as_array()
        .expect("markers should be present");

    assert!(
        !streams.is_empty() && !events.is_empty() && !markers.is_empty(),
        "sample viewer payload should include streams, span events, and markers"
    );
    assert!(
        events.iter().any(|event| event["audio_ref"].is_object()),
        "sample viewer payload should include event clip references"
    );
    assert!(
        markers.iter().any(|marker| marker["audio_ref"].is_object()),
        "sample viewer payload should include marker clip references"
    );
}
