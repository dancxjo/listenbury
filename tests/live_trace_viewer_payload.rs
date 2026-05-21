use listenbury::trace::viewer_payload::live_trace_jsonl_to_viewer_payload;
use serde_json::Value;

#[test]
fn sample_live_trace_jsonl_converts_to_mixed_word_and_event_lanes() {
    let payload = live_trace_jsonl_to_viewer_payload(include_str!(
        "../examples/browser-transcript-player/fixtures/live-trace.sample.jsonl"
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
        "../examples/browser-transcript-player/fixtures/live-trace.sample.viewer.json"
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

#[test]
fn prosody_plan_event_expands_to_phone_breath_and_break_lanes() {
    let jsonl = r#"{"turn":1,"kind":"prosody_plan","t_unix_ns":0,"elapsed_ms":1200,"artifact":{"utterance_id":"utt-1","segments":[{"word":"going","t0":1.24,"t1":1.52,"phones":[{"p":"g","t0":1.24,"t1":1.29,"nucleus":false,"pace_target_ms":50},{"p":"oʊ","t0":1.29,"t1":1.45,"nucleus":true,"pace_target_ms":173}],"break_hint_ms":160,"break_reason":"punctuation"}],"breath_groups":[{"t0":1.24,"t1":1.52}]}}"#;

    let payload =
        live_trace_jsonl_to_viewer_payload(jsonl).expect("prosody plan trace event should convert");

    assert!(
        payload.events.iter().any(|event| event.lane == "Phones"
            && event.kind == "phone"
            && event.label.as_deref() == Some("oʊ")
            && event.start_ms == 1290
            && event.end_ms == Some(1450)),
        "expected phone lane span from prosody plan"
    );
    assert!(
        payload
            .events
            .iter()
            .any(|event| event.lane == "Breath Groups" && event.kind == "breath_group"),
        "expected breath group lane span from prosody plan"
    );
    assert!(
        payload.markers.iter().any(|marker| marker.lane == "Breaks"
            && marker.kind == "break"
            && marker.at_ms == 1520
            && marker.label.as_deref() == Some("160ms")),
        "expected break marker from prosody plan"
    );
}
