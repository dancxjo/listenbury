use listenbury::word::TimedWordStream;
use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Deserialize)]
struct BrowserTranscriptPlayerPayload {
    title: Option<String>,
    audio: Option<BrowserTranscriptPlayerAudio>,
    streams: Vec<BrowserTranscriptPlayerLane>,
    #[serde(default)]
    events: Vec<BrowserTranscriptPlayerEvent>,
    #[serde(default)]
    markers: Vec<BrowserTranscriptPlayerMarker>,
}

#[derive(Debug, Deserialize)]
struct BrowserTranscriptPlayerAudio {
    url: String,
    duration_ms: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct BrowserTranscriptPlayerLane {
    label: Option<String>,
    kind: Option<String>,
    metadata: Option<Value>,
    stream: TimedWordStream,
}

#[derive(Debug, Deserialize)]
struct BrowserTranscriptPlayerEvent {
    lane: Option<String>,
    kind: String,
    start_ms: u64,
    end_ms: Option<u64>,
    audio_ref: Option<Value>,
    metadata: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct BrowserTranscriptPlayerMarker {
    lane: Option<String>,
    kind: String,
    at_ms: u64,
    metadata: Option<Value>,
}

#[test]
fn browser_transcript_player_demo_json_deserializes() {
    let payload: BrowserTranscriptPlayerPayload = serde_json::from_str(include_str!(
        "../examples/browser-transcript-player/demo.json"
    ))
    .expect("demo JSON should deserialize");

    assert_eq!(
        payload.title.as_deref(),
        Some("Listenbury TimedWordStream Demo")
    );

    let audio = payload.audio.expect("demo should provide audio metadata");
    assert!(audio.url.ends_with("welcome.wav"));
    assert_eq!(audio.duration_ms, Some(2081));

    assert_eq!(
        payload.streams.len(),
        3,
        "demo should include all bundled word lanes"
    );
    assert!(
        payload
            .streams
            .iter()
            .all(|lane| lane.label.as_ref().is_some_and(|label| !label.is_empty()))
    );

    let stream_ids: Vec<u64> = payload
        .streams
        .iter()
        .map(|lane| lane.stream.id.0)
        .collect();
    assert_eq!(
        stream_ids,
        vec![1, 2, 3],
        "expected stream IDs in demo order"
    );
    assert!(
        payload
            .streams
            .iter()
            .all(|lane| !lane.stream.words.is_empty())
    );
    assert!(
        payload
            .streams
            .iter()
            .flat_map(|lane| lane.stream.words.iter())
            .any(|word| word.timing.is_none()),
        "demo should include at least one untimed word to exercise fallback timing display"
    );

    assert!(
        payload
            .events
            .iter()
            .any(|event| event.kind == "overlap_started"),
        "demo should include overlap events"
    );
    assert!(
        payload
            .events
            .iter()
            .any(|event| event.kind == "interruption_decision"),
        "demo should include interruption events"
    );
    assert!(
        payload
            .events
            .iter()
            .any(|event| event.kind == "environment_observation"),
        "demo should include environmental observation events"
    );
    assert!(
        payload
            .events
            .iter()
            .all(|event| { event.end_ms.is_none_or(|end_ms| end_ms >= event.start_ms) }),
        "event spans should have non-negative durations"
    );
    assert!(
        payload
            .events
            .iter()
            .all(|event| event.lane.as_ref().is_some_and(|lane| !lane.is_empty())),
        "event lanes should be labeled"
    );
    assert!(
        payload.events.iter().any(|event| event.metadata.is_some()),
        "at least one event should include inspectable metadata"
    );
    assert!(
        payload.events.iter().any(|event| event.audio_ref.is_some()),
        "at least one event should include a clip audio reference"
    );

    assert!(
        payload
            .markers
            .iter()
            .any(|marker| marker.kind == "playback_started"),
        "demo should include marker lanes"
    );
    assert!(
        payload
            .markers
            .iter()
            .all(|marker| marker.lane.as_ref().is_some_and(|lane| !lane.is_empty())),
        "marker lanes should be labeled"
    );
    assert!(
        payload.markers.iter().all(|marker| marker.at_ms <= 2081),
        "marker timestamps should be within demo timeline"
    );
    assert!(
        payload
            .markers
            .iter()
            .any(|marker| marker.metadata.is_some()),
        "at least one marker should include inspectable metadata"
    );

    assert!(
        payload
            .streams
            .iter()
            .all(|lane| lane.kind.is_none() && lane.metadata.is_none()),
        "word lanes should stay plain TimedWordStream lanes"
    );
    assert!(
        payload
            .events
            .iter()
            .any(|event| event.kind == "latency_marker"),
        "demo should include latency events"
    );
    assert!(
        payload.events.iter().any(|event| event.start_ms <= 1000),
        "demo should include early timeline events"
    );
    assert!(
        payload
            .events
            .iter()
            .any(|event| event.end_ms.unwrap_or(event.start_ms) >= 1700),
        "demo should include late timeline events"
    );
    assert!(
        payload
            .streams
            .iter()
            .flat_map(|lane| lane.stream.words.iter())
            .any(|word| word.timing.is_some()),
        "demo should keep measured words"
    );
    assert!(
        payload
            .streams
            .iter()
            .flat_map(|lane| lane.stream.words.iter())
            .any(|word| word.timing.is_none()),
        "demo should include untimed words to exercise fallback timing display"
    );

    assert!(
        payload
            .events
            .iter()
            .any(|event| event.lane.as_deref() == Some("Overlap")),
        "demo should expose overlap event lane"
    );
    assert!(
        payload
            .events
            .iter()
            .any(|event| event.lane.as_deref() == Some("Environmental")),
        "demo should expose environmental event lane"
    );
    assert!(
        payload
            .events
            .iter()
            .any(|event| event.lane.as_deref() == Some("Interruption")),
        "demo should expose interruption event lane"
    );
    assert!(
        payload
            .events
            .iter()
            .any(|event| event.lane.as_deref() == Some("Latency")),
        "demo should expose latency event lane"
    );
    assert!(
        payload
            .markers
            .iter()
            .any(|marker| marker.lane.as_deref() == Some("Latency")),
        "demo should expose latency marker lane"
    );
    assert!(
        payload
            .events
            .iter()
            .all(|event| event.start_ms <= 2081 && event.end_ms.unwrap_or(event.start_ms) <= 2081),
        "event timing should remain within the demo duration"
    );
    assert!(
        payload.markers.iter().all(|marker| marker.at_ms <= 2081),
        "marker timing should remain within the demo duration"
    );
    assert!(
        payload.events.len() >= 4,
        "demo should include multiple first-class events"
    );
    assert!(
        !payload.markers.is_empty(),
        "demo should include at least one marker"
    );

    assert!(
        payload
            .events
            .iter()
            .any(|event| event.lane.as_deref() == Some("Overlap") && event.end_ms.is_some()),
        "overlap lane should include span-like events"
    );
    assert!(
        payload
            .events
            .iter()
            .any(|event| event.kind == "latency_marker" && event.end_ms.is_none()),
        "latency events can be marker-like (point-in-time)"
    );
    assert!(
        payload
            .markers
            .iter()
            .any(|marker| marker.kind == "playback_started"),
        "markers should include playback milestones"
    );

    assert!(
        payload.events.iter().any(|event| {
            event.kind == "interruption_decision"
                && event
                    .metadata
                    .as_ref()
                    .and_then(|m| m.get("action"))
                    .is_some()
        }),
        "interruption events should carry action metadata"
    );
    assert!(
        payload.events.iter().any(|event| {
            event.kind == "environment_observation"
                && event
                    .metadata
                    .as_ref()
                    .and_then(|m| m.get("class"))
                    .is_some()
        }),
        "environment events should carry observation metadata"
    );

    assert!(
        payload.events.iter().any(|event| {
            event.kind == "overlap_started"
                && event
                    .metadata
                    .as_ref()
                    .and_then(|m| m.get("routing"))
                    .is_some()
        }),
        "overlap events should carry routing metadata"
    );
    assert!(
        payload
            .events
            .iter()
            .any(|event| event.kind == "latency_marker" && event.metadata.is_some()),
        "latency events should carry milestone metadata"
    );
    assert!(
        payload.markers.iter().any(|marker| marker
            .metadata
            .as_ref()
            .and_then(|m| m.get("stream_id"))
            .is_some()),
        "markers should carry inspectable metadata"
    );

    let _ = payload
        .streams
        .iter()
        .find(|lane| lane.stream.id.0 == 3)
        .expect("playback lane should be present");
}
