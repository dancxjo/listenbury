use listenbury::word::TimedWordStream;
use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Deserialize)]
struct BrowserTranscriptPlayerPayload {
    title: Option<String>,
    audio: Option<BrowserTranscriptPlayerAudio>,
    streams: Vec<BrowserTranscriptPlayerLane>,
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

    assert_eq!(payload.streams.len(), 4, "demo should include all bundled lanes");
    assert!(payload
        .streams
        .iter()
        .all(|lane| lane.label.as_ref().is_some_and(|label| !label.is_empty())));

    let stream_ids: Vec<u64> = payload.streams.iter().map(|lane| lane.stream.id.0).collect();
    assert_eq!(stream_ids, vec![1, 2, 3, 4], "expected stream IDs in demo order");
    assert!(payload
        .streams
        .iter()
        .all(|lane| !lane.stream.words.is_empty()));
    assert!(
        payload
            .streams
            .iter()
            .flat_map(|lane| lane.stream.words.iter())
            .any(|word| word.timing.is_none()),
        "demo should include at least one untimed word to exercise fallback timing display"
    );

    let scene_lane = payload
        .streams
        .iter()
        .find(|lane| lane.stream.id.0 == 4)
        .expect("scene lane should be present");
    assert_eq!(
        scene_lane.kind.as_deref(),
        Some("provisional-event-like-lane")
    );
    assert!(
        scene_lane
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.get("note"))
            .is_some(),
        "scene lane should include a caveat note"
    );
}
