use listenbury::word::TimedWordStream;
use serde::Deserialize;

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

    assert!(payload.streams.len() >= 3, "demo should include multi-lane data");
    assert!(payload
        .streams
        .iter()
        .all(|lane| lane.label.as_deref().is_some_and(|label| !label.is_empty())));
    assert!(payload
        .streams
        .iter()
        .flat_map(|lane| lane.stream.words.iter())
        .all(|word| word.timing.is_some()));
}
