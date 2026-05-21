//! Golden-trace regression tests.
//!
//! Each test replays a fixture from `fixtures/traces/` and validates that the
//! resulting [`ViewerPayload`] satisfies the structural assertions in that
//! fixture's `expected_key_spans.json`.
//!
//! ## Regenerating expected viewer payloads
//!
//! If you change the viewer payload converter, regenerate the golden files with:
//!
//! ```text
//! LISTENBURY_UPDATE_GOLDEN=1 cargo test --no-default-features \
//!     --features tts-piper --test golden_traces
//! ```
//!
//! ## Adding a new fixture
//!
//! See `docs/golden-trace-fixtures.md` for step-by-step instructions.

use listenbury::trace::golden::{KeySpanAssertions, diff_viewer_payloads, replay_trace_jsonl};
use serde_json::Value;
use std::path::{Path, PathBuf};

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join("traces")
}

fn update_golden() -> bool {
    std::env::var("LISTENBURY_UPDATE_GOLDEN")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Replays a fixture and runs both the key-span assertions and, when the
/// expected payload file exists, a full snapshot comparison.
///
/// When `LISTENBURY_UPDATE_GOLDEN=1` is set the expected payload file is
/// written (or overwritten) with the current converter output instead of
/// being compared.
fn run_golden_fixture(fixture_name: &str) {
    let dir = fixtures_dir().join(fixture_name);

    let input_jsonl = std::fs::read_to_string(dir.join("input.jsonl"))
        .unwrap_or_else(|e| panic!("read {fixture_name}/input.jsonl: {e}"));
    let key_spans_json = std::fs::read_to_string(dir.join("expected_key_spans.json"))
        .unwrap_or_else(|e| panic!("read {fixture_name}/expected_key_spans.json: {e}"));

    let payload = replay_trace_jsonl(&input_jsonl)
        .unwrap_or_else(|e| panic!("{fixture_name}: replay failed: {e}"));

    let key_spans = KeySpanAssertions::from_json(&key_spans_json)
        .unwrap_or_else(|e| panic!("{fixture_name}: parse expected_key_spans.json: {e}"));

    let failures = key_spans.check_all(&payload);
    assert!(
        failures.is_empty(),
        "{fixture_name}: key-span assertions failed:\n{}",
        failures.join("\n")
    );

    let payload_path = dir.join("expected_viewer_payload.json");

    if update_golden() {
        let json = serde_json::to_string_pretty(&payload)
            .unwrap_or_else(|e| panic!("{fixture_name}: serialize payload: {e}"));
        std::fs::write(&payload_path, json)
            .unwrap_or_else(|e| panic!("write {}: {e}", payload_path.display()));
        eprintln!("[golden] updated {}", payload_path.display());
        return;
    }

    if !payload_path.exists() {
        eprintln!(
            "[golden] {}: expected_viewer_payload.json not found – \
             run with LISTENBURY_UPDATE_GOLDEN=1 to generate it",
            fixture_name
        );
        return;
    }

    let expected_json = std::fs::read_to_string(&payload_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", payload_path.display()));
    let expected_value: Value = serde_json::from_str(&expected_json)
        .unwrap_or_else(|e| panic!("{fixture_name}: parse expected_viewer_payload.json: {e}"));
    let expected_payload = serde_json::from_value(expected_value)
        .unwrap_or_else(|e| panic!("{fixture_name}: deserialise expected ViewerPayload: {e}"));

    let diffs = diff_viewer_payloads(&expected_payload, &payload);
    assert!(
        diffs.is_empty(),
        "{fixture_name}: viewer payload regression detected:\n{}",
        diffs.join("\n")
    );
}

/// Fixture: `half_duplex_clean`
///
/// A clean single-turn exchange — user speaks, Pete responds — with no overlap,
/// yield, or self-hearing suppression.
#[test]
fn half_duplex_clean_replays_correctly() {
    run_golden_fixture("half_duplex_clean");
}

/// Fixture: `user_interrupts_pete`
///
/// The user interrupts Pete mid-speech.  The viewer payload must contain
/// overlap and yield span events and an interruption marker.
#[test]
fn user_interrupts_pete_replays_correctly() {
    run_golden_fixture("user_interrupts_pete");
}

/// Fixture: `pete_self_leakage`
///
/// Pete's playback audio bleeds into the ASR input.  The self-hearing
/// suppression span must appear in the Latency lane and must be ordered
/// before the real user turn.
#[test]
fn pete_self_leakage_replays_correctly() {
    run_golden_fixture("pete_self_leakage");
}
