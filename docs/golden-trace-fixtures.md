# Golden-Trace Fixtures

Golden-trace fixtures provide deterministic replay-based regression testing for
Listenbury's timing-sensitive runtime behaviour. They are stored under
`fixtures/traces/` and are validated automatically by the `golden_traces`
integration test suite.

---

## Overview

Each fixture is a directory under `fixtures/traces/<name>/` containing:

| File | Purpose |
|---|---|
| `input.jsonl` | Raw `LiveTraceEvent` stream; one JSON object per line |
| `expected_key_spans.json` | Structural assertions over the converted `ViewerPayload` |
| `expected_screenplay.txt` | Human-readable conversation transcript for the scenario |
| `expected_viewer_payload.json` | Full serialised `ViewerPayload` snapshot for regression comparison |

---

## Existing fixtures

| Fixture | Scenario |
|---|---|
| `half_duplex_clean` | Clean single-turn exchange — no overlap, yield, or suppression |
| `user_interrupts_pete` | User interrupts Pete mid-speech; overlap + yield events must be present |
| `pete_self_leakage` | Pete's playback audio bleeds into ASR; self-hearing suppression ordering is tested |

---

## Running the tests

```bash
cargo test --no-default-features --features tts-piper --test golden_traces
```

All three fixture tests must pass before merging any change that touches the
viewer payload converter (`src/trace/viewer_payload.rs`) or the live-trace
event format (`src/live_trace.rs`).

---

## Regenerating expected viewer payloads

When you intentionally change the converter output (e.g. adding a new field or
lane), regenerate the golden snapshot files:

```bash
LISTENBURY_UPDATE_GOLDEN=1 \
  cargo test --no-default-features --features tts-piper --test golden_traces
```

Commit the updated `expected_viewer_payload.json` files alongside your code
change.  The key-span assertions in `expected_key_spans.json` must be updated
by hand if their targeted properties change.

---

## Adding a new fixture

1. **Create the fixture directory** under `fixtures/traces/<your_name>/`.

2. **Write `input.jsonl`**.  Each line is a serialised `LiveTraceEvent`
   (see `src/live_trace.rs` for the struct definition).  The events should be
   in chronological order and must be parseable by
   `live_trace_jsonl_to_viewer_payload`.  Timestamps use monotonic `elapsed_ms`
   and Unix nanoseconds (`t_unix_ns`).

3. **Write `expected_key_spans.json`**.  This file drives the automated
   assertions.  It follows the `KeySpanAssertions` schema
   (`src/trace/golden.rs`):

   ```json
   {
     "description": "What this fixture tests",
     "assertions": [
       { "kind": "has_lane", "label": "User transcript" },
       { "kind": "has_event", "lane": "Overlap", "event_kind": "overlap" },
       { "kind": "has_marker", "lane": "Latency", "marker_kind": "first_llm_token" },
       { "kind": "no_event", "event_kind": "yield" },
       { "kind": "marker_ordering", "first_kind": "first_llm_token", "second_kind": "playback_started" },
       { "kind": "latency_budget", "marker_kind": "first_llm_token", "max_ms": 800 }
     ]
   }
   ```

   **Available assertion kinds:**

   | Kind | Fields | Description |
   |---|---|---|
   | `has_lane` | `label` | A word lane with the given label must exist |
   | `has_event` | `lane`, `event_kind` | An event of the given kind must be in the lane |
   | `has_marker` | `lane`, `marker_kind` | A marker of the given kind must be in the lane |
   | `no_event` | `event_kind` | No event with this kind may be present |
   | `no_marker` | `marker_kind` | No marker with this kind may be present |
   | `marker_ordering` | `first_kind`, `second_kind` | First marker must appear before second |
   | `event_ordering` | `first_kind`, `second_kind` | First event must start before second |
   | `latency_budget` | `marker_kind`, `max_ms` | Marker must appear within `max_ms` of session start |

4. **Write `expected_screenplay.txt`**.  A plain-text transcript using the
   format `[Speaker]  Text`:

   ```
   [User]  Hello, how are you?
   [Pete]  I am doing well, thank you!
   ```

5. **Generate `expected_viewer_payload.json`**:

   ```bash
   LISTENBURY_UPDATE_GOLDEN=1 \
     cargo test --no-default-features --features tts-piper --test golden_traces
   ```

6. **Add a test** in `tests/golden_traces.rs`:

   ```rust
   #[test]
   fn my_scenario_replays_correctly() {
       run_golden_fixture("my_scenario");
   }
   ```

7. **Run the suite** to confirm everything passes:

   ```bash
   cargo test --no-default-features --features tts-piper --test golden_traces
   ```

8. Commit all four files and the new test.

---

## Key-span vs. full payload comparison

The test runner performs **two** checks:

1. **Key-span assertions** (`expected_key_spans.json`) – targeted structural
   checks that are resilient to unrelated payload changes.  These always run.

2. **Full payload snapshot** (`expected_viewer_payload.json`) – an exact JSON
   comparison.  This catches any regression in the converter output, including
   field renames, ordering changes, or dropped data.  If
   `expected_viewer_payload.json` does not exist for a fixture the snapshot
   check is skipped with a warning.

When both checks pass, the fixture is considered green.

---

## Scenarios not requiring real hardware

Fixture traces are entirely synthetic JSONL — no microphone, model inference, or
audio hardware is needed.  This makes them safe to run in CI and on developer
machines without any model files or audio devices present.
