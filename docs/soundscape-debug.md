# Soundscape Debug View

The soundscape debug command (`dev soundscape-debug`) prints a JSON snapshot of
the active soundscape: which sources are present, what attribution hypotheses
support them, which voices were heard, where overlap was detected, and what the
voice-count estimates look like.

## Quick start

Print a built-in sample that demonstrates the full output shape:

```sh
cargo run -- dev soundscape-debug --sample --pretty
```

Expected output (IDs omitted for brevity):

```json
{
  "range": "12.000..15.000",
  "voice_count": {
    "active_now": 2,
    "recently_heard": 3,
    "known": 1,
    "unknown": 2,
    "confidence": 0.74
  },
  "sources": [
    { "label": "_PETE VOICE_",        "kind": "KnownSelfVoice", "confidence": 0.96 },
    { "label": "_UNKNOWN VOICE #1_",  "kind": "Voice",          "confidence": 0.68 }
  ],
  "overlaps": [...],
  "hypotheses": [...],
  "events": [
    {
      "label": "_UNKNOWN VOICE #1_",
      "text": "wait, what?",
      "transcript_confidence": 0.71,
      "attribution_confidence": 0.62,
      "overlapped": false
    }
  ]
}
```

## Loading a custom frame from JSON

Pass a JSON file that bundles all inputs needed for the debug view:

```sh
cargo run -- dev soundscape-debug --input my_frame.json --pretty
```

The expected input schema (all fields except `frame` and `voice_count` are
optional and default to empty):

```json
{
  "frame": {
    "range": { "start": {"millis": 12000}, "end": {"millis": 15000} },
    "sources": [
      {
        "id": "<uuid>",
        "kind": "KnownSelfVoice",
        "label": {"NamedVoice": "Pete"},
        "confidence": 0.96
      }
    ],
    "events": [],
    "mixtures": []
  },
  "voice_count": {
    "active_now": 1, "recently_heard": 1,
    "known": 1,      "unknown": 0,
    "confidence": 0.96
  },
  "hypotheses": [],
  "transcripts": []
}
```

## Output fields

| Field        | Description |
|--------------|-------------|
| `range`      | Frame time window as `"start_s..end_s"` (three-decimal seconds). |
| `voice_count`| Estimates: active voices now, recently heard, known vs unknown, and overall confidence. |
| `sources`    | Sources active in the frame — label, kind, and attribution confidence. |
| `overlaps`   | Overlap mixtures where two or more voice-like sources co-occur. Includes per-component hypotheses. Empty when no overlap was detected. |
| `hypotheses` | Raw attribution hypotheses with evidence labels and confidence values. |
| `events`     | Source-attributed transcript events with ASR and attribution confidence, and an `overlapped` flag. |

## Running from the compiled binary

```sh
./target/debug/listenbury dev soundscape-debug --sample --pretty
```

## Golden fixture

A hand-crafted sample fixture lives at
`fixtures/soundscape/sample_debug_view.json`.  The library unit test
`soundscape::debug::tests::golden_fixture_round_trips` deserialises it and
validates its structure on every `cargo test` run.
