import test from "node:test";
import assert from "node:assert/strict";

import {
  isTraceSessionEnvelope,
  parseTraceEventsJsonl,
  toLegacyLiveEvent,
  traceSessionLabel,
} from "./trace-session.mjs";

test("parseTraceEventsJsonl returns events and parse errors", () => {
  const { events, parseErrors } = parseTraceEventsJsonl(`
{"kind":"capture_started","elapsed_ms":0}
not json
{"kind":"transcript","elapsed_ms":250,"text":"hello"}
`);

  assert.equal(events.length, 2);
  assert.equal(parseErrors, 1);
  assert.equal(events[1].text, "hello");
});

test("trace session helpers identify envelopes and derive labels", () => {
  const envelope = {
    metadata: {
      session_id: "session-123",
      runtime: {
        command: "listenbury listen",
        mode: "half_duplex",
      },
    },
    events: [{ kind: "capture_started", elapsed_ms: 0 }],
  };

  assert.equal(isTraceSessionEnvelope(envelope), true);
  assert.equal(
    traceSessionLabel(envelope),
    "listenbury listen · half_duplex · session-123",
  );
  assert.equal(isTraceSessionEnvelope({ metadata: {}, events: null }), false);
});

test("canonical runtime envelope adapts to legacy live event shape", () => {
  const canonical = {
    id: "live:test:1:250:playback_started",
    session_id: null,
    timestamp: "2026-01-01T00:00:00Z",
    monotonic_ms: 250,
    source: "runtime_trace",
    kind: {
      domain: "playback",
      event: {
        kind: "playback_started",
        text: null,
        reason: null,
        artifact: { clip: "a.wav" },
      },
    },
    causality: ["turn:1"],
  };
  const legacy = toLegacyLiveEvent(canonical);
  assert.equal(legacy.kind, "playback_started");
  assert.equal(legacy.elapsed_ms, 250);
  assert.deepEqual(legacy.causality, ["turn:1"]);
  assert.deepEqual(legacy.artifact, { clip: "a.wav" });
});

test("parseTraceEventsJsonl accepts canonical runtime events", () => {
  const { events, parseErrors } = parseTraceEventsJsonl(`
{"id":"evt-1","session_id":null,"timestamp":"2026-01-01T00:00:00Z","monotonic_ms":99,"source":"runtime_trace","kind":{"domain":"asr","event":{"kind":"asr_started"}},"causality":["turn:1"]}
`);
  assert.equal(parseErrors, 0);
  assert.equal(events.length, 1);
  assert.equal(events[0].kind, "asr_started");
  assert.equal(events[0].elapsed_ms, 99);
  assert.equal(events[0].runtime_event.id, "evt-1");
});
