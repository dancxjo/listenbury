import test from "node:test";
import assert from "node:assert/strict";

import {
  isTraceSessionEnvelope,
  parseTraceEventsJsonl,
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
