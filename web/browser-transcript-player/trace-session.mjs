export function isTraceSessionEnvelope(value) {
  return !!value
    && typeof value === "object"
    && !!value.metadata
    && typeof value.metadata === "object"
    && Array.isArray(value.events);
}

export function parseTraceEventsJsonl(text) {
  const events = [];
  let parseErrors = 0;
  for (const line of String(text).split("\n")) {
    if (!line.trim()) {
      continue;
    }
    try {
      events.push(JSON.parse(line));
    } catch {
      parseErrors += 1;
    }
  }
  return { events, parseErrors };
}

export function traceSessionLabel(envelope, fallback = "Recorded trace") {
  if (!isTraceSessionEnvelope(envelope)) {
    return fallback;
  }
  const runtimeCommand = envelope.metadata.runtime?.command;
  const runtimeMode = envelope.metadata.runtime?.mode;
  const sessionId = envelope.metadata.session_id;
  const parts = [runtimeCommand, runtimeMode, sessionId].filter(Boolean);
  return parts.length ? parts.join(" · ") : fallback;
}
