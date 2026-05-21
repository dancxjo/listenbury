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
      const parsed = JSON.parse(line);
      events.push(toLegacyLiveEvent(parsed));
    } catch {
      parseErrors += 1;
    }
  }
  return { events, parseErrors };
}

function runtimeKindName(kind) {
  if (!kind || typeof kind !== "object") {
    return null;
  }
  if (kind.event && typeof kind.event.kind === "string") {
    return kind.event.kind;
  }
  return null;
}

function runtimeEventText(kind) {
  if (!kind || typeof kind !== "object") {
    return null;
  }
  if (kind.event && typeof kind.event.text === "string") {
    return kind.event.text;
  }
  return null;
}

function runtimeEventReason(kind) {
  if (!kind || typeof kind !== "object") {
    return null;
  }
  if (kind.event && typeof kind.event.reason === "string") {
    return kind.event.reason;
  }
  return null;
}

function runtimeEventArtifact(kind) {
  if (!kind || typeof kind !== "object") {
    return null;
  }
  if (kind.event && kind.event.artifact && typeof kind.event.artifact === "object") {
    return kind.event.artifact;
  }
  return null;
}

function timestampToUnixNs(timestamp) {
  if (typeof timestamp !== "string") {
    return 0;
  }
  const ms = Date.parse(timestamp);
  if (!Number.isFinite(ms)) {
    return 0;
  }
  return Math.max(0, Math.round(ms * 1_000_000));
}

export function toLegacyLiveEvent(event) {
  if (!event || typeof event !== "object") {
    return event;
  }
  if (typeof event.kind === "string") {
    return event;
  }
  const canonicalKind = runtimeKindName(event.kind);
  if (!canonicalKind) {
    return event;
  }
  return {
    turn: 0,
    kind: canonicalKind,
    source: typeof event.source === "string" ? event.source : null,
    t_unix_ns: timestampToUnixNs(event.timestamp),
    elapsed_ms: Number.isFinite(event.monotonic_ms) ? event.monotonic_ms : 0,
    normalized_elapsed_ms: Number.isFinite(event.monotonic_ms) ? event.monotonic_ms : 0,
    normalized_unix_ns: timestampToUnixNs(event.timestamp),
    text: runtimeEventText(event.kind),
    reason: runtimeEventReason(event.kind),
    artifact: runtimeEventArtifact(event.kind),
    causality: Array.isArray(event.causality) ? event.causality : [],
    runtime_event: event,
  };
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
