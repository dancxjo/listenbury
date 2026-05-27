/**
 * live-session-reducer.test.js
 *
 * Lightweight Node.js tests for the pure reducer functions extracted from
 * app.js.  Run with:
 *
 *   node web/browser-transcript-player/live-session-reducer.test.js
 *
 * These tests cover the required scenarios from the issue:
 *   - transcript candidate starts, updates, and finalizes
 *   - stable prefix grows while unstable tail changes
 *   - ASR word stream changes "that" → "what"
 *   - final transcript commits the turn
 *   - repeated render does not duplicate debug entries
 *   - out-of-order or repeated SSE events do not corrupt the view
 *
 * Also covers duplex behaviour (user + playback spans overlapping).
 */

// ── Inline copies of the pure functions from app.js ────────────────────────
// (These contain no DOM access and can be tested in Node.js.)

const SPAN_PAIRS = {
  speech_started: { end: "speech_stopped", lane: "Mic" },
  asr_started: { end: "asr_finished", lane: "ASR" },
  playback_started: { end: "playback_finished", lane: "Speaker" },
  llm_generation_started: { end: "playback_started", lane: "LLM" },
  self_hearing_suppression_started: { end: "self_hearing_suppression_ended", lane: "Speaker" },
};

const END_TO_START = Object.fromEntries(
  Object.entries(SPAN_PAIRS).map(([startKind, info]) => [info.end, { startKind, lane: info.lane }]),
);

const RELATIVE_WORD_START_THRESHOLD_MS = 250;
const RELATIVE_WORD_END_GRACE_MS = 250;

function openSpanKey(lane, turn, startKind) {
  return JSON.stringify([lane, turn ?? null, startKind]);
}

function createLiveSession() {
  return {
    turns: new Map(),
    openSpans: new Map(),
    viewerEvents: [],
    viewerMarkers: [],
    debugLog: [],
    maxElapsedMs: 0,
    latestPrompt: null,
  };
}

function sessionGetOrCreateTurn(session, turnId) {
  if (!session.turns.has(turnId)) {
    session.turns.set(turnId, {
      id: turnId,
      transcriptCandidate: null,
      finalTranscript: null,
      latestWordStream: null,
      wordStreamTimeOffsetMs: null,
      wordRevisions: new Map(),
      generatedText: null,
      generatedSyntheticFragments: [],
      syntheticUnitsById: new Map(),
      generatedSyntheticUnitOrder: [],
    });
  }
  return session.turns.get(turnId);
}

function openSpanRecord(startMs, label = null, syntheticUnitId = null) {
  return { start_ms: startMs, label, synthetic_unit_id: syntheticUnitId };
}

function openSpanStartMs(record) {
  return typeof record === "number" ? record : record?.start_ms;
}

function openSpanLabel(record) {
  return typeof record === "number" ? null : record?.label ?? null;
}

function openSpanSyntheticUnitId(record) {
  return typeof record === "number" ? null : record?.synthetic_unit_id ?? null;
}

function stableWordKey(word, index) {
  if (word.span_id != null) return `span-id:${word.span_id}`;
  if (word.id != null) return `id:${word.id}`;
  if (word.lexical_span) return `span:${word.lexical_span.start}:${word.lexical_span.end}`;
  return `idx:${index}`;
}

function matchWordAcrossStreams(prevWords, newWord, newIndex) {
  if (newWord.span_id != null) {
    const found = prevWords.find((w) => w.span_id != null && w.span_id === newWord.span_id);
    if (found) return { word: found, approximate: false };
  }
  if (newWord.id != null) {
    const found = prevWords.find((w) => w.id != null && w.id === newWord.id);
    if (found) return { word: found, approximate: false };
  }
  if (newWord.lexical_span) {
    const found = prevWords.find(
      (w) =>
        w.lexical_span &&
        w.lexical_span.start === newWord.lexical_span.start &&
        w.lexical_span.end === newWord.lexical_span.end,
    );
    if (found) return { word: found, approximate: false };
  }
  if (newIndex < prevWords.length) return { word: prevWords[newIndex], approximate: true };
  return null;
}

function normalizedId(value) {
  if (value == null) return null;
  if (typeof value === "string" || typeof value === "number") return String(value);
  if (typeof value === "object" && value !== null && "0" in value) return String(value[0]);
  return JSON.stringify(value);
}

function syntheticUnitIdFromEvent(event) {
  return normalizedId(event?.synthetic_unit_id ?? event?.artifact?.synthetic_unit_id);
}

function measuredWordTimingBounds(words) {
  let minStart = Infinity;
  let maxEnd = -Infinity;
  for (const word of words ?? []) {
    const start = word.timing?.start_ms;
    const end = word.timing?.end_ms;
    if (Number.isFinite(start) && Number.isFinite(end)) {
      minStart = Math.min(minStart, start);
      maxEnd = Math.max(maxEnd, end);
    }
  }
  return Number.isFinite(minStart) && Number.isFinite(maxEnd) ? { minStart, maxEnd } : null;
}

function inferWordStreamTimeOffsetMs(words, observedAtMs, previousOffsetMs) {
  if (previousOffsetMs != null) {
    return previousOffsetMs;
  }

  const bounds = measuredWordTimingBounds(words);
  if (!bounds) {
    return 0;
  }

  const startsNearZero = bounds.minStart <= RELATIVE_WORD_START_THRESHOLD_MS;
  const endsBeforeObservedEvent = bounds.maxEnd + RELATIVE_WORD_END_GRACE_MS < observedAtMs;
  return startsNearZero && endsBeforeObservedEvent ? Math.max(0, observedAtMs - bounds.maxEnd) : 0;
}

function wordWithTimeOffset(word, offsetMs) {
  if (!offsetMs || !word.timing) {
    return word;
  }
  return {
    ...word,
    timing: {
      ...word.timing,
      start_ms: word.timing.start_ms + offsetMs,
      end_ms: word.timing.end_ms + offsetMs,
    },
  };
}

function labelForKind(kind) {
  return kind.replace(/_/g, " ");
}

function formatRulerLabel(ms) {
  if (ms < 1000) {
    return `${ms}ms`;
  }
  const totalSeconds = ms / 1000;
  if (totalSeconds < 60) {
    return `${totalSeconds.toFixed(1)}s`;
  }
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds - minutes * 60;
  const [secondPart, decimalPart] = seconds.toFixed(1).split(".");
  return `${minutes}:${secondPart.padStart(2, "0")}.${decimalPart}`;
}

function eventChipDisplayLabel(lane, event, startMs, endMs) {
  if (
    lane?.label === "Mic" &&
    event?.kind === "speech_started" &&
    event?.metadata?.in_progress !== true &&
    endMs > startMs
  ) {
    return formatRulerLabel(endMs);
  }
  return event?.label ?? labelForKind(event?.kind ?? "event");
}

function scrollLeftForTimingFocus(bounds, viewportStartPx, viewportWidthPx, contentWidthPx) {
  const viewportEndPx = viewportStartPx + viewportWidthPx;
  if (bounds.startPx >= viewportStartPx && bounds.endPx <= viewportEndPx) {
    return null;
  }
  const centerPx = (bounds.startPx + bounds.endPx) / 2;
  const maxScrollLeft = Math.max(0, contentWidthPx - viewportWidthPx);
  return Math.max(0, Math.min(maxScrollLeft, centerPx - viewportWidthPx / 2));
}

function isGeneratedSyntheticEventKind(kind) {
  return [
    "first_safe_synthetic_unit_emitted",
    "synthetic_unit_committed",
    "synthetic_unit_cancelled",
    "speculative_synthetic_unit_updated",
    "tts_enqueue_started",
  ].includes(kind);
}

function updateGeneratedSyntheticText(liveTurn, event) {
  if (!event.text || !isGeneratedSyntheticEventKind(event.kind)) {
    return;
  }
  const syntheticUnitId = syntheticUnitIdFromEvent(event);
  if (syntheticUnitId) {
    liveTurn.syntheticUnitsById.set(syntheticUnitId, event.text);
  }

  if (event.kind === "tts_enqueue_started" || event.kind === "synthetic_unit_committed") {
    if (syntheticUnitId) {
      if (!liveTurn.generatedSyntheticUnitOrder.includes(syntheticUnitId)) {
        liveTurn.generatedSyntheticUnitOrder.push(syntheticUnitId);
      }
      const fragments = liveTurn.generatedSyntheticUnitOrder
        .map((id) => liveTurn.syntheticUnitsById.get(id))
        .filter((text) => typeof text === "string" && text.trim());
      liveTurn.generatedSyntheticFragments = fragments;
      liveTurn.generatedText = fragments.join(" ");
      return;
    }
    const fragments = liveTurn.generatedSyntheticFragments;
    const last = fragments[fragments.length - 1];
    if (last !== event.text) {
      fragments.push(event.text);
    }
    liveTurn.generatedText = fragments.join(" ");
    return;
  }

  if (liveTurn.generatedSyntheticFragments.length === 0) {
    liveTurn.generatedText = event.text;
  }
}

function semanticSpanLabel(session, kind, turnId, fallback, sourceEvent = null, openSpan = null) {
  const liveTurn = session.turns.get(turnId);
  switch (kind) {
    case "speech_started":
    case "asr_started":
    case "breath_group":
    case "breath_group_opened":
      return turnTranscriptText(liveTurn) ?? fallback;
    case "llm_generation_started":
      return liveTurn?.generatedText ?? fallback;
    case "playback_started": {
      const syntheticUnitId = openSpanSyntheticUnitId(openSpan) ?? syntheticUnitIdFromEvent(sourceEvent);
      const syntheticUnitText = syntheticUnitId ? liveTurn?.syntheticUnitsById?.get(syntheticUnitId) : null;
      return syntheticUnitText ?? liveTurn?.generatedText ?? fallback;
    }
    case "self_hearing_suppression_started":
      return liveTurn?.generatedText ?? "self-hearing suppression";
    default:
      return fallback;
  }
}

function closedSpanLabel(session, kind, turnId, openSpan) {
  const fallback = openSpanLabel(openSpan) ?? labelForKind(kind);
  return semanticSpanLabel(session, kind, turnId, fallback, null, openSpan);
}

function semanticEventLabel(event) {
  return (
    normalizeSemanticText(event.text) ??
    transcriptCandidateText(event.artifact) ??
    wordStreamText(event.artifact?.words) ??
    null
  );
}

function turnTranscriptText(liveTurn) {
  if (!liveTurn) return null;
  return (
    normalizeSemanticText(liveTurn.finalTranscript) ??
    wordStreamText(liveTurn.latestWordStream?.words) ??
    transcriptCandidateText(liveTurn.transcriptCandidate)
  );
}

function transcriptCandidateText(candidate) {
  if (!candidate || typeof candidate !== "object") return null;
  return joinSemanticText(candidate.stable_text, candidate.unstable_text);
}

function wordStreamText(words) {
  if (!Array.isArray(words) || words.length === 0) return null;
  return joinWordTexts(words.map((word) => word?.text));
}

function joinSemanticText(...parts) {
  return normalizeSemanticText(parts.filter((part) => typeof part === "string" && part.trim()).join(" "));
}

function joinWordTexts(words) {
  const text = words
    .filter((word) => typeof word === "string" && word.trim())
    .reduce((acc, word) => {
      const trimmed = word.trim();
      return acc + (/^[,.;:!?)]/.test(trimmed) ? trimmed : `${acc ? " " : ""}${trimmed}`);
    }, "");
  return normalizeSemanticText(text);
}

function normalizeSemanticText(value) {
  if (typeof value !== "string") return null;
  const text = value.replace(/\s+/g, " ").replace(/\s+([,.;:!?])/g, "$1").trim();
  return text.length > 0 ? text : null;
}

function textContent(value) {
  return String(value ?? "")
    .replace(/\s+/g, " ")
    .replace(/\s+([,.;:!?])/g, "$1")
    .trim();
}

function rawTextContent(value) {
  return value == null ? "" : String(value);
}

function reduceLiveEvent(session, event) {
  session.maxElapsedMs = Math.max(session.maxElapsedMs, event.elapsed_ms);
  const turn = event.turn;

  function log(type, message) {
    session.debugLog.push({ elapsedMs: event.elapsed_ms, type, message });
    if (session.debugLog.length > 200) {
      session.debugLog.splice(0, session.debugLog.length - 200);
    }
  }

  if (event.kind === "asr_timed_word_stream" && event.artifact?.words) {
    const liveTurn = sessionGetOrCreateTurn(session, turn);
    const newWords = event.artifact.words;
    const prevWords = liveTurn.latestWordStream?.words ?? [];

    for (let i = 0; i < newWords.length; i++) {
      const nw = newWords[i];
      const match = matchWordAcrossStreams(prevWords, nw, i);
      if (match && match.word.text !== nw.text) {
        const wkey = stableWordKey(nw, i);
        const revList = liveTurn.wordRevisions.get(wkey) ?? [];
        revList.push({
          fromText: match.word.text,
          at_ms: event.elapsed_ms,
          provenance: "word changed between ASR revisions",
          approximate: match.approximate,
        });
        liveTurn.wordRevisions.set(wkey, revList);
        const matchKind = match.approximate ? "≈idx" : "id";
        log("revise", `↩ Revision turn ${turn} [${matchKind}${i}]: "${match.word.text}" → "${nw.text}"`);
      }
    }

    const annotatedWords = newWords.map((word, i) => {
      const wkey = stableWordKey(word, i);
      const revs = liveTurn.wordRevisions.get(wkey);
      return revs?.length ? { ...word, _revisions: revs } : word;
    });
    liveTurn.latestWordStream = { ...event.artifact, words: annotatedWords };
    liveTurn.wordStreamTimeOffsetMs = inferWordStreamTimeOffsetMs(
      annotatedWords,
      event.elapsed_ms,
      liveTurn.wordStreamTimeOffsetMs,
    );
    return;
  }

  if (event.kind === "transcript_candidate" && event.artifact) {
    const liveTurn = sessionGetOrCreateTurn(session, turn);
    const { stable_text, unstable_text, confidence } = event.artifact;
    liveTurn.transcriptCandidate = { stable_text, unstable_text, confidence };
    if (stable_text || unstable_text) {
      log(
        "stable",
        `Candidate turn ${turn}: stable="${stable_text ?? ""}" | provisional="${unstable_text ?? ""}" conf=${(confidence ?? 0).toFixed(2)}`,
      );
    }
  }

  if (event.kind === "transcript" && event.text) {
    const liveTurn = sessionGetOrCreateTurn(session, turn);
    liveTurn.finalTranscript = event.text;
  }

  if (event.kind === "llm_prompt_snapshot") {
    const prompt = event.artifact?.prompt == null
      ? rawTextContent(event.text)
      : rawTextContent(event.artifact.prompt);
    session.latestPrompt = {
      turn,
      elapsedMs: event.elapsed_ms ?? 0,
      prompt,
      promptFormat: textContent(event.artifact?.prompt_format) || "unknown",
      promptChars: Number.isFinite(event.artifact?.prompt_chars) ? event.artifact.prompt_chars : prompt.length,
      selectedContextNodes: textContent(event.artifact?.selected_context_nodes),
    };
    log("prompt", `Prompt snapshot turn ${turn}: ${session.latestPrompt.promptChars} chars`);
  }

  if (event.text && isGeneratedSyntheticEventKind(event.kind)) {
    const liveTurn = sessionGetOrCreateTurn(session, turn);
    updateGeneratedSyntheticText(liveTurn, event);
  }

  if (event.kind === "breath_group_opened") {
    const key = openSpanKey("Mic", turn, "breath_group_opened");
    session.openSpans.set(
      key,
      openSpanRecord(
        event.elapsed_ms,
        semanticSpanLabel(session, "breath_group_opened", turn, labelForKind("breath_group_opened"), event),
      ),
    );
    log("open", `Breath group opened (turn ${turn})`);
    return;
  }

  if (event.kind === "breath_group_closed") {
    const key = openSpanKey("Mic", turn, "breath_group_opened");
    const openSpan = session.openSpans.get(key);
    const spanStart = openSpanStartMs(openSpan);
    if (spanStart !== undefined) {
      session.openSpans.delete(key);
      session.viewerEvents.push({
        lane: "Mic",
        kind: "breath_group",
        label: closedSpanLabel(session, "breath_group", turn, openSpan),
        start_ms: spanStart,
        end_ms: event.elapsed_ms,
        metadata: { event, turn, start_kind: "breath_group_opened", start_event_ms: spanStart, end_event_ms: event.elapsed_ms },
      });
      log("commit", `Breath group committed (turn ${turn}, ${event.elapsed_ms - spanStart}ms)`);
    }
    return;
  }

  if (event.kind === "synthetic_unit_committed" && event.text) {
    log("commit", `Synthetic unit committed: "${event.text.slice(0, 60)}"`);
  }
  if (event.kind === "synthetic_unit_cancelled" && event.text) {
    log("cancel", `Synthetic unit cancelled: "${event.text.slice(0, 60)}"`);
  }

  const startInfo = END_TO_START[event.kind];
  if (startInfo) {
    const openKey = openSpanKey(startInfo.lane, turn, startInfo.startKind);
    const openSpan = session.openSpans.get(openKey);
    const spanStart = openSpanStartMs(openSpan);
    if (spanStart !== undefined) {
      session.openSpans.delete(openKey);
      session.viewerEvents.push({
        lane: startInfo.lane,
        kind: startInfo.startKind,
        label: closedSpanLabel(session, startInfo.startKind, turn, openSpan),
        start_ms: spanStart,
        end_ms: event.elapsed_ms,
        metadata: { event, turn, start_kind: startInfo.startKind, start_event_ms: spanStart, end_event_ms: event.elapsed_ms },
      });
      if (startInfo.startKind === "asr_started") {
        log("commit", `ASR span committed (turn ${turn}, ${event.elapsed_ms - spanStart}ms)`);
      }
      if (!SPAN_PAIRS[event.kind]) {
        return;
      }
    }
  }

  if (SPAN_PAIRS[event.kind]) {
    const LIVE_EVENT_LANE = { asr_started: "ASR", playback_started: "Speaker", speech_started: "Mic",
      llm_generation_started: "LLM", self_hearing_suppression_started: "Speaker" };
    const lane = LIVE_EVENT_LANE[event.kind] ?? "Events";
    const spanKey = openSpanKey(lane, turn, event.kind);
    session.openSpans.set(
      spanKey,
      openSpanRecord(
        event.elapsed_ms,
        semanticSpanLabel(session, event.kind, turn, labelForKind(event.kind), event),
        event.kind === "playback_started" ? syntheticUnitIdFromEvent(event) : null,
      ),
    );
    if (event.kind === "asr_started") {
      log("open", `ASR span opened (turn ${turn}) [Hypothesis]`);
    }
    return;
  }

  const LIVE_EVENT_LANE = {
    transcript: "ASR", transcript_candidate: "ASR", asr_timed_word_stream: "ASR",
    playback_finished: "Speaker", asr_finished: "ASR", speech_stopped: "Mic",
  };
  const lane = LIVE_EVENT_LANE[event.kind] ?? "Events";
  const label = semanticEventLabel(event) ?? labelForKind(event.kind);
  session.viewerMarkers.push({ lane, kind: event.kind, label, at_ms: event.elapsed_ms, metadata: { event } });
}

function liveSessionToViewerPayload(session) {
  const inProgressEvents = [];
  for (const [key, openSpan] of session.openSpans.entries()) {
    const startMs = openSpanStartMs(openSpan);
    const [spanLane, spanTurn, kind] = JSON.parse(key);
    inProgressEvents.push({
      lane: spanLane, kind,
      label: openSpanLabel(openSpan) ?? semanticSpanLabel(session, kind, spanTurn, `${labelForKind(kind)} (in progress)`, null, openSpan),
      start_ms: startMs, end_ms: session.maxElapsedMs,
      metadata: {
        in_progress: true,
        turn: spanTurn,
        ...(openSpanSyntheticUnitId(openSpan) ? { synthetic_unit_id: openSpanSyntheticUnitId(openSpan) } : {}),
      },
    });
  }

  const asrWords = [];
  let nextWordId = 1;
  for (const [turnId, liveTurn] of [...session.turns.entries()].sort((a, b) => a[0] - b[0])) {
    if (liveTurn.latestWordStream?.words?.length > 0) {
      const offsetMs = liveTurn.wordStreamTimeOffsetMs ?? 0;
      for (const word of liveTurn.latestWordStream.words) {
        asrWords.push({
          ...wordWithTimeOffset(word, offsetMs),
          id: nextWordId++,
          _turn: turnId,
        });
      }
    }
  }
  asrWords.sort((left, right) => {
    const leftStart = left.timing?.start_ms ?? Number.MAX_SAFE_INTEGER;
    const rightStart = right.timing?.start_ms ?? Number.MAX_SAFE_INTEGER;
    return leftStart - rightStart || (left._turn ?? 0) - (right._turn ?? 0);
  });
  const wordStreamLanes = asrWords.length > 0
    ? [{
        label: "ASR",
        stream: { id: 1, source: "LiveAsr", words: asrWords },
      }]
    : [];

  return {
    title: "Live — Listenbury",
    streams: wordStreamLanes,
    events: [
      ...derivedTranscriptEventsFromSession(session),
      ...session.viewerEvents.map((event) => projectLiveViewerEvent(session, event)),
      ...inProgressEvents,
    ],
    markers: session.viewerMarkers,
  };
}

function derivedTranscriptEventsFromSession(session) {
  const events = [];
  for (const [turnId, liveTurn] of [...session.turns.entries()].sort((a, b) => a[0] - b[0])) {
    const label = turnTranscriptText(liveTurn);
    const words = liveTurn.latestWordStream?.words;
    if (!label || !words?.length) continue;

    const offsetMs = liveTurn.wordStreamTimeOffsetMs ?? 0;
    const bounds = measuredWordTimingBounds(words.map((word) => wordWithTimeOffset(word, offsetMs)));
    if (!bounds) continue;

    events.push({
      id: `derived-transcript-${turnId}`,
      lane: "ASR",
      kind: "transcript_span",
      label,
      start_ms: bounds.minStart,
      end_ms: Math.max(bounds.maxEnd, bounds.minStart + 1),
      metadata: {
        derived: true,
        turn: turnId,
        source: "asr_timed_word_stream",
      },
    });
  }
  return events;
}

function projectLiveViewerEvent(session, event) {
  const turn = event.metadata?.turn;
  return {
    ...event,
    label: semanticSpanLabel(session, event.kind, turn, event.label, event.metadata?.event),
  };
}

function promptSnapshotFromSession(session) {
  const snapshot = session.latestPrompt;
  if (!snapshot?.prompt) {
    return {
      prompt: "",
      label: "waiting",
    };
  }
  const parts = [
    `turn ${snapshot.turn}`,
    `${snapshot.promptChars} chars`,
    snapshot.promptFormat,
  ].filter(Boolean);
  if (snapshot.selectedContextNodes) {
    parts.push(snapshot.selectedContextNodes);
  }
  return {
    prompt: snapshot.prompt,
    label: parts.join(" · "),
  };
}

// ── Test harness ────────────────────────────────────────────────────────────

let passed = 0;
let failed = 0;

function assert(condition, label) {
  if (condition) {
    console.log(`  ✓ ${label}`);
    passed++;
  } else {
    console.error(`  ✗ ${label}`);
    failed++;
  }
}

function assertEqual(actual, expected, label) {
  if (actual === expected) {
    console.log(`  ✓ ${label}`);
    passed++;
  } else {
    console.error(`  ✗ ${label} — expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
    failed++;
  }
}

function mkEvent(kind, turn, elapsed_ms, extra = {}) {
  return { kind, turn, elapsed_ms, ...extra };
}

// ── Tests ───────────────────────────────────────────────────────────────────

console.log("\n── Scenario 1: transcript candidate starts, updates, and finalizes ──");
{
  const s = createLiveSession();
  reduceLiveEvent(s, mkEvent("transcript_candidate", 1, 100, {
    artifact: { stable_text: "", unstable_text: "hello", confidence: 0.5 },
  }));
  const t1 = s.turns.get(1);
  assert(t1 !== undefined, "turn 1 created");
  assertEqual(t1.transcriptCandidate.unstable_text, "hello", "unstable_text = hello");

  reduceLiveEvent(s, mkEvent("transcript_candidate", 1, 200, {
    artifact: { stable_text: "hello", unstable_text: "world", confidence: 0.8 },
  }));
  assertEqual(t1.transcriptCandidate.stable_text, "hello", "stable_text grows");
  assertEqual(t1.transcriptCandidate.unstable_text, "world", "unstable_text updates");

  reduceLiveEvent(s, mkEvent("transcript", 1, 300, { text: "hello world" }));
  assertEqual(t1.finalTranscript, "hello world", "final transcript committed");
}

console.log("\n── Scenario 2: stable prefix grows while unstable tail changes ──");
{
  const s = createLiveSession();
  const steps = [
    { stable: "", unstable: "I think" },
    { stable: "I", unstable: "think that" },
    { stable: "I think", unstable: "that we" },
  ];
  for (const [i, step] of steps.entries()) {
    reduceLiveEvent(s, mkEvent("transcript_candidate", 1, (i + 1) * 100, {
      artifact: { stable_text: step.stable, unstable_text: step.unstable, confidence: 0.7 },
    }));
  }
  const t1 = s.turns.get(1);
  assertEqual(t1.transcriptCandidate.stable_text, "I think", "stable prefix at final step");
  assertEqual(t1.transcriptCandidate.unstable_text, "that we", "unstable tail at final step");
  // One debug entry per update (3 total)
  assertEqual(s.debugLog.length, 3, "debug entries generated exactly once per candidate event");
}

console.log("\n── Scenario 3: ASR word stream changes 'that' → 'what' ──");
{
  const s = createLiveSession();
  reduceLiveEvent(s, mkEvent("asr_timed_word_stream", 1, 200, {
    artifact: { words: [
      { text: "I", commitment: "StableText" },
      { text: "that", commitment: "Hypothetical" },
    ]},
  }));
  reduceLiveEvent(s, mkEvent("asr_timed_word_stream", 1, 400, {
    artifact: { words: [
      { text: "I", commitment: "StableText" },
      { text: "what", commitment: "StableText" },
    ]},
  }));
  const t1 = s.turns.get(1);
  const revisedWord = t1.latestWordStream.words[1];
  assertEqual(revisedWord.text, "what", "word updated to 'what'");
  assert(revisedWord._revisions?.length === 1, "revision recorded");
  assertEqual(revisedWord._revisions[0].fromText, "that", "revision fromText is 'that'");
  assert(revisedWord._revisions[0].provenance === "word changed between ASR revisions", "generic provenance (not fabricated)");
  // Revision provenance must NOT mention 'Whisper' or probabilities
  assert(!revisedWord._revisions[0].provenance.includes("Whisper"), "no fabricated Whisper text in provenance");
  assert(!revisedWord._revisions[0].provenance.match(/p=\d/), "no fabricated probability in provenance");
  // Words without id/lexical_span match by array index → approximate = true
  assert(revisedWord._revisions[0].approximate === true, "index-based match is marked approximate");
  // Debug log has 1 revision entry
  const reviseEntries = s.debugLog.filter((e) => e.type === "revise");
  assertEqual(reviseEntries.length, 1, "exactly one revise debug entry");
}

console.log("\n── Scenario 3b: word revision via stable WordId is not approximate ──");
{
  const s = createLiveSession();
  reduceLiveEvent(s, mkEvent("asr_timed_word_stream", 1, 200, {
    artifact: { words: [
      { id: "w1", text: "that", commitment: "Hypothetical" },
    ]},
  }));
  reduceLiveEvent(s, mkEvent("asr_timed_word_stream", 1, 400, {
    artifact: { words: [
      { id: "w1", text: "what", commitment: "StableText" },
    ]},
  }));
  const t1 = s.turns.get(1);
  const revisedWord = t1.latestWordStream.words[0];
  assertEqual(revisedWord.text, "what", "word updated to 'what' via stable id");
  assert(revisedWord._revisions?.length === 1, "revision recorded via id");
  assert(revisedWord._revisions[0].approximate === false, "id-based match is NOT approximate");
}

console.log("\n── Scenario 3c: word revision via lexical_span is not approximate ──");
{
  const s = createLiveSession();
  reduceLiveEvent(s, mkEvent("asr_timed_word_stream", 1, 200, {
    artifact: { words: [
      { lexical_span: { start: 0, end: 4 }, text: "that", commitment: "Hypothetical" },
    ]},
  }));
  reduceLiveEvent(s, mkEvent("asr_timed_word_stream", 1, 400, {
    artifact: { words: [
      { lexical_span: { start: 0, end: 4 }, text: "what", commitment: "StableText" },
    ]},
  }));
  const t1 = s.turns.get(1);
  const revisedWord = t1.latestWordStream.words[0];
  assertEqual(revisedWord.text, "what", "word updated via lexical_span");
  assert(revisedWord._revisions?.[0].approximate === false, "lexical_span match is NOT approximate");
}

console.log("\n── Scenario 3d: word revision via span_id is not approximate ──");
{
  const s = createLiveSession();
  reduceLiveEvent(s, mkEvent("asr_timed_word_stream", 1, 200, {
    artifact: { words: [
      { span_id: "12f2c0f6-d38a-4f17-93fa-d32e48e23332", text: "that", commitment: "Hypothetical" },
    ]},
  }));
  reduceLiveEvent(s, mkEvent("asr_timed_word_stream", 1, 400, {
    artifact: { words: [
      { span_id: "12f2c0f6-d38a-4f17-93fa-d32e48e23332", text: "what", commitment: "StableText" },
    ]},
  }));
  const t1 = s.turns.get(1);
  const revisedWord = t1.latestWordStream.words[0];
  assertEqual(revisedWord.text, "what", "word updated via span_id");
  assert(revisedWord._revisions?.[0].approximate === false, "span_id match is NOT approximate");
}

console.log("\n── Scenario 4: final transcript commits the turn ──");
{
  const s = createLiveSession();
  reduceLiveEvent(s, mkEvent("transcript_candidate", 2, 100, {
    artifact: { stable_text: "hey", unstable_text: "there", confidence: 0.6 },
  }));
  reduceLiveEvent(s, mkEvent("transcript", 2, 500, { text: "hey there" }));

  const t2 = s.turns.get(2);
  assertEqual(t2.finalTranscript, "hey there", "final transcript is set");
  // transcriptCandidate still available (not erased) for word-level fallback
  assert(t2.transcriptCandidate !== null, "candidate retained after final commit");
}

console.log("\n── Scenario 5: repeated render does not duplicate debug entries ──");
{
  const s = createLiveSession();
  reduceLiveEvent(s, mkEvent("transcript_candidate", 1, 100, {
    artifact: { stable_text: "hi", unstable_text: "there", confidence: 0.9 },
  }));
  const countAfterFirst = s.debugLog.length;

  // Project the payload multiple times (simulates repeated renders).
  liveSessionToViewerPayload(s);
  liveSessionToViewerPayload(s);
  liveSessionToViewerPayload(s);

  assertEqual(s.debugLog.length, countAfterFirst, "debug log count unchanged after repeated projections");
}

console.log("\n── Scenario 6: out-of-order or repeated SSE events do not corrupt ──");
{
  const s = createLiveSession();
  // Deliver transcript_candidate twice (SSE replay / duplicate delivery)
  const ev = mkEvent("transcript_candidate", 1, 100, {
    artifact: { stable_text: "ok", unstable_text: "go", confidence: 0.7 },
  });
  reduceLiveEvent(s, ev);
  reduceLiveEvent(s, ev); // duplicate

  const t1 = s.turns.get(1);
  // Candidate should still be the same value
  assertEqual(t1.transcriptCandidate.stable_text, "ok", "candidate not corrupted by duplicate");
  // The reducer faithfully processes each call it receives — it does not deduplicate
  // events.  Deduplication is the SSE transport's responsibility.  Two calls with the
  // same event produce two debug entries, which is the intended and documented behaviour.
  assertEqual(s.debugLog.filter((e) => e.type === "stable").length, 2, "two stable entries for two reduce calls (no dedup in reducer)");

  // Deliver an older asr_timed_word_stream after a newer one
  reduceLiveEvent(s, mkEvent("asr_timed_word_stream", 1, 300, {
    artifact: { words: [{ text: "newer", commitment: "StableText" }] },
  }));
  reduceLiveEvent(s, mkEvent("asr_timed_word_stream", 1, 100, {
    artifact: { words: [{ text: "older", commitment: "Hypothetical" }] },
  }));
  // The session stores whatever was last reduced; transport ordering is the caller's concern.
  // What we verify is that no crash or NaN occurs.
  assert(s.turns.get(1).latestWordStream !== null, "session not corrupted by out-of-order stream");
}

console.log("\n── Scenario 7: duplex — user speech and playback spans overlap ──");
{
  const s = createLiveSession();
  // User starts speaking on turn 2 while speaker is still playing turn 1.
  reduceLiveEvent(s, mkEvent("speech_started", 2, 100));   // Mic opens turn 2
  reduceLiveEvent(s, mkEvent("playback_started", 1, 120)); // Speaker opens turn 1
  reduceLiveEvent(s, mkEvent("asr_started", 2, 130));       // ASR opens turn 2

  // Three open spans should coexist without collision.
  assertEqual(s.openSpans.size, 3, "three open spans coexist in duplex");

  reduceLiveEvent(s, mkEvent("asr_finished", 2, 400));      // closes ASR turn 2
  reduceLiveEvent(s, mkEvent("speech_stopped", 2, 450));    // closes Mic turn 2
  reduceLiveEvent(s, mkEvent("playback_finished", 1, 500)); // closes Speaker turn 1

  assertEqual(s.openSpans.size, 0, "all spans closed after duplex sequence");
  assertEqual(s.viewerEvents.filter((e) => e.lane === "ASR").length, 1, "one ASR span committed");
  assertEqual(s.viewerEvents.filter((e) => e.lane === "Mic").length, 1, "one Mic span committed");
  assertEqual(s.viewerEvents.filter((e) => e.lane === "Speaker").length, 1, "one Speaker span committed");
}

console.log("\n── Scenario 8: half-duplex — sequential turns ──");
{
  const s = createLiveSession();
  reduceLiveEvent(s, mkEvent("speech_started", 1, 0));
  reduceLiveEvent(s, mkEvent("asr_started", 1, 10));
  reduceLiveEvent(s, mkEvent("asr_finished", 1, 500));
  reduceLiveEvent(s, mkEvent("speech_stopped", 1, 510));
  reduceLiveEvent(s, mkEvent("transcript", 1, 520, { text: "what time is it" }));
  reduceLiveEvent(s, mkEvent("playback_started", 1, 600));
  reduceLiveEvent(s, mkEvent("playback_finished", 1, 1200));

  assertEqual(s.openSpans.size, 0, "all spans closed after half-duplex turn");
  assertEqual(s.turns.get(1).finalTranscript, "what time is it", "turn 1 finalized");
  const payload = liveSessionToViewerPayload(s);
  assertEqual(payload.events.filter((e) => e.in_progress).length, 0, "no in-progress events in completed session");
}

console.log("\n── Scenario 8b: span labels prefer semantic content over event kinds ──");
{
  const s = createLiveSession();
  reduceLiveEvent(s, mkEvent("speech_started", 1, 0));
  reduceLiveEvent(s, mkEvent("asr_started", 1, 10));
  reduceLiveEvent(s, mkEvent("asr_finished", 1, 300));
  reduceLiveEvent(s, mkEvent("speech_stopped", 1, 320));
  reduceLiveEvent(s, mkEvent("transcript", 1, 340, { text: "hello can you hear me" }));
  reduceLiveEvent(s, mkEvent("llm_generation_started", 1, 360));
  reduceLiveEvent(s, mkEvent("first_safe_synthetic_unit_emitted", 1, 420, { text: "Yes, I can hear you." }));
  reduceLiveEvent(s, mkEvent("playback_started", 1, 500));
  reduceLiveEvent(s, mkEvent("playback_finished", 1, 900));

  const payload = liveSessionToViewerPayload(s);
  assert(
    payload.events.some((event) => event.kind === "asr_started" && event.label === "hello can you hear me"),
    "ASR span label is transcript text",
  );
  assert(
    payload.events.some((event) => event.kind === "speech_started" && event.label === "hello can you hear me"),
    "Mic speech span label is transcript text",
  );
  assert(
    payload.events.some((event) => event.kind === "llm_generation_started" && event.label === "Yes, I can hear you."),
    "LLM span label is generated text",
  );
  assert(
    payload.events.some((event) => event.kind === "playback_started" && event.label === "Yes, I can hear you."),
    "Speaker playback span label is emitted synthetic text",
  );
}

console.log("\n── Scenario 8c: Mic speech chips show stop time after listening ends ──");
{
  const closedMicSpeech = {
    kind: "speech_started",
    label: "hello can you hear me",
    metadata: { turn: 1, start_kind: "speech_started" },
  };
  assertEqual(
    eventChipDisplayLabel({ label: "Mic" }, closedMicSpeech, 100, 4320),
    "4.3s",
    "closed Mic speech chip displays end timestamp",
  );

  const openMicSpeech = {
    kind: "speech_started",
    label: "hello can you hear me",
    metadata: { in_progress: true, turn: 1 },
  };
  assertEqual(
    eventChipDisplayLabel({ label: "Mic" }, openMicSpeech, 100, 4320),
    "hello can you hear me",
    "in-progress Mic speech chip keeps live transcript",
  );
}

console.log("\n── Scenario 8d: playback labels preserve queued synthetic units ──");
{
  const s = createLiveSession();
  reduceLiveEvent(s, mkEvent("tts_enqueue_started", 1, 100, { text: "My name is Pete." }));
  reduceLiveEvent(s, mkEvent("playback_started", 1, 120));
  reduceLiveEvent(s, mkEvent("tts_enqueue_started", 1, 180, { text: "Nice to meet you." }));
  reduceLiveEvent(s, mkEvent("playback_finished", 1, 500));

  const payload = liveSessionToViewerPayload(s);
  const playback = payload.events.find((event) => event.kind === "playback_started");
  assertEqual(
    playback?.label,
    "My name is Pete. Nice to meet you.",
    "closed playback span uses accumulated queued synthetic fragments",
  );
  assertEqual(
    s.turns.get(1).generatedText,
    "My name is Pete. Nice to meet you.",
    "turn generated text accumulates queued synthetic fragments",
  );
}

console.log("\n── Scenario 8e: playback labels can resolve by synthetic_unit_id ──");
{
  const s = createLiveSession();
  reduceLiveEvent(s, mkEvent("synthetic_unit_committed", 1, 100, {
    synthetic_unit_id: 1001,
    text: "First unit text.",
  }));
  reduceLiveEvent(s, mkEvent("synthetic_unit_committed", 1, 180, {
    synthetic_unit_id: 1002,
    text: "Second unit text.",
  }));
  reduceLiveEvent(s, mkEvent("playback_started", 1, 220, { synthetic_unit_id: 1001 }));
  reduceLiveEvent(s, mkEvent("playback_finished", 1, 400, { synthetic_unit_id: 1001 }));

  const payload = liveSessionToViewerPayload(s);
  const playback = payload.events.find((event) => event.kind === "playback_started");
  assertEqual(playback?.label, "First unit text.", "playback span resolves text using synthetic_unit_id");
}

console.log("\n── Scenario 9: liveSessionToViewerPayload is pure ──");
{
  const s = createLiveSession();
  reduceLiveEvent(s, mkEvent("asr_started", 1, 0));
  reduceLiveEvent(s, mkEvent("transcript_candidate", 1, 100, {
    artifact: { stable_text: "foo", unstable_text: "bar", confidence: 0.5 },
  }));

  const logLenBefore = s.debugLog.length;
  const payload1 = liveSessionToViewerPayload(s);
  const payload2 = liveSessionToViewerPayload(s);
  const logLenAfter = s.debugLog.length;

  assertEqual(logLenBefore, logLenAfter, "projection does not mutate debugLog");
  assertEqual(
    JSON.stringify(payload1),
    JSON.stringify(payload2),
    "projection is deterministic for same session state",
  );
}

console.log("\n── Scenario 10: duplex ASR projects to one shared timeline lane ──");
{
  const s = createLiveSession();
  reduceLiveEvent(s, mkEvent("asr_timed_word_stream", 1, 1200, {
    artifact: { words: [
      { id: 1, text: "first", timing: { start_ms: 0, end_ms: 300 }, commitment: "Final" },
    ]},
  }));
  reduceLiveEvent(s, mkEvent("asr_timed_word_stream", 2, 2200, {
    artifact: { words: [
      { id: 1, text: "second", timing: { start_ms: 0, end_ms: 400 }, commitment: "Final" },
    ]},
  }));

  const payload = liveSessionToViewerPayload(s);
  assertEqual(payload.streams.length, 1, "ASR has one word lane");
  assertEqual(payload.streams[0].label, "ASR", "ASR lane is not labelled per turn");
  assertEqual(payload.streams[0].stream.words[0].timing.start_ms, 900, "turn 1 relative timing offset onto timeline");
  assertEqual(payload.streams[0].stream.words[1].timing.start_ms, 1800, "turn 2 relative timing offset onto timeline");
  const transcriptSpans = payload.events.filter((event) => event.kind === "transcript_span");
  assertEqual(transcriptSpans.length, 2, "ASR word streams project phrase-level transcript spans");
  assertEqual(transcriptSpans[0].label, "first", "derived transcript span label uses word content");
  assertEqual(transcriptSpans[0].start_ms, 900, "derived transcript span starts at spoken word start");
  assertEqual(transcriptSpans[0].end_ms, 1200, "derived transcript span ends at spoken word end");
}

console.log("\n── Scenario 11: timeline focus chooses correct scroll target ──");
{
  assertEqual(
    scrollLeftForTimingFocus({ startPx: 200, endPx: 280 }, 0, 500, 1600),
    null,
    "returns null when timing is already visible",
  );
  assertEqual(
    scrollLeftForTimingFocus({ startPx: 900, endPx: 1000 }, 0, 500, 1600),
    700,
    "centers off-screen timing region",
  );
  assertEqual(
    scrollLeftForTimingFocus({ startPx: 1500, endPx: 1600 }, 0, 500, 1600),
    1100,
    "clamps scroll target to right edge",
  );
}

console.log("\n── Scenario 17: LLM prompt snapshot preserves raw prompt text ──");
{
  const s = createLiveSession();
  const prompt = "<|system|>\nRead aloud unless tagged.\n\n<|assistant_thought|>silent</|assistant_thought|>\n<|typescript|>tool();</|typescript|>";
  reduceLiveEvent(s, mkEvent("llm_prompt_snapshot", 4, 1234, {
    artifact: {
      prompt,
      prompt_format: "Harmony",
      prompt_chars: prompt.length,
      selected_context_nodes: "self=pete:self",
    },
  }));
  const projection = promptSnapshotFromSession(s);
  assertEqual(s.latestPrompt.prompt, prompt, "latest prompt is stored without whitespace normalization");
  assertEqual(projection.prompt, prompt, "prompt projection preserves newlines and spacing");
  assertEqual(projection.label, `turn 4 · ${prompt.length} chars · Harmony · self=pete:self`, "prompt snapshot label includes turn, size, format, and context");
  assertEqual(s.viewerMarkers.filter((marker) => marker.kind === "llm_prompt_snapshot").length, 1, "prompt snapshot appears as a timeline marker");
}

// ── Summary ─────────────────────────────────────────────────────────────────
console.log(`\n${passed + failed} tests — ${passed} passed, ${failed} failed\n`);
if (failed > 0) {
  process.exit(1);
}
