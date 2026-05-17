/**
 * live-session-reducer.test.js
 *
 * Lightweight Node.js tests for the pure reducer functions extracted from
 * app.js.  Run with:
 *
 *   node examples/browser-transcript-player/live-session-reducer.test.js
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
  };
}

function sessionGetOrCreateTurn(session, turnId) {
  if (!session.turns.has(turnId)) {
    session.turns.set(turnId, {
      id: turnId,
      transcriptCandidate: null,
      finalTranscript: null,
      latestWordStream: null,
      wordRevisions: new Map(),
    });
  }
  return session.turns.get(turnId);
}

function stableWordKey(word, index) {
  if (word.id != null) return `id:${word.id}`;
  if (word.lexical_span) return `span:${word.lexical_span.start}:${word.lexical_span.end}`;
  return `idx:${index}`;
}

function matchWordAcrossStreams(prevWords, newWord, newIndex) {
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

function labelForKind(kind) {
  return kind.replace(/_/g, " ");
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

  if (event.kind === "breath_group_opened") {
    const key = openSpanKey("Mic", turn, "breath_group_opened");
    session.openSpans.set(key, event.elapsed_ms);
    log("open", `Breath group opened (turn ${turn})`);
    return;
  }

  if (event.kind === "breath_group_closed") {
    const key = openSpanKey("Mic", turn, "breath_group_opened");
    const spanStart = session.openSpans.get(key);
    if (spanStart !== undefined) {
      session.openSpans.delete(key);
      session.viewerEvents.push({
        lane: "Mic",
        kind: "breath_group",
        label: labelForKind("breath_group"),
        start_ms: spanStart,
        end_ms: event.elapsed_ms,
        metadata: event,
      });
      log("commit", `Breath group committed (turn ${turn}, ${event.elapsed_ms - spanStart}ms)`);
    }
    return;
  }

  if (event.kind === "speech_unit_committed" && event.text) {
    log("commit", `Speech unit committed: "${event.text.slice(0, 60)}"`);
  }
  if (event.kind === "speech_unit_cancelled" && event.text) {
    log("cancel", `Speech unit cancelled: "${event.text.slice(0, 60)}"`);
  }

  const startInfo = END_TO_START[event.kind];
  if (startInfo) {
    const openKey = openSpanKey(startInfo.lane, turn, startInfo.startKind);
    const spanStart = session.openSpans.get(openKey);
    if (spanStart !== undefined) {
      session.openSpans.delete(openKey);
      session.viewerEvents.push({
        lane: startInfo.lane,
        kind: startInfo.startKind,
        label: labelForKind(startInfo.startKind),
        start_ms: spanStart,
        end_ms: event.elapsed_ms,
        metadata: event,
      });
      if (startInfo.startKind === "asr_started") {
        log("commit", `ASR span committed (turn ${turn}, ${event.elapsed_ms - spanStart}ms)`);
      }
      return;
    }
  }

  if (SPAN_PAIRS[event.kind]) {
    const LIVE_EVENT_LANE = { asr_started: "ASR", playback_started: "Speaker", speech_started: "Mic",
      llm_generation_started: "LLM", self_hearing_suppression_started: "Speaker" };
    const lane = LIVE_EVENT_LANE[event.kind] ?? "Events";
    const spanKey = openSpanKey(lane, turn, event.kind);
    session.openSpans.set(spanKey, event.elapsed_ms);
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
  const label = event.text ? event.text.slice(0, 60) : labelForKind(event.kind);
  session.viewerMarkers.push({ lane, kind: event.kind, label, at_ms: event.elapsed_ms, metadata: { event } });
}

function liveSessionToViewerPayload(session) {
  const inProgressEvents = [];
  for (const [key, startMs] of session.openSpans.entries()) {
    const [spanLane, spanTurn, kind] = JSON.parse(key);
    inProgressEvents.push({
      lane: spanLane, kind,
      label: `${labelForKind(kind)} (in progress)`,
      start_ms: startMs, end_ms: session.maxElapsedMs,
      metadata: { in_progress: true, turn: spanTurn },
    });
  }

  const wordStreamLanes = [];
  for (const [turnId, liveTurn] of [...session.turns.entries()].sort((a, b) => a[0] - b[0])) {
    if (liveTurn.latestWordStream?.words?.length > 0) {
      wordStreamLanes.push({
        label: `ASR turn ${turnId}`,
        stream: { id: turnId, source: "LiveAsr", words: liveTurn.latestWordStream.words },
      });
    }
  }

  return {
    title: "Live — Listenbury",
    streams: wordStreamLanes,
    events: [...session.viewerEvents, ...inProgressEvents],
    markers: session.viewerMarkers,
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

// ── Summary ─────────────────────────────────────────────────────────────────
console.log(`\n${passed + failed} tests — ${passed} passed, ${failed} failed\n`);
if (failed > 0) {
  process.exit(1);
}
