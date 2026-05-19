import { Fragment, h, render as preactRender } from "https://esm.sh/preact@10.26.9";

const viewer = document.getElementById("viewer");
const chromeShellRoot = document.getElementById("chrome-shell-root");
const transcriptShellRoot = document.getElementById("transcript-shell-root");
const inspectorShellRoot = document.getElementById("inspector-shell-root");
const audio = document.getElementById("audio");
const VIEWER_NAME = "WaveDeck";
const MIN_VIEW_DURATION_MS = 100;
const MIN_SELECTION_VIEW_MS = 500;
const RANGE_SELECTION_DRAG_THRESHOLD_PX = 12;
const MAX_RULER_TICKS = 200;
// How long a point-in-time marker appears "active" during playback (ms).
const MARKER_ACTIVE_DURATION_MS = 120;

// Custom timeline renderer settings
const DEFAULT_ZOOM_PX_PER_SECOND = 80;
const MIN_ZOOM_PX_PER_SECOND = 4;
const MAX_ZOOM_PX_PER_SECOND = 4000;
const ZOOM_STEP_FACTOR = 1.3;
const WHEEL_ZOOM_SENSITIVITY = 0.002;
const WHEEL_ZOOM_MIN_FACTOR = 0.5;
const WHEEL_ZOOM_MAX_FACTOR = 2;
const ZOOM_SELECTION_PADDING_FACTOR = 0.3;
const ZOOM_SELECTION_PADDING_MIN_MS = 250;
const ZOOM_LATEST_WINDOW_MS = 30_000;
const WORD_DENSITY_LABEL_MIN_PX = 22;
const WORD_DENSITY_BADGE_MIN_PX = 64;
const HOVER_PREVIEW_OFFSET_PX = 14;
const WAVEFORM_CANVAS_MAX_WIDTH_PX = 12_000;
const WAVEFORM_PEAK_BUCKETS = 2_400;

// Lane assignment for live trace event kinds.
const LIVE_EVENT_LANE = {
  capture_started: "Mic",
  listening_started: "Mic",
  speech_started: "Mic",
  speech_stopped: "Mic",
  breath_group_opened: "Mic",
  breath_group_closed: "Mic",
  auditory_observation: "Mic",
  environmental_sound: "Mic",
  self_voice_heard: "Speaker",
  overlap_detected: "Mic",
  asr_started: "ASR",
  asr_finished: "ASR",
  transcript: "ASR",
  transcript_candidate: "ASR",
  asr_timed_word_stream: "ASR",
  llm_generation_started: "LLM",
  first_llm_token: "LLM",
  llm_token: "LLM",
  llm_token_delta: "LLM",
  token_emitted: "LLM",
  first_safe_speech_unit_emitted: "LLM",
  speech_unit_committed: "LLM",
  speech_unit_cancelled: "LLM",
  speculative_speech_updated: "LLM",
  first_tts_audio_frame_available: "Speaker",
  playback_started: "Speaker",
  playback_finished: "Speaker",
  self_hearing_suppression_started: "Speaker",
  self_hearing_suppression_ended: "Speaker",
  face_expression: "Emotion",
  face_state: "Emotion",
  facial_expression: "Emotion",
  expression_changed: "Emotion",
  emotion_state: "Emotion",
  affect_state: "Emotion",
};

// Span pairing rules: maps start-event kind → { end-event kind, lane }.
// Used by both the live-session reducer and the projection function.
const SPAN_PAIRS = {
  speech_started: { end: "speech_stopped", lane: "Mic" },
  asr_started: { end: "asr_finished", lane: "ASR" },
  playback_started: { end: "playback_finished", lane: "Speaker" },
  llm_generation_started: { end: "playback_started", lane: "LLM" },
  self_hearing_suppression_started: { end: "self_hearing_suppression_ended", lane: "Speaker" },
};
// Reverse mapping: end-event kind → { startKind, lane }.
const END_TO_START = Object.fromEntries(
  Object.entries(SPAN_PAIRS).map(([startKind, info]) => [info.end, { startKind, lane: info.lane }]),
);

// Serialize an open-span map key as [lane, turn, startKind].
function openSpanKey(lane, turn, startKind) {
  return JSON.stringify([lane, turn ?? null, startKind]);
}


// Accumulated live trace events (kept for error recovery / diagnostics).
const liveEvents = [];
let liveRenderScheduled = false;
// Debounce interval for live re-renders (ms). Balances UI responsiveness vs. render cost.
const LIVE_RENDER_DEBOUNCE_MS = 80;

// Durable client-side live session model.  Each incoming SSE event is reduced
// into this model exactly once; renderers read from it without mutating it.
const liveSession = createLiveSession();

const state = {
  payload: null,
  lanes: [],
  selectedItem: null,
  maxDurationMs: 1000,
  stopAtMs: null,
  // Custom timeline renderer state
  zoomPxPerSecond: DEFAULT_ZOOM_PX_PER_SECOND,
  followLatest: false,
  dragSelection: null,
  brushSelection: null,
  waveform: { url: null, peaks: null, durationMs: 0, status: "idle" },
  suppressTimelineClick: false,
  itemTimingByKey: new Map(),   // itemKey → {startMs, endMs}
  chipElementByKey: new Map(),  // itemKey → DOM element
  playbackCursorElements: [],
};

const uiState = {
  liveMode: false,
  connectionStatusClass: "live-status-connecting",
  connectionStatusText: "connecting…",
  statusMessage: "Connecting to live event stream…",
};

const sourceLabels = {
  RecordedAudio: "Recorded audio",
  LiveAsr: "Live ASR",
  GeneratedText: "Generated text",
  SyntheticSpeech: "Synthetic speech",
};

const DEFAULT_SELECTION_MESSAGE = "Select a word or event to inspect timing and metadata.";

function RibbonToken({ token }) {
  return h(
    "span",
    {
      className: token.className,
      title: token.title,
    },
    token.text,
  );
}

function ConnectionChrome({ projection }) {
  return h(
    Fragment,
    null,
    h(
      "div",
      { id: "live-banner", className: "live-banner", hidden: !projection.liveMode },
      h("span", { className: "live-dot" }),
      h("strong", null, "Live"),
      h("span", { id: "live-event-count" }, projection.liveEventCountLabel),
      h("span", { id: "live-connection-status", className: projection.connectionStatusClass }, projection.connectionStatusText),
    ),
    h(
      "section",
      { className: "toolbar", id: "playback-toolbar" },
      h("button", { id: "jump-prev", type: "button", "aria-label": "Previous word", onClick: () => jumpSelectedWord(-1) }, "◀ Prev"),
      h("button", { id: "play-pause", type: "button", onClick: () => togglePlayback() }, projection.playPauseLabel),
      h("button", { id: "jump-next", type: "button", "aria-label": "Next word", onClick: () => jumpSelectedWord(1) }, "Next ▶"),
      h(
        "button",
        {
          id: "play-selection-clip",
          type: "button",
          disabled: !projection.canPlaySelectionClip,
          onClick: () => playSelectedClip(),
        },
        projection.playSelectionClipLabel,
      ),
    ),
    h(
      "section",
      { className: "toolbar zoom-toolbar", id: "zoom-toolbar", "aria-label": "Timeline zoom controls" },
      h("button", { id: "zoom-out", type: "button", "aria-label": "Zoom out (−)", disabled: !projection.canZoom, onClick: () => zoomTimelineOut() }, "−"),
      h("button", { id: "zoom-in", type: "button", "aria-label": "Zoom in (+)", disabled: !projection.canZoom, onClick: () => zoomTimelineIn() }, "+"),
      h("button", { id: "zoom-to-sel", type: "button", "aria-label": "Zoom to selection (F)", disabled: !projection.hasSelection, onClick: () => zoomToSelection() }, "⌗ Sel"),
      h("button", { id: "zoom-to-all", type: "button", "aria-label": "Zoom to full session (Shift+F)", onClick: () => zoomToFullSession() }, "↔ All"),
      h("button", { id: "zoom-to-latest", type: "button", "aria-label": "Zoom to latest activity", onClick: () => zoomToLatest() }, "▶ Latest"),
      h("button", {
        id: "follow-toggle",
        type: "button",
        "aria-label": "Toggle follow latest (L)",
        "aria-pressed": projection.followLatest,
        className: projection.followLatest ? "active-toggle" : "",
        onClick: () => toggleFollowLatest(),
      }, projection.followLatest ? "⬤ Follow" : "○ Follow"),
      h("button", { id: "zoom-reset", type: "button", "aria-label": "Reset zoom (0)", onClick: () => resetZoom() }, "⟳ Reset"),
    ),
    h(
      "section",
      { className: "status-bar" },
      h("strong", { id: "viewer-title" }, projection.viewerTitle),
      h("span", { id: "status-message" }, projection.statusMessage),
      h("span", { id: "playback-time" }, projection.playbackTimeLabel),
    ),
  );
}

function TranscriptRibbonPane({ projection }) {
  return h(
    "div",
    { id: "transcript-ribbon", className: "transcript-ribbon", hidden: !projection.liveMode, "aria-live": "polite", "aria-label": "Live transcript" },
    h("span", { className: "transcript-ribbon-label" }, "Transcript"),
    h(
      "span",
      { id: "transcript-ribbon-text", className: "transcript-ribbon-text" },
      projection.transcriptTokens.flatMap((token, index) =>
        index === projection.transcriptTokens.length - 1 ? [h(RibbonToken, { token })] : [h(RibbonToken, { token }), " "],
      ),
    ),
    h(
      "span",
      { id: "transcript-ribbon-hint", className: "transcript-ribbon-hint" },
      h("span", { className: "span-legend-item span-state-hypothetical" }, "Hypothesis"),
      h("span", { className: "span-legend-item span-state-stable" }, "Stable"),
      h("span", { className: "span-legend-item span-state-committed" }, "Committed"),
      h("span", { className: "span-legend-item span-state-revised" }, "Revised"),
    ),
  );
}

function InspectorPane({ projection }) {
  return h(
    Fragment,
    null,
    h("h2", null, "Inspector"),
    h(
      "div",
      { id: "selection-summary", className: "selection-summary" },
      projection.selectionBadge
        ? h("span", { className: projection.selectionBadge.className }, projection.selectionBadge.text)
        : null,
      projection.selectionSummaryParts,
      projection.selectionRevisions.length
        ? h(
            "div",
            { className: "inspector-revision-history" },
            h("strong", null, "↩ Retroactive revision"),
            projection.selectionRevisions.map((rev) =>
              h(
                "div",
                { className: "inspector-revision-entry" },
                h("span", null, `at ${rev.atMs}ms:`),
                h("del", null, rev.fromText),
                h("span", null, "→"),
                h("span", { className: "revision-new" }, rev.toText),
              ),
            ),
          )
        : null,
    ),
    h("pre", { id: "selection-json", className: "selection-json" }, projection.selectionJson),
    h(
      "details",
      { id: "span-debug-section", className: "span-debug-section", open: true },
      h("summary", { className: "span-debug-summary" }, "Span debug log"),
      h(
        "div",
        { id: "span-debug-log", className: "span-debug-log" },
        projection.debugEntries.length
          ? projection.debugEntries.map((entry) =>
              h(
                "div",
                { className: `span-debug-entry entry-${entry.type}` },
                h("span", { className: "span-debug-time" }, entry.time),
                h("span", { className: "span-debug-msg" }, entry.message),
              ),
            )
          : h("p", { className: "span-debug-empty" }, "Span events will appear here during a live session."),
      ),
    ),
  );
}

function buildShellProjection() {
  const selectionProjection = buildSelectionProjection();
  return {
    liveMode: uiState.liveMode,
    viewerTitle: state.payload?.title ?? (uiState.liveMode ? "Live — Listenbury" : "No stream loaded"),
    statusMessage: uiState.statusMessage,
    playbackTimeLabel: formatPlaybackTime(),
    playPauseLabel: audio.paused ? "Play" : "Pause",
    canPlaySelectionClip: selectionProjection.canPlaySelectionClip,
    playSelectionClipLabel: selectionProjection.playSelectionClipLabel,
    canZoom: state.lanes.length > 0,
    hasSelection: Boolean(state.selectedItem || state.brushSelection),
    followLatest: state.followLatest,
    liveEventCountLabel: `${liveEvents.length} event${liveEvents.length === 1 ? "" : "s"}`,
    connectionStatusText: uiState.connectionStatusText,
    connectionStatusClass: uiState.connectionStatusClass,
    transcriptTokens: transcriptTokensFromSession(liveSession),
    selectionBadge: selectionProjection.badge,
    selectionSummaryParts: selectionProjection.summaryParts,
    selectionRevisions: selectionProjection.revisions,
    selectionJson: selectionProjection.selectionJson,
    debugEntries: debugEntriesFromSession(liveSession),
  };
}

function renderShell() {
  const projection = buildShellProjection();
  if (chromeShellRoot) {
    preactRender(h(ConnectionChrome, { projection }), chromeShellRoot);
  }
  if (transcriptShellRoot) {
    preactRender(h(TranscriptRibbonPane, { projection }), transcriptShellRoot);
  }
  if (inspectorShellRoot) {
    preactRender(h(InspectorPane, { projection }), inspectorShellRoot);
  }
}

audio.addEventListener("timeupdate", () => {
  if (state.stopAtMs !== null && audio.currentTime * 1000 >= state.stopAtMs) {
    audio.pause();
    clearPlaybackStop();
  }
  refreshPlaybackState();
});
audio.addEventListener("play", refreshPlaybackState);
audio.addEventListener("pause", refreshPlaybackState);
audio.addEventListener("loadedmetadata", () => {
  syncMaxDurationWithAudio();
  render();
});

void bootstrap();

async function bootstrap() {
  enterLiveMode();
}

function enterLiveMode() {
  document.body.classList.add("live-mode");
  uiState.liveMode = true;
  document.title = "WaveDeck · Live";
  uiState.statusMessage = "Connecting to live event stream…";
  uiState.connectionStatusClass = "live-status-connecting";
  uiState.connectionStatusText = "connecting…";
  renderShell();

  connectLiveEvents();
}

function connectLiveEvents() {
  const source = new EventSource("/api/live-events");

  source.onopen = () => {
    uiState.connectionStatusText = "connected";
    uiState.connectionStatusClass = "live-status-connected";
    uiState.statusMessage = "Listening for live events…";
    renderShell();
  };

  source.onmessage = (event) => {
    try {
      const traceEvent = JSON.parse(event.data);
      addLiveEvent(traceEvent);
    } catch (err) {
      console.error("Failed to parse live event:", err, event.data);
    }
  };

  source.addEventListener("live-unavailable", (event) => {
    let message = "Live event stream is unavailable. Start with listen --web to stream events.";
    try {
      const payload = JSON.parse(event.data);
      if (payload.message) {
        message = payload.message;
      }
    } catch (err) {
      console.error("Failed to parse live availability event:", err, event.data);
    }
    uiState.connectionStatusText = "unavailable";
    uiState.connectionStatusClass = "live-status-error";
    uiState.statusMessage = message;
    renderShell();
    source.close();
  });

  source.onerror = () => {
    uiState.connectionStatusText = "disconnected";
    uiState.connectionStatusClass = "live-status-error";
    uiState.statusMessage = "Live event stream disconnected. Session may have ended.";
    renderShell();
    source.close();
  };
}

function addLiveEvent(traceEvent) {
  if (traceEvent && typeof traceEvent === "object" && traceEvent.received_ms == null) {
    traceEvent.received_ms = Math.round(performance.now() - liveSession.receivedOriginMs);
  }
  liveEvents.push(traceEvent);
  reduceLiveEvent(liveSession, traceEvent);

  if (!liveRenderScheduled) {
    liveRenderScheduled = true;
    setTimeout(() => {
      liveRenderScheduled = false;
      applyLiveEvents();
    }, LIVE_RENDER_DEBOUNCE_MS);
  }
}

// ── Live session model ────────────────────────────────────────────────────
//
// Architecture:
//   EventSource message
//     → parse LiveTraceEvent
//     → reduceLiveEvent(liveSession, event)   ← mutates session ONCE per event
//     → renderLiveSession(liveSession)        ← read-only projection
//
// LiveSession shape:
//   { turns: Map<turnId, LiveTurn>, openSpans, viewerEvents, viewerMarkers, debugLog, maxElapsedMs }
//
// LiveTurn shape:
//   { id, transcriptCandidate, finalTranscript, latestWordStream, wordStreamTimeOffsetMs, wordRevisions }

const RELATIVE_WORD_START_THRESHOLD_MS = 250;
const RELATIVE_WORD_END_GRACE_MS = 250;

function createLiveSession() {
  return {
    turns: new Map(),      // turnId → LiveTurn
    openSpans: new Map(),  // openSpanKey → { start_ms, label }
    viewerEvents: [],      // accumulated closed span events
    viewerMarkers: [],     // accumulated point markers
    debugLog: [],          // debug entries, generated exactly once per input event
    maxElapsedMs: 0,
    receivedOriginMs: performance.now(),
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
      wordRevisions: new Map(), // stableWordKey → [{fromText, at_ms, provenance, approximate}]
      generatedText: null,
      generatedSpeechFragments: [],
    });
  }
  return session.turns.get(turnId);
}

function openSpanRecord(startMs, label = null) {
  return { start_ms: startMs, label };
}

function openSpanStartMs(record) {
  return typeof record === "number" ? record : record?.start_ms;
}

function openSpanLabel(record) {
  return typeof record === "number" ? null : record?.label ?? null;
}

// Returns a stable string key for a word, preferring:
//   1. stable WordId  2. lexical span bounds  3. array-index fallback
function stableWordKey(word, index) {
  if (word.id != null) {
    return `id:${word.id}`;
  }
  if (word.lexical_span) {
    return `span:${word.lexical_span.start}:${word.lexical_span.end}`;
  }
  return `idx:${index}`;
}

// Find the previous word that corresponds to newWord (at newIndex in a revised stream).
// Returns { word, approximate } or null.
function matchWordAcrossStreams(prevWords, newWord, newIndex) {
  // 1. Stable WordId
  if (newWord.id != null) {
    const found = prevWords.find((w) => w.id != null && w.id === newWord.id);
    if (found) {
      return { word: found, approximate: false };
    }
  }
  // 2. Lexical span / text-offset overlap
  if (newWord.lexical_span) {
    const found = prevWords.find(
      (w) =>
        w.lexical_span &&
        w.lexical_span.start === newWord.lexical_span.start &&
        w.lexical_span.end === newWord.lexical_span.end,
    );
    if (found) {
      return { word: found, approximate: false };
    }
  }
  // 3. Array index fallback (approximate — provenance is marked accordingly)
  if (newIndex < prevWords.length) {
    return { word: prevWords[newIndex], approximate: true };
  }
  return null;
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

// Reduce one LiveTraceEvent into the session.  All state mutations happen
// here; projection functions must not mutate the session.
function reduceLiveEvent(session, event) {
  session.maxElapsedMs = Math.max(session.maxElapsedMs, event.elapsed_ms);
  const turn = event.turn;

  // Internal helper — log a debug entry into the session (not into any global).
  function log(type, message) {
    session.debugLog.push({ elapsedMs: event.elapsed_ms, type, message });
    // Cap the log at 200 entries to avoid unbounded growth.
    if (session.debugLog.length > 200) {
      session.debugLog.splice(0, session.debugLog.length - 200);
    }
  }

  // ── asr_timed_word_stream: update word model and detect retroactive revisions
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
          // Generic provenance — the UI does not fabricate model-specific reasons.
          provenance: "word changed between ASR revisions",
          approximate: match.approximate,
        });
        liveTurn.wordRevisions.set(wkey, revList);
        const matchKind = match.approximate ? "≈idx" : "id";
        log(
          "revise",
          `↩ Revision turn ${turn} [${matchKind}${i}]: "${match.word.text}" → "${nw.text}"`,
        );
      }
    }

    // Re-annotate each word with its accumulated revision history.
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

  // ── transcript_candidate: update the current candidate for this turn
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
    // Fall through to also emit as a lane marker below.
  }

  // ── transcript: commit final text for this turn
  if (event.kind === "transcript" && event.text) {
    const liveTurn = sessionGetOrCreateTurn(session, turn);
    liveTurn.finalTranscript = event.text;
    // Fall through to emit as a lane marker below.
  }

  if (event.text && isGeneratedSpeechEventKind(event.kind)) {
    const liveTurn = sessionGetOrCreateTurn(session, turn);
    updateGeneratedSpeechText(liveTurn, event);
  }

  // ── breath_group_opened → open span
  if (event.kind === "breath_group_opened") {
    const key = openSpanKey("Mic", turn, "breath_group_opened");
    session.openSpans.set(
      key,
      openSpanRecord(event.elapsed_ms, semanticSpanLabel(session, "breath_group_opened", turn, labelForKind("breath_group_opened"))),
    );
    log("open", `Breath group opened (turn ${turn})`);
    return;
  }

  // ── breath_group_closed → close span
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

  // ── speech_unit lifecycle
  if (event.kind === "speech_unit_committed" && event.text) {
    log("commit", `Speech unit committed: "${event.text.slice(0, 60)}"`);
  }
  if (event.kind === "speech_unit_cancelled" && event.text) {
    log("cancel", `Speech unit cancelled: "${event.text.slice(0, 60)}"`);
  }

  // ── Span end event → close the matching open span
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

  // ── Span start event → record open span
  if (SPAN_PAIRS[event.kind]) {
    const lane = LIVE_EVENT_LANE[event.kind] ?? "Events";
    const spanKey = openSpanKey(lane, turn, event.kind);
    session.openSpans.set(
      spanKey,
      openSpanRecord(event.elapsed_ms, semanticSpanLabel(session, event.kind, turn, labelForKind(event.kind))),
    );
    if (event.kind === "asr_started") {
      log("open", `ASR span opened (turn ${turn}) [Hypothesis]`);
    }
    return;
  }

  // ── All other events → point marker
  const lane = LIVE_EVENT_LANE[event.kind] ?? "Events";
  const label = semanticEventLabel(event) ?? labelForKind(event.kind);
  session.viewerMarkers.push({
    lane,
    kind: event.kind,
    label,
    at_ms: event.elapsed_ms,
    metadata: { event },
  });
}

// Pure projection: convert a LiveSession into a ViewerPayload.
// Must not call addSpanDebugEntry, DOM functions, or any function with
// persistent side effects.  Given the same session state it always returns
// the same payload.
function liveSessionToViewerPayload(session) {
  // Flush any still-open spans as in-progress spans up to maxElapsedMs.
  const inProgressEvents = [];
  for (const [key, openSpan] of session.openSpans.entries()) {
    const startMs = openSpanStartMs(openSpan);
    const [spanLane, spanTurn, kind] = JSON.parse(key);
    inProgressEvents.push({
      lane: spanLane,
      kind,
      label: openSpanLabel(openSpan) ?? semanticSpanLabel(session, kind, spanTurn, `${labelForKind(kind)} (in progress)`),
      start_ms: startMs,
      end_ms: session.maxElapsedMs,
      metadata: { in_progress: true, turn: spanTurn },
    });
  }

  // Live duplex ASR is one continuous timeline.  Keep reducer state per turn
  // for revision tracking, but project the latest ASR words onto one lane.
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
        stream: {
          id: 1,
          source: "LiveAsr",
          words: asrWords,
        },
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
    if (!label || !words?.length) {
      continue;
    }

    const offsetMs = liveTurn.wordStreamTimeOffsetMs ?? 0;
    const bounds = measuredWordTimingBounds(words.map((word) => wordWithTimeOffset(word, offsetMs)));
    if (!bounds) {
      continue;
    }

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
    label: semanticSpanLabel(session, event.kind, turn, event.label),
  };
}

// ── Span debug log ─────────────────────────────────────────────────────────

function debugEntriesFromSession(session) {
  return session.debugLog.slice(-40).reverse().map((entry) => ({
    type: entry.type,
    time: `${(entry.elapsedMs / 1000).toFixed(3)}s`,
    message: entry.message,
  }));
}

// ── Transcript ribbon ──────────────────────────────────────────────────────

// Render the live transcript ribbon from the durable LiveSession state.
// Both past turns (via finalTranscript + latestWordStream) and the current
// in-progress turn (via transcriptCandidate or latestWordStream) are driven
// by the session model, not by raw event lists.
function transcriptTokensFromSession(session) {
  const tokens = [];
  const sortedTurns = [...session.turns.entries()].sort((a, b) => a[0] - b[0]);

  for (const [, liveTurn] of sortedTurns) {
    if (liveTurn.finalTranscript != null) {
      // Committed turn: use word-level commitment states when available.
      const wordStream = liveTurn.latestWordStream;
      if (wordStream?.words?.length > 0) {
        for (const word of wordStream.words) {
          const commitClass = `commit-${commitmentClass(word.commitment)}`;
          tokens.push({
            className: `transcript-token ${commitClass}${word._revisions?.length ? " was-revised" : ""}`,
            text: word.text,
            title: formatRevisionTooltip(word) || null,
          });
        }
      } else {
        // Fall back to plain committed text.
        tokens.push({
          className: "transcript-token span-state-committed",
          text: liveTurn.finalTranscript,
          title: null,
        });
      }
    } else if (liveTurn.transcriptCandidate) {
      // In-progress turn: stable prefix + unstable tail from transcript_candidate.
      const { stable_text, unstable_text } = liveTurn.transcriptCandidate;
      if (stable_text) {
        tokens.push({
          className: "transcript-token span-state-stable",
          text: stable_text,
          title: null,
        });
      }
      if (unstable_text) {
        tokens.push({
          className: "transcript-token span-state-hypothetical",
          text: unstable_text,
          title: null,
        });
      }
    } else if (liveTurn.latestWordStream?.words?.length > 0) {
      // Word-stream fallback when no transcript_candidate is available.
      for (const word of liveTurn.latestWordStream.words) {
        const commitClass = `commit-${commitmentClass(word.commitment)}`;
        tokens.push({
          className: `transcript-token ${commitClass}`,
          text: word.text,
          title: null,
        });
      }
    }
  }
  return tokens;
}

// Map WordCommitment enum variant to a lowercase CSS class fragment.
function commitmentClass(commitment) {
  if (!commitment) {
    return "unknown";
  }
  // Normalize Rust PascalCase to lowercase with no separator (matches CSS class names).
  return String(commitment).toLowerCase().replace(/[^a-z]/g, "");
}

// Build a tooltip string showing all revisions for a word (oldest → newest).
function formatRevisionTooltip(word) {
  const revs = word._revisions;
  if (!revs?.length) {
    return null;
  }
  const lines = revs.map(
    (rev, i) =>
      `${i + 1}. "${rev.fromText}" → "${word.text}" — ${rev.provenance}${rev.approximate ? " (approx)" : ""}`,
  );
  return `↩ ${revs.length} revision${revs.length === 1 ? "" : "s"}:\n${lines.join("\n")}`;
}

function applyLiveEvents() {
  const payload = liveSessionToViewerPayload(liveSession);
  applyPayload(payload, { preserveSelection: true });
}

function labelForKind(kind) {
  return kind.replace(/_/g, " ");
}

function isGeneratedSpeechEventKind(kind) {
  return [
    "first_safe_speech_unit_emitted",
    "speech_unit_committed",
    "speech_unit_cancelled",
    "speculative_speech_updated",
    "tts_enqueue_started",
  ].includes(kind);
}

function updateGeneratedSpeechText(liveTurn, event) {
  if (!event.text || !isGeneratedSpeechEventKind(event.kind)) {
    return;
  }

  if (event.kind === "tts_enqueue_started" || event.kind === "speech_unit_committed") {
    const fragments = liveTurn.generatedSpeechFragments;
    const last = fragments[fragments.length - 1];
    if (last !== event.text) {
      fragments.push(event.text);
    }
    liveTurn.generatedText = fragments.join(" ");
    return;
  }

  if (liveTurn.generatedSpeechFragments.length === 0) {
    liveTurn.generatedText = event.text;
  }
}

function semanticSpanLabel(session, kind, turnId, fallback) {
  const liveTurn = session.turns.get(turnId);
  switch (kind) {
    case "speech_started":
    case "asr_started":
    case "breath_group":
    case "breath_group_opened":
      return turnTranscriptText(liveTurn) ?? fallback;
    case "llm_generation_started":
    case "playback_started":
      return liveTurn?.generatedText ?? fallback;
    case "self_hearing_suppression_started":
      return liveTurn?.generatedText ?? "self-hearing suppression";
    default:
      return fallback;
  }
}

function closedSpanLabel(session, kind, turnId, openSpan) {
  const fallback = openSpanLabel(openSpan) ?? labelForKind(kind);
  return semanticSpanLabel(session, kind, turnId, fallback);
}

function semanticEventLabel(event) {
  return (
    normalizeSemanticText(event.text) ??
    normalizeSemanticText(event.face) ??
    normalizeSemanticText(event.emotion) ??
    normalizeSemanticText(event.affect) ??
    transcriptCandidateText(event.artifact) ??
    wordStreamText(event.artifact?.words) ??
    null
  );
}

function semanticLabelFromPayloadEntry(entry, fallback) {
  return (
    normalizeSemanticText(entry.text) ??
    normalizeSemanticText(entry.metadata?.text) ??
    normalizeSemanticText(entry.metadata?.event?.text) ??
    normalizeSemanticText(entry.metadata?.face) ??
    normalizeSemanticText(entry.metadata?.event?.face) ??
    normalizeSemanticText(entry.metadata?.emotion) ??
    normalizeSemanticText(entry.metadata?.affect) ??
    transcriptCandidateText(entry.metadata?.artifact) ??
    transcriptCandidateText(entry.metadata?.event?.artifact) ??
    wordStreamText(entry.metadata?.artifact?.words) ??
    wordStreamText(entry.metadata?.event?.artifact?.words) ??
    fallback
  );
}

function turnTranscriptText(liveTurn) {
  if (!liveTurn) {
    return null;
  }
  return (
    normalizeSemanticText(liveTurn.finalTranscript) ??
    wordStreamText(liveTurn.latestWordStream?.words) ??
    transcriptCandidateText(liveTurn.transcriptCandidate)
  );
}

function transcriptCandidateText(candidate) {
  if (!candidate || typeof candidate !== "object") {
    return null;
  }
  return joinSemanticText(candidate.stable_text, candidate.unstable_text);
}

function wordStreamText(words) {
  if (!Array.isArray(words) || words.length === 0) {
    return null;
  }
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
  if (typeof value !== "string") {
    return null;
  }
  const text = value.replace(/\s+/g, " ").replace(/\s+([,.;:!?])/g, "$1").trim();
  return text.length > 0 ? text : null;
}

function togglePlayback() {
  if (!audio.src) {
    uiState.statusMessage = "No audio source loaded.";
    renderShell();
    return;
  }

  if (audio.paused) {
    void audio.play();
  } else {
    audio.pause();
  }
}

function jumpSelectedWord(delta) {
  const words = flattenWords();
  if (!words.length) {
    return;
  }

  let index = 0;
  if (state.selectedItem?.type === "word") {
    const selectedKey = itemKey(state.selectedItem.type, state.selectedItem.laneIndex, state.selectedItem.itemIndex);
    index = words.findIndex((word) => itemKey("word", word.laneIndex, word.wordIndex) === selectedKey);
    index = index === -1 ? 0 : Math.min(words.length - 1, Math.max(0, index + delta));
  } else if (delta > 0) {
    index = 0;
  } else {
    index = words.length - 1;
  }

  selectWord(words[index].laneIndex, words[index].wordIndex, true);
}

function applyPayload(rawPayload, options = {}) {
  const previousSelection = options.preserveSelection ? state.selectedItem : null;
  const normalized = normalizePayload(rawPayload);

  state.payload = normalized;
  state.lanes = buildLanes(normalized);
  state.selectedItem = validSelection(previousSelection) ? previousSelection : firstItemSelection();
  clearPlaybackStop();
  configureAudio(normalized.audio);
  syncMaxDurationWithAudio();
  render();
}

function buildLanes(normalizedPayload) {
  const wordLanes = normalizedPayload.streams.map((lane) => normalizeWordLane(lane));
  const eventLanes = normalizeEventLanes(normalizedPayload.events);

  return [...wordLanes, ...eventLanes].map((lane, laneIndex) => {
    if (lane.type === "word") {
      return {
        ...lane,
        words: lane.words.map((word, wordIndex) => ({ ...word, laneIndex, wordIndex })),
      };
    }
    return {
      ...lane,
      events: lane.events.map((event, eventIndex) => ({ ...event, laneIndex, eventIndex })),
    };
  });
}

function normalizePayload(rawPayload) {
  if (
    rawPayload &&
    (Array.isArray(rawPayload.streams) || Array.isArray(rawPayload.events) || Array.isArray(rawPayload.markers))
  ) {
    const streams = Array.isArray(rawPayload.streams)
      ? rawPayload.streams.map((entry, index) => {
          if (isTimedWordStream(entry)) {
            return { label: defaultLaneLabel(entry, index), stream: entry };
          }
          if (entry?.stream && isTimedWordStream(entry.stream)) {
            return { label: entry.label ?? defaultLaneLabel(entry.stream, index), stream: entry.stream };
          }
          throw new Error(`stream entry ${index} is not a TimedWordStream`);
        })
      : [];

    return {
      title: rawPayload.title ?? VIEWER_NAME,
      audio: rawPayload.audio ?? null,
      streams,
      events: normalizeEvents(rawPayload.events ?? [], rawPayload.markers ?? []),
    };
  }

  throw new Error("payload must be an object with streams/events");
}

function normalizeWordLane(lane) {
  const totalWords = lane.stream.words.length || 1;
  return {
    ...lane,
    type: "word",
    words: lane.stream.words.map((word, wordIndex) => {
      const hasMeasuredTiming = Boolean(word.timing);
      const timing = hasMeasuredTiming
        ? word.timing
        : fallbackTiming(totalWords, wordIndex, inferPayloadDuration(lane.stream));
      return {
        ...word,
        resolvedTiming: timing,
        timingResolution: hasMeasuredTiming ? "word.timing" : "fallback-layout",
        timingSourceDetail: describeTimingSource(word, hasMeasuredTiming),
      };
    }),
  };
}

function normalizeEvents(events, markers) {
  const normalizedEvents = [];

  events.forEach((entry, index) => {
    if (!entry || typeof entry !== "object") {
      throw new Error(`event entry ${index} must be an object`);
    }

    const startMs = coerceMs(entry.start_ms, `event ${index} start_ms`);
    const rawEndMs = entry.end_ms ?? startMs;
    const endMs = Math.max(startMs, coerceMs(rawEndMs, `event ${index} end_ms`));
    const audioRef = normalizeEventAudioRef(entry, startMs, endMs);
    const clocks = eventClocksFromEntry(entry, startMs, endMs, audioRef);

    normalizedEvents.push({
      id: entry.id ?? `event-${index + 1}`,
      lane: semanticLaneForEvent(entry.lane, entry.kind ?? "event", entry.metadata),
      kind: entry.kind ?? "event",
      label: semanticLabelFromPayloadEntry(entry, entry.label ?? entry.kind ?? `Event ${index + 1}`),
      start_ms: startMs,
      end_ms: endMs,
      metadata: entry.metadata ?? null,
      audio_ref: audioRef,
      clocks,
      style: endMs > startMs ? "span" : "marker",
      original: entry,
    });
  });

  markers.forEach((entry, index) => {
    if (!entry || typeof entry !== "object") {
      throw new Error(`marker entry ${index} must be an object`);
    }

    const atMs = coerceMs(entry.at_ms ?? entry.start_ms, `marker ${index} at_ms`);
    const audioRef = normalizeEventAudioRef(entry, atMs, atMs);
    const clocks = eventClocksFromEntry(entry, atMs, atMs, audioRef);

    normalizedEvents.push({
      id: entry.id ?? `marker-${index + 1}`,
      lane: semanticLaneForEvent(entry.lane ?? "Markers", entry.kind ?? "marker", entry.metadata),
      kind: entry.kind ?? "marker",
      label: semanticLabelFromPayloadEntry(entry, entry.label ?? entry.kind ?? `Marker ${index + 1}`),
      start_ms: atMs,
      end_ms: atMs,
      metadata: entry.metadata ?? null,
      audio_ref: audioRef,
      clocks,
      style: "marker",
      original: entry,
    });
  });

  return normalizedEvents;
}

function normalizeEventLanes(events) {
  const grouped = new Map();

  events.forEach((event) => {
    const laneLabel = event.lane ?? "Events";
    if (!grouped.has(laneLabel)) {
      grouped.set(laneLabel, []);
    }
    grouped.get(laneLabel).push(event);
  });

  return [...grouped.entries()].map(([label, laneEvents]) => ({
    type: "event",
    label,
    events: laneEvents.sort((left, right) => left.start_ms - right.start_ms),
  }));
}

function semanticLaneForEvent(explicitLane, kind, metadata) {
  const normalizedKind = String(kind ?? "");
  if (isEmotionKind(normalizedKind) || metadata?.face || metadata?.emotion || metadata?.affect) {
    return "Emotion";
  }
  if (normalizedKind.includes("token") && explicitLane === "Markers") {
    return "LLM";
  }
  return explicitLane ?? "Events";
}

function inferPayloadDuration(stream) {
  const timedEnd = Math.max(0, ...stream.words.map((word) => word.timing?.end_ms ?? 0));
  return timedEnd || 1000;
}

function configureAudio(audioConfig) {
  if (!audioConfig?.url) {
    state.waveform = { url: null, peaks: null, durationMs: 0, status: "idle" };
    return;
  }

  audio.src = audioConfig.url;
  void loadWaveform(audioConfig.url);
}

async function loadWaveform(url) {
  if (!url || state.waveform.url === url) {
    return;
  }
  state.waveform = { url, peaks: null, durationMs: 0, status: "loading" };
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const buffer = await response.arrayBuffer();
    const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextCtor) {
      throw new Error("Web Audio API unavailable");
    }
    const context = new AudioContextCtor();
    const audioBuffer = await context.decodeAudioData(buffer.slice(0));
    await context.close?.();
    state.waveform = {
      url,
      peaks: waveformPeaksFromAudioBuffer(audioBuffer),
      durationMs: Math.round(audioBuffer.duration * 1000),
      status: "ready",
    };
    renderCustomTimeline();
  } catch (err) {
    console.warn("Unable to build waveform overlay:", err);
    state.waveform = { url, peaks: null, durationMs: 0, status: "error" };
  }
}

function waveformPeaksFromAudioBuffer(audioBuffer) {
  const channelCount = audioBuffer.numberOfChannels;
  const sampleCount = audioBuffer.length;
  const bucketCount = Math.min(WAVEFORM_PEAK_BUCKETS, Math.max(1, sampleCount));
  const samplesPerBucket = Math.max(1, Math.floor(sampleCount / bucketCount));
  const peaks = [];

  for (let bucket = 0; bucket < bucketCount; bucket++) {
    const start = bucket * samplesPerBucket;
    const end = bucket === bucketCount - 1 ? sampleCount : Math.min(sampleCount, start + samplesPerBucket);
    let peak = 0;
    for (let channel = 0; channel < channelCount; channel++) {
      const data = audioBuffer.getChannelData(channel);
      for (let i = start; i < end; i++) {
        peak = Math.max(peak, Math.abs(data[i] ?? 0));
      }
    }
    peaks.push(Math.min(1, peak));
  }
  return peaks;
}

function syncMaxDurationWithAudio() {
  const fromPayload = state.payload?.audio?.duration_ms ?? 0;
  const fromStreams = Math.max(
    0,
    ...state.lanes
      .filter((lane) => lane.type === "word")
      .flatMap((lane) => lane.words.map((word) => word.resolvedTiming.end_ms)),
  );
  const fromEvents = Math.max(
    0,
    ...state.lanes
      .filter((lane) => lane.type === "event")
      .flatMap((lane) => lane.events.map((event) => event.end_ms)),
  );
  const fromAudio = Number.isFinite(audio.duration) ? Math.round(audio.duration * 1000) : 0;
  state.maxDurationMs = Math.max(fromPayload, fromStreams, fromEvents, fromAudio, 1000);
}

function render() {
  if (!state.lanes.length) {
    viewer.className = "viewer empty";
    state.chipElementByKey = new Map();
    state.itemTimingByKey = new Map();
    viewer.innerHTML = "<p>No streams or events loaded yet.</p>";
    renderSelection();
    renderShell();
    return;
  }

  viewer.className = "viewer";
  renderCustomTimeline();
  renderSelection();
  renderShell();
}

// ── Custom timeline renderer ───────────────────────────────────────────────

let _programmaticScroll = false;

function pxPerMs() {
  return state.zoomPxPerSecond / 1000;
}

function pxForMs(ms) {
  return ms * pxPerMs();
}

function msForPx(px) {
  return px / pxPerMs();
}

function clampZoom(pxPerSec) {
  return Math.max(MIN_ZOOM_PX_PER_SECOND, Math.min(MAX_ZOOM_PX_PER_SECOND, pxPerSec));
}

function getScrollContainer() {
  return document.getElementById("timeline-tracks-col");
}

function getTrackContentWidth() {
  const col = getScrollContainer();
  const viewWidth = col ? col.clientWidth : 600;
  return Math.max(viewWidth, Math.ceil(state.maxDurationMs * pxPerMs()));
}

function getScrollViewport() {
  const col = getScrollContainer();
  if (!col) {
    return { startMs: 0, endMs: state.maxDurationMs, durationMs: Math.max(MIN_VIEW_DURATION_MS, state.maxDurationMs) };
  }
  const startMs = Math.max(0, msForPx(col.scrollLeft));
  const endMs = Math.min(state.maxDurationMs, msForPx(col.scrollLeft + col.clientWidth));
  const durationMs = Math.max(MIN_VIEW_DURATION_MS, endMs - startMs);
  return { startMs, endMs, durationMs };
}

function ensureCustomTimeline() {
  if (document.getElementById("timeline-tracks-col")) return;

  viewer.replaceChildren();

  const host = document.createElement("div");
  host.className = "timeline-host";
  host.id = "timeline-host";

  const labelsCol = document.createElement("div");
  labelsCol.className = "timeline-labels-col";
  labelsCol.id = "timeline-labels-col";

  const tracksCol = document.createElement("div");
  tracksCol.className = "timeline-tracks-col";
  tracksCol.id = "timeline-tracks-col";

  const scrollContent = document.createElement("div");
  scrollContent.className = "timeline-scroll-content";
  scrollContent.id = "timeline-scroll-content";
  tracksCol.append(scrollContent);

  host.append(labelsCol, tracksCol);
  viewer.append(host);

  tracksCol.addEventListener("scroll", onTracksScroll, { passive: true });
  tracksCol.addEventListener("wheel", onTimelineWheel, { passive: false });
  tracksCol.addEventListener("click", onTimelineClick);
  tracksCol.addEventListener("pointerdown", startTimeRangeSelection);
  tracksCol.addEventListener("pointermove", moveTimeRangeSelection);
  tracksCol.addEventListener("pointerup", finishTimeRangeSelection);
  tracksCol.addEventListener("pointercancel", cancelTimeRangeSelection);
  tracksCol.addEventListener("pointerover", showTimelineHoverPreview);
  tracksCol.addEventListener("pointermove", positionTimelineHoverPreview);
  tracksCol.addEventListener("pointerout", hideTimelineHoverPreview);
}

function renderCustomTimeline() {
  ensureCustomTimeline();

  const trackContentWidth = getTrackContentWidth();
  const labelsCol = document.getElementById("timeline-labels-col");
  const scrollContent = document.getElementById("timeline-scroll-content");
  if (!labelsCol || !scrollContent) return;

  labelsCol.innerHTML = "";
  scrollContent.innerHTML = "";
  scrollContent.style.width = `${trackContentWidth}px`;

  state.chipElementByKey = new Map();
  state.itemTimingByKey = new Map();
  state.playbackCursorElements = [];

  const nowMs = currentPlaybackTimeMs();

  // Ruler label (for the labels column)
  const rulerLabelEl = document.createElement("div");
  rulerLabelEl.className = "lane-ruler-label";
  labelsCol.append(rulerLabelEl);

  // Ruler track (for the tracks column) — ticks based on full session
  const rulerEl = document.createElement("div");
  rulerEl.className = "timeline-ruler";
  const col = getScrollContainer();
  const vpMs = col ? Math.max(MIN_VIEW_DURATION_MS, msForPx(col.clientWidth)) : state.maxDurationMs;
  const rulerTicks = buildRulerTicks(state.maxDurationMs, vpMs);
  rulerTicks.forEach((ms) => {
    const tick = document.createElement("span");
    tick.className = "ruler-tick";
    tick.style.left = `${pxForMs(ms)}px`;
    const label = document.createElement("span");
    label.className = "ruler-label";
    label.style.left = `${pxForMs(ms)}px`;
    label.textContent = formatRulerLabel(ms);
    rulerEl.append(tick, label);
  });
  scrollContent.append(rulerEl);

  // Lane rows
  state.lanes.forEach((lane, laneIndex) => {
    // Label entry
    const labelEntryEl = document.createElement("div");
    labelEntryEl.className = `lane-label-entry${lane.type === "event" ? " event-lane" : ""}`;
    const laneHeader = document.createElement("div");
    laneHeader.className = ["lane-header", `lane-${classToken(lane.label)}`].join(" ");
    const h2El = document.createElement("h2");
    h2El.textContent = lane.label;
    laneHeader.append(h2El);
    const metaEl = document.createElement("div");
    metaEl.className = "lane-meta";
    if (lane.type === "word") {
      metaEl.textContent = `${lane.words.length} word${lane.words.length === 1 ? "" : "s"}`;
    } else {
      metaEl.textContent = `${lane.events.length} event${lane.events.length === 1 ? "" : "s"}`;
    }
    laneHeader.append(metaEl);
    labelEntryEl.append(laneHeader);
    labelsCol.append(labelEntryEl);

    // Track entry
    const trackEl = document.createElement("div");
    trackEl.className = [
      "lane-track",
      lane.type === "event" ? "event-track" : "",
      `lane-${classToken(lane.label)}`,
    ]
      .filter(Boolean)
      .join(" ");
    trackEl.dataset.laneIndex = String(laneIndex);

    if (lane.type === "word") {
      appendWaveformOverlay(trackEl, trackContentWidth);
    }

    // Time-range selection overlay
    const selOverlay = document.createElement("div");
    selOverlay.className = "time-range-selection";
    selOverlay.setAttribute("aria-hidden", "true");
    selOverlay.hidden = true;
    const selLabel = document.createElement("span");
    selLabel.className = "time-range-selection-label";
    selOverlay.append(selLabel);
    trackEl.append(selOverlay);

    const focusOverlay = document.createElement("div");
    focusOverlay.className = "selection-focus-overlay";
    focusOverlay.setAttribute("aria-hidden", "true");
    focusOverlay.hidden = true;
    trackEl.append(focusOverlay);

    const playbackCursor = document.createElement("div");
    playbackCursor.className = "playback-cursor";
    playbackCursor.setAttribute("aria-hidden", "true");
    playbackCursor.hidden = true;
    trackEl.append(playbackCursor);
    state.playbackCursorElements.push(playbackCursor);

    if (lane.type === "word") {
      lane.words.forEach((word, wordIndex) => {
        const key = itemKey("word", laneIndex, wordIndex);
        const startMs = word.resolvedTiming.start_ms;
        const endMs = Math.max(word.resolvedTiming.end_ms, startMs + 1);
        state.itemTimingByKey.set(key, { startMs, endMs });

        const isSelected =
          state.selectedItem?.type === "word" &&
          state.selectedItem.laneIndex === laneIndex &&
          state.selectedItem.itemIndex === wordIndex;
        const isActive = nowMs >= startMs && nowMs <= endMs;
        const widthPx = Math.max(2, pxForMs(endMs - startMs));
        const prosodyLevel = prosodyLevelForWord(word);
        const densityClass =
          widthPx < WORD_DENSITY_LABEL_MIN_PX
            ? "density-word"
            : widthPx < WORD_DENSITY_BADGE_MIN_PX
              ? "compact-word"
              : "detailed-word";

        const baseClass = [
          "timeline-chip word-chip",
          `lane-${classToken(lane.label)}`,
          `source-${classToken(lane.stream.source)}`,
          densityClass,
          word.timingResolution === "fallback-layout" ? "fallback-timing" : "",
          prosodyLevel !== null ? "has-prosody" : "",
          `commit-${commitmentClass(word.commitment)}`,
          word._revisions?.length ? "was-revised" : "",
        ]
          .filter(Boolean)
          .join(" ");

        const chip = document.createElement("div");
        chip.className = [baseClass, isActive ? "active" : "", isSelected ? "selected" : ""]
          .filter(Boolean)
          .join(" ");
        chip.dataset.baseClass = baseClass;
        chip.dataset.itemKey = key;
        chip.dataset.laneIndex = String(laneIndex);
        chip.dataset.itemIndex = String(wordIndex);
        chip.dataset.itemType = "word";
        chip.style.left = `${pxForMs(startMs)}px`;
        chip.style.width = `${widthPx}px`;
        if (prosodyLevel !== null) {
          chip.style.setProperty("--prosody-level", String(prosodyLevel));
        }
        chip.title = `${lane.label} · ${word.text} (${startMs}–${endMs} ms) · ${word.timingSourceDetail}`;
        chip.setAttribute("aria-label", `${lane.label}: ${word.text}`);
        if (densityClass === "density-word") {
          chip.append(createTokenDensityBar(word));
        } else {
          appendWordChipContent(chip, word, densityClass === "detailed-word");
        }
        appendAlignmentHandles(chip, word);
        trackEl.append(chip);
        state.chipElementByKey.set(key, chip);
      });
    } else {
      lane.events.forEach((event, eventIndex) => {
        const key = itemKey("event", laneIndex, eventIndex);
        const isMarker = event.style === "marker" || event.start_ms === event.end_ms;
        const startMs = event.start_ms;
        const endMs = Math.max(event.end_ms, startMs + 1);
        const visualBounds = isMarker ? { startMs, endMs } : eventVisualBounds(event, startMs, endMs);
        const visualStartMs = visualBounds.startMs;
        const visualEndMs = Math.max(visualBounds.endMs, visualStartMs + 1);
        state.itemTimingByKey.set(key, { startMs: visualStartMs, endMs: visualEndMs });

        const isSelected =
          state.selectedItem?.type === "event" &&
          state.selectedItem.laneIndex === laneIndex &&
          state.selectedItem.itemIndex === eventIndex;
        const activeEndMs = isMarker ? startMs + MARKER_ACTIVE_DURATION_MS : endMs;
        const isActive = nowMs >= startMs && nowMs <= activeEndMs;

        const baseClass = [
          "timeline-chip event-chip",
          `lane-${classToken(lane.label)}`,
          event.style,
          `kind-${classToken(event.kind)}`,
          isOverlapKind(event.kind) ? "overlap-region" : "",
          isInterruptionKind(event.kind) ? "interruption-region" : "",
          isTokenKind(event.kind) ? "token-region" : "",
          isEmotionKind(event.kind) ? "emotion-region" : "",
          eventHasClockDivergence(event) ? "has-clock-divergence" : "",
        ]
          .filter(Boolean)
          .join(" ");

        const chip = document.createElement("div");
        chip.className = [baseClass, isActive ? "active" : "", isSelected ? "selected" : ""]
          .filter(Boolean)
          .join(" ");
        chip.dataset.baseClass = baseClass;
        chip.dataset.itemKey = key;
        chip.dataset.laneIndex = String(laneIndex);
        chip.dataset.itemIndex = String(eventIndex);
        chip.dataset.itemType = "event";
        const widthPx = isMarker ? 10 : Math.max(12, pxForMs(visualEndMs - visualStartMs));
        chip.style.left = `${pxForMs(visualStartMs)}px`;
        chip.style.width = `${widthPx}px`;
        const displayLabel = eventChipDisplayLabel(lane, event, startMs, endMs);
        chip.title = [
          `${lane.label} · ${event.label} · ${event.kind}`,
          ...eventClockLines(event),
        ].join("\n");
        if (isMarker) {
          chip.setAttribute("aria-label", `${lane.label}: ${displayLabel}`);
          const markerPin = document.createElement("span");
          markerPin.className = "marker-pin";
          markerPin.setAttribute("aria-hidden", "true");
          chip.append(markerPin);
        } else {
          chip.setAttribute("aria-label", `${lane.label}: ${displayLabel}`);
          appendEventClockOverlays(chip, event, visualStartMs, visualEndMs);
          chip.append(createChipContent(displayLabel), createChipBadges([shortKindLabel(event.kind)]));
        }
        trackEl.append(chip);
        state.chipElementByKey.set(key, chip);
      });
    }

    scrollContent.append(trackEl);
  });

  updateTimeRangeSelectionOverlays();
  updatePlaybackCursorOverlays();

  if (state.followLatest) {
    applyFollowLatest();
  }
}

function appendWaveformOverlay(trackEl, trackContentWidth) {
  if (state.waveform.status !== "ready" || !state.waveform.peaks?.length || state.waveform.durationMs <= 0) {
    return;
  }

  const canvas = document.createElement("canvas");
  canvas.className = "waveform-overlay";
  canvas.setAttribute("aria-hidden", "true");
  canvas.style.width = `${trackContentWidth}px`;
  trackEl.append(canvas);

  requestAnimationFrame(() => drawWaveformOverlay(canvas, trackContentWidth));
}

function drawWaveformOverlay(canvas, trackContentWidth) {
  const peaks = state.waveform.peaks;
  if (!peaks?.length) {
    return;
  }

  const rect = canvas.getBoundingClientRect();
  const cssWidth = Math.max(1, Math.min(trackContentWidth, WAVEFORM_CANVAS_MAX_WIDTH_PX));
  const cssHeight = Math.max(1, rect.height || 80);
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  canvas.width = Math.round(cssWidth * dpr);
  canvas.height = Math.round(cssHeight * dpr);
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return;
  }

  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, cssWidth, cssHeight);
  ctx.strokeStyle = "rgba(99, 210, 255, 0.18)";
  ctx.lineWidth = 1;
  const centerY = cssHeight / 2;
  const maxAmp = cssHeight * 0.34;
  const durationScale = state.maxDurationMs / Math.max(1, state.waveform.durationMs);

  ctx.beginPath();
  for (let x = 0; x < cssWidth; x++) {
    const timelineRatio = x / Math.max(1, cssWidth);
    const audioRatio = Math.min(1, timelineRatio * durationScale);
    const peakIndex = Math.min(peaks.length - 1, Math.floor(audioRatio * peaks.length));
    const amp = Math.max(1, peaks[peakIndex] * maxAmp);
    ctx.moveTo(x + 0.5, centerY - amp);
    ctx.lineTo(x + 0.5, centerY + amp);
  }
  ctx.stroke();
}

// Update only active/selected classes on existing chips (no DOM rebuild).
function updateChipStates() {
  const nowMs = currentPlaybackTimeMs();
  for (const [key, chip] of state.chipElementByKey.entries()) {
    const timing = state.itemTimingByKey.get(key);
    if (!timing) continue;
    const { itemType, laneIndex, itemIndex } = parseItemKey(key);
    const isMarker = timing.endMs <= timing.startMs;
    const activeEndMs = isMarker ? timing.startMs + MARKER_ACTIVE_DURATION_MS : timing.endMs;
    const isActive = nowMs >= timing.startMs && nowMs <= activeEndMs;
    const isSelected =
      state.selectedItem?.type === itemType &&
      state.selectedItem.laneIndex === laneIndex &&
      state.selectedItem.itemIndex === itemIndex;
    const baseClass = chip.dataset.baseClass ?? "";
    const newClass = [baseClass, isActive ? "active" : "", isSelected ? "selected" : ""]
      .filter(Boolean)
      .join(" ");
    if (chip.className !== newClass) {
      chip.className = newClass;
    }
  }
}

// Build ruler tick timestamps covering [0..maxDurationMs].
// Spacing is based on viewportMs so ~10 ticks appear per visible screen width.
// Capped at MAX_RULER_TICKS to avoid excessive DOM nodes for long sessions.
function buildRulerTicks(maxDurationMs, viewportMs) {
  const safeDuration = Math.max(MIN_VIEW_DURATION_MS, viewportMs);
  const targetSegments = 10;
  // Candidate step sizes in ms: from fine-grained (25ms) to coarse (10min)
  const preferredSteps = [25, 50, 100, 250, 500, 1000, 2000, 5000, 10_000, 15_000, 30_000, 60_000, 120_000, 300_000, 600_000];
  const desiredStep = safeDuration / targetSegments;
  // Enforce a minimum step to keep total tick count under MAX_RULER_TICKS
  const minStepForMaxTicks = maxDurationMs / MAX_RULER_TICKS;
  const effectiveDesiredStep = Math.max(desiredStep, minStepForMaxTicks);
  // Candidate steps: 25ms, 50ms, 100ms, 250ms, 500ms, 1s, 2s, 5s, 10s, 15s, 30s, 1min, 2min, 5min, 10min
  const stepMs = preferredSteps.find((step) => step >= effectiveDesiredStep) ?? 600_000;

  const ticks = [];
  for (let at = 0; at <= maxDurationMs; at += stepMs) {
    ticks.push(at);
  }
  if (!ticks.length || ticks[ticks.length - 1] !== maxDurationMs) {
    ticks.push(maxDurationMs);
  }
  return [...new Set(ticks)];
}

// ── Event handlers for the custom timeline ──────────────────────────────────

function onTracksScroll() {
  if (_programmaticScroll) return;
  if (state.followLatest) {
    state.followLatest = false;
    renderShell();
  }
}

function onTimelineWheel(event) {
  if (!state.lanes.length || event.deltaY === 0) return;

  const col = getScrollContainer();
  if (!col) return;

  const rect = col.getBoundingClientRect();
  if (!rect.width) return;

  event.preventDefault();

  const xInViewport = Math.max(0, Math.min(rect.width, event.clientX - rect.left));
  const anchorMs = msForPx(col.scrollLeft + xInViewport);
  const deltaY = normalizeWheelDeltaY(event);
  const factor = Math.max(
    WHEEL_ZOOM_MIN_FACTOR,
    Math.min(WHEEL_ZOOM_MAX_FACTOR, Math.exp(-deltaY * WHEEL_ZOOM_SENSITIVITY)),
  );
  const nextZoom = clampZoom(state.zoomPxPerSecond * factor);
  if (nextZoom === state.zoomPxPerSecond) return;

  state.zoomPxPerSecond = nextZoom;
  if (state.followLatest) {
    state.followLatest = false;
  }
  renderCustomTimeline();

  _programmaticScroll = true;
  const maxScrollLeft = Math.max(0, col.scrollWidth - col.clientWidth);
  col.scrollLeft = Math.max(0, Math.min(maxScrollLeft, pxForMs(anchorMs) - xInViewport));
  requestAnimationFrame(() => {
    _programmaticScroll = false;
  });
  updateZoomControls();
}

function normalizeWheelDeltaY(event) {
  if (event.deltaMode === WheelEvent.DOM_DELTA_LINE) {
    return event.deltaY * 16;
  }
  if (event.deltaMode === WheelEvent.DOM_DELTA_PAGE) {
    return event.deltaY * window.innerHeight;
  }
  return event.deltaY;
}

function onTimelineClick(event) {
  if (state.suppressTimelineClick) return;
  const chip = event.target?.closest(".timeline-chip");
  if (!chip) return;
  const laneIndex = parseInt(chip.dataset.laneIndex, 10);
  const itemIndex = parseInt(chip.dataset.itemIndex, 10);
  const itemType = chip.dataset.itemType;
  if (itemType === "word") {
    selectWord(laneIndex, itemIndex, true);
  } else {
    selectEvent(laneIndex, itemIndex, true);
  }
}

function showTimelineHoverPreview(event) {
  const chip = event.target?.closest?.(".timeline-chip");
  if (!chip) return;
  const preview = ensureTimelineHoverPreview();
  renderTimelineHoverPreview(preview, chip);
  preview.hidden = false;
  positionTimelineHoverPreview(event);
}

function positionTimelineHoverPreview(event) {
  const preview = document.getElementById("timeline-hover-preview");
  if (!preview || preview.hidden) return;
  const margin = 8;
  const previewRect = preview.getBoundingClientRect();
  const x = Math.min(
    window.innerWidth - previewRect.width - margin,
    event.clientX + HOVER_PREVIEW_OFFSET_PX,
  );
  const y = Math.min(
    window.innerHeight - previewRect.height - margin,
    event.clientY + HOVER_PREVIEW_OFFSET_PX,
  );
  preview.style.left = `${Math.max(margin, x)}px`;
  preview.style.top = `${Math.max(margin, y)}px`;
}

function hideTimelineHoverPreview(event) {
  const leavingChip = event.target?.closest?.(".timeline-chip");
  const enteringChip = event.relatedTarget?.closest?.(".timeline-chip");
  if (!leavingChip || leavingChip === enteringChip) return;
  const preview = document.getElementById("timeline-hover-preview");
  if (preview) {
    preview.hidden = true;
  }
}

function ensureTimelineHoverPreview() {
  let preview = document.getElementById("timeline-hover-preview");
  if (preview) {
    return preview;
  }
  preview = document.createElement("div");
  preview.id = "timeline-hover-preview";
  preview.className = "timeline-hover-preview";
  preview.hidden = true;
  document.body.append(preview);
  return preview;
}

function renderTimelineHoverPreview(preview, chip) {
  const item = timelineItemFromChip(chip);
  if (!item) {
    preview.hidden = true;
    return;
  }

  preview.replaceChildren();
  const title = document.createElement("strong");
  title.className = "hover-preview-title";
  title.textContent = item.title;

  const meta = document.createElement("span");
  meta.className = "hover-preview-meta";
  meta.textContent = item.meta;

  preview.append(title, meta);
  if (item.detail) {
    const detail = document.createElement("span");
    detail.className = "hover-preview-detail";
    detail.textContent = item.detail;
    preview.append(detail);
  }
}

function timelineItemFromChip(chip) {
  const laneIndex = parseInt(chip.dataset.laneIndex, 10);
  const itemIndex = parseInt(chip.dataset.itemIndex, 10);
  const itemType = chip.dataset.itemType;
  const lane = state.lanes[laneIndex];
  if (!lane) return null;

  if (itemType === "word") {
    const word = lane.words?.[itemIndex];
    if (!word) return null;
    const revision = word._revisions?.[word._revisions.length - 1];
    const prosodyLevel = prosodyLevelForWord(word);
    return {
      title: word.text,
      meta: `${lane.label} · ${word.resolvedTiming.start_ms}–${word.resolvedTiming.end_ms} ms · ${describeSpanState(word.commitment)}`,
      detail: [
        revision ? `revised from "${revision.fromText}" at ${revision.at_ms} ms` : word.timingSourceDetail,
        prosodyLevel !== null ? `prosody ${(prosodyLevel * 100).toFixed(0)}%` : null,
      ].filter(Boolean).join(" · "),
    };
  }

  const timelineEvent = lane.events?.[itemIndex];
  if (!timelineEvent) return null;
  return {
    title: timelineEvent.label,
    meta: `${lane.label} · ${timelineEvent.start_ms}–${timelineEvent.end_ms} ms · ${labelForKind(timelineEvent.kind)}`,
    detail: eventPreviewDetail(timelineEvent),
  };
}

function eventPreviewDetail(event) {
  const details = eventClockLines(event);
  if (event.metadata?.derived) {
    details.push("derived semantic span");
  }
  const face = event.metadata?.face ?? event.metadata?.event?.face;
  if (face) {
    details.push(`face ${face}`);
  }
  const reason = event.metadata?.reason ?? event.metadata?.event?.reason;
  if (reason) {
    details.push(reason);
  }
  const action = event.metadata?.action ?? event.metadata?.policy ?? event.metadata?.event?.artifact?.action;
  if (action) {
    details.push(String(action));
  }
  return details.length ? details.join(" · ") : null;
}

// Keyboard shortcuts for DAW-style navigation.
window.addEventListener("keydown", function onTimelineKeyDown(event) {
  if (event.target?.tagName === "INPUT" || event.target?.tagName === "TEXTAREA" || event.target?.isContentEditable) return;

  switch (event.key) {
    case "f":
      if (event.shiftKey) {
        event.preventDefault();
        zoomToFullSession();
      } else if (!event.ctrlKey && !event.metaKey) {
        event.preventDefault();
        zoomToSelection();
      }
      break;
    case "l":
      if (!event.shiftKey && !event.ctrlKey && !event.metaKey) {
        event.preventDefault();
        toggleFollowLatest();
      }
      break;
    case "j":
      if (!event.ctrlKey && !event.metaKey) {
        event.preventDefault();
        jumpSelectedWord(-1);
      }
      break;
    case "k":
      if (!event.ctrlKey && !event.metaKey) {
        event.preventDefault();
        jumpSelectedWord(1);
      }
      break;
    case "0":
      if (!event.ctrlKey && !event.metaKey) {
        event.preventDefault();
        resetZoom();
      }
      break;
    case "+":
    case "=":
      if (!event.ctrlKey && !event.metaKey) {
        event.preventDefault();
        zoomTimelineIn();
      }
      break;
    case "-":
      if (!event.ctrlKey && !event.metaKey) {
        event.preventDefault();
        zoomTimelineOut();
      }
      break;
    case "Escape":
      state.selectedItem = null;
      state.brushSelection = null;
      updateChipStates();
      updateTimeRangeSelectionOverlays();
      renderSelection();
      renderShell();
      break;
  }
});

function refreshPlaybackState() {
  updatePlaybackCursorOverlays();
  updateChipStates();
  renderShell();
}

function renderSelection() {
  updateZoomControls();
}

function selectWord(laneIndex, wordIndex, seekAudio) {
  const word = state.lanes[laneIndex]?.words?.[wordIndex];
  if (!word) {
    return;
  }

  state.selectedItem = { type: "word", laneIndex, itemIndex: wordIndex };
  if (seekAudio && audio.src) {
    audio.currentTime = word.resolvedTiming.start_ms / 1000;
  }

  clearPlaybackStop();
  ensureTimingVisible({
    startMs: word.resolvedTiming.start_ms,
    endMs: Math.max(word.resolvedTiming.end_ms, word.resolvedTiming.start_ms + 1),
  });
  updateChipStates();
  updateTimeRangeSelectionOverlays();
  renderSelection();
  renderShell();
}

function selectEvent(laneIndex, eventIndex, seekAudio) {
  const event = state.lanes[laneIndex]?.events?.[eventIndex];
  if (!event) {
    return;
  }

  state.selectedItem = { type: "event", laneIndex, itemIndex: eventIndex };
  if (seekAudio) {
    if (event.audio_ref?.url) {
      playAudioClip(event.audio_ref, event.start_ms, event.end_ms, false);
    } else if (audio.src) {
      audio.currentTime = event.start_ms / 1000;
    }
  }

  clearPlaybackStop();
  ensureTimingVisible({
    startMs: event.start_ms,
    endMs: Math.max(event.end_ms, event.start_ms + 1),
  });
  updateChipStates();
  updateTimeRangeSelectionOverlays();
  renderSelection();
  renderShell();
}

function ensureTimingVisible(timing) {
  if (!timing) {
    return;
  }
  const col = getScrollContainer();
  if (!col) {
    return;
  }
  const startPx = pxForMs(timing.startMs);
  const endPx = pxForMs(Math.max(timing.endMs, timing.startMs + 1));
  const nextScrollLeft = scrollLeftForTimingFocus(
    { startPx, endPx },
    col.scrollLeft,
    col.clientWidth,
    col.scrollWidth,
  );
  if (nextScrollLeft === null) {
    return;
  }
  _programmaticScroll = true;
  col.scrollLeft = nextScrollLeft;
  requestAnimationFrame(() => {
    _programmaticScroll = false;
  });
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

function zoomTimelineIn() {
  state.zoomPxPerSecond = clampZoom(state.zoomPxPerSecond * ZOOM_STEP_FACTOR);
  renderCustomTimeline();
  updateZoomControls();
}

function zoomTimelineOut() {
  state.zoomPxPerSecond = clampZoom(state.zoomPxPerSecond / ZOOM_STEP_FACTOR);
  renderCustomTimeline();
  updateZoomControls();
}

function zoomToTimeSelection(selection) {
  const clamped = clampTimeSelection(selection);
  if (!clamped) return;

  const selMs = clamped.endMs - clamped.startMs;
  const paddingMs = Math.max(ZOOM_SELECTION_PADDING_MIN_MS, selMs * ZOOM_SELECTION_PADDING_FACTOR);
  const viewStartMs = Math.max(0, clamped.startMs - paddingMs);
  const viewEndMs = Math.min(state.maxDurationMs, clamped.endMs + paddingMs);
  const viewMs = viewEndMs - viewStartMs;

  const col = getScrollContainer();
  const trackPx = col ? col.clientWidth : 600;
  state.zoomPxPerSecond = clampZoom((trackPx / Math.max(viewMs, 1)) * 1000);
  renderCustomTimeline();

  if (col) {
    _programmaticScroll = true;
    col.scrollLeft = pxForMs(viewStartMs);
    requestAnimationFrame(() => {
      _programmaticScroll = false;
    });
  }
  updateZoomControls();
}

function zoomToSelection() {
  const timing = state.brushSelection ?? selectedItemTiming();
  if (!timing) return;
  zoomToTimeSelection(timing);
}

function zoomToFullSession() {
  state.zoomPxPerSecond = DEFAULT_ZOOM_PX_PER_SECOND;
  renderCustomTimeline();
  const col = getScrollContainer();
  if (col) {
    _programmaticScroll = true;
    col.scrollLeft = 0;
    requestAnimationFrame(() => {
      _programmaticScroll = false;
    });
  }
  updateZoomControls();
}

function zoomToLatest() {
  const endMs = state.maxDurationMs;
  const startMs = Math.max(0, endMs - ZOOM_LATEST_WINDOW_MS);
  zoomToTimeSelection({ startMs, endMs });
}

function resetZoom() {
  state.zoomPxPerSecond = DEFAULT_ZOOM_PX_PER_SECOND;
  renderCustomTimeline();
  updateZoomControls();
}

function toggleFollowLatest() {
  state.followLatest = !state.followLatest;
  if (state.followLatest) {
    applyFollowLatest();
  }
  renderShell();
}

function applyFollowLatest() {
  const col = getScrollContainer();
  if (!col) return;
  const targetScroll = Math.max(0, getTrackContentWidth() - col.clientWidth);
  _programmaticScroll = true;
  col.scrollLeft = targetScroll;
  requestAnimationFrame(() => {
    _programmaticScroll = false;
  });
}

function selectedItemTiming() {
  if (!state.selectedItem) {
    return null;
  }

  const lane = state.lanes[state.selectedItem.laneIndex];
  if (state.selectedItem.type === "event") {
    const event = lane?.events?.[state.selectedItem.itemIndex];
    if (!event) {
      return null;
    }
    return {
      startMs: event.start_ms,
      endMs: Math.max(event.end_ms, event.start_ms + 1),
    };
  }

  const word = lane?.words?.[state.selectedItem.itemIndex];
  if (!word) {
    return null;
  }
  return {
    startMs: word.resolvedTiming.start_ms,
    endMs: Math.max(word.resolvedTiming.end_ms, word.resolvedTiming.start_ms + 1),
  };
}

function updateTimeRangeSelectionOverlays() {
  const timeRangeSelection = activeTimeRangeSelection();
  const selectedTiming = timeRangeSelection ? null : selectedItemTiming();
  document.querySelectorAll(".lane-track").forEach((trackEl) => {
    const overlay = trackEl.querySelector(".time-range-selection");
    if (overlay) {
      if (!timeRangeSelection) {
        overlay.hidden = true;
      } else {
        const startPx = pxForMs(timeRangeSelection.startMs);
        const widthPx = Math.max(0, pxForMs(timeRangeSelection.endMs) - startPx);
        overlay.hidden = false;
        overlay.style.left = `${startPx}px`;
        overlay.style.width = `${widthPx}px`;
        const label = overlay.querySelector(".time-range-selection-label");
        if (label) {
          label.textContent = `${formatRulerLabel(timeRangeSelection.startMs)}–${formatRulerLabel(timeRangeSelection.endMs)}`;
        }
      }
    }
    const focusOverlay = trackEl.querySelector(".selection-focus-overlay");
    if (!focusOverlay) {
      return;
    }
    if (!selectedTiming) {
      focusOverlay.hidden = true;
      return;
    }
    const startPx = pxForMs(selectedTiming.startMs);
    const widthPx = Math.max(0, pxForMs(selectedTiming.endMs) - startPx);
    focusOverlay.hidden = false;
    focusOverlay.style.left = `${startPx}px`;
    focusOverlay.style.width = `${widthPx}px`;
  });
  updatePlaybackCursorOverlays();
  updateZoomControls();
}

function updatePlaybackCursorOverlays() {
  if (!state.playbackCursorElements.length) {
    return;
  }
  const nowMs = currentPlaybackTimeMs();
  const leftPx = pxForMs(nowMs);
  for (const cursor of state.playbackCursorElements) {
    cursor.style.left = `${leftPx}px`;
    cursor.hidden = false;
  }
}

function currentPlaybackTimeMs() {
  return Math.max(0, Math.min(state.maxDurationMs, Math.round(audio.currentTime * 1000)));
}

function activeTimeRangeSelection() {
  if (state.dragSelection) {
    return normalizeTimeSelection(state.dragSelection.startMs, state.dragSelection.endMs);
  }
  return clampTimeSelection(state.brushSelection);
}

function startTimeRangeSelection(event) {
  const target = event.target instanceof Element ? event.target : null;
  if (!state.lanes.length || event.button !== 0 || target?.closest(".timeline-chip")) {
    return;
  }

  const surface = target?.closest(".lane-track, .timeline-ruler");
  if (!surface) {
    return;
  }

  const startMs = clientXToTimelineMs(event.clientX);
  if (startMs === null) {
    return;
  }

  event.preventDefault();
  state.dragSelection = {
    pointerId: event.pointerId,
    surface,
    startClientX: event.clientX,
    startMs,
    endMs: startMs,
  };
  const col = getScrollContainer();
  if (col) col.setPointerCapture(event.pointerId);
  updateTimeRangeSelectionOverlays();
}

function moveTimeRangeSelection(event) {
  if (!state.dragSelection || state.dragSelection.pointerId !== event.pointerId) {
    return;
  }

  const endMs = clientXToTimelineMs(event.clientX);
  if (endMs === null) {
    return;
  }
  state.dragSelection.endMs = endMs;
  updateTimeRangeSelectionOverlays();
}

function finishTimeRangeSelection(event) {
  if (!state.dragSelection || state.dragSelection.pointerId !== event.pointerId) {
    return;
  }

  const dragSelection = state.dragSelection;
  const endMs = clientXToTimelineMs(event.clientX);
  if (endMs !== null) {
    dragSelection.endMs = endMs;
  }

  state.dragSelection = null;
  const col = getScrollContainer();
  if (col && col.hasPointerCapture(event.pointerId)) {
    col.releasePointerCapture(event.pointerId);
  }

  const delta = Math.abs(event.clientX - dragSelection.startClientX);
  if (delta < RANGE_SELECTION_DRAG_THRESHOLD_PX) {
    state.brushSelection = null;
    if (audio.src && endMs !== null) {
      audio.currentTime = endMs / 1000;
      refreshPlaybackState();
    }
    updateTimeRangeSelectionOverlays();
    return;
  }

  const selection = normalizeTimeSelection(dragSelection.startMs, dragSelection.endMs);
  if (!selection) {
    updateTimeRangeSelectionOverlays();
    return;
  }

  state.suppressTimelineClick = true;
  window.setTimeout(() => {
    state.suppressTimelineClick = false;
  }, 0);
  state.brushSelection = selection;
  zoomToTimeSelection(selection);
}

function cancelTimeRangeSelection(event) {
  if (!state.dragSelection || state.dragSelection.pointerId !== event.pointerId) {
    return;
  }

  state.dragSelection = null;
  const col = getScrollContainer();
  if (col && col.hasPointerCapture(event.pointerId)) {
    col.releasePointerCapture(event.pointerId);
  }
  updateTimeRangeSelectionOverlays();
}

function clientXToTimelineMs(clientX) {
  const col = getScrollContainer();
  if (!col) return null;
  const rect = col.getBoundingClientRect();
  if (!rect.width) return null;
  const xInContent = clientX - rect.left + col.scrollLeft;
  return Math.max(0, Math.min(state.maxDurationMs, msForPx(xInContent)));
}

function normalizeTimeSelection(startMs, endMs) {
  if (!Number.isFinite(startMs) || !Number.isFinite(endMs)) {
    return null;
  }

  const start = Math.max(0, Math.min(state.maxDurationMs, Math.min(startMs, endMs)));
  const end = Math.max(0, Math.min(state.maxDurationMs, Math.max(startMs, endMs)));
  if (end <= start) {
    return null;
  }
  return { startMs: start, endMs: end };
}

function clampTimeSelection(selection) {
  if (!selection) {
    return null;
  }
  return normalizeTimeSelection(selection.startMs, selection.endMs);
}

function updateZoomControls() {
  renderShell();
}

function firstItemSelection() {
  const firstWordLane = state.lanes.find((lane) => lane.type === "word" && lane.words.length > 0);
  if (firstWordLane) {
    return { type: "word", laneIndex: firstWordLane.words[0].laneIndex, itemIndex: 0 };
  }

  const firstEventLane = state.lanes.find((lane) => lane.type === "event" && lane.events.length > 0);
  if (firstEventLane) {
    return { type: "event", laneIndex: firstEventLane.events[0].laneIndex, itemIndex: 0 };
  }

  return null;
}

function validSelection(selection) {
  if (!selection) {
    return false;
  }
  const lane = state.lanes[selection.laneIndex];
  if (selection.type === "word") {
    return Boolean(lane?.type === "word" && lane.words[selection.itemIndex]);
  }
  return Boolean(lane?.type === "event" && lane.events[selection.itemIndex]);
}

function flattenWords() {
  return state.lanes
    .filter((lane) => lane.type === "word")
    .flatMap((lane) => lane.words)
    .sort((left, right) => left.resolvedTiming.start_ms - right.resolvedTiming.start_ms);
}

function fallbackTiming(wordCount, wordIndex, durationMs) {
  const slot = durationMs / wordCount;
  return {
    start_ms: Math.round(slot * wordIndex),
    end_ms: Math.round(slot * (wordIndex + 1)),
  };
}

function defaultLaneLabel(stream, index) {
  return `${sourceLabels[stream?.source] ?? "Stream"} ${index + 1}`;
}

function isTimedWordStream(candidate) {
  return Boolean(candidate && typeof candidate === "object" && Array.isArray(candidate.words));
}

function describeTimingSource(word, hasMeasuredTiming) {
  if (!hasMeasuredTiming) {
    return "fallback layout timing (resolved locally, not measured)";
  }

  if (word.boundary_source) {
    return `measured word.timing (${word.boundary_source})`;
  }

  return "measured word.timing";
}

function itemKey(itemType, laneIndex, itemIndex) {
  return `${itemType}:${laneIndex}:${itemIndex}`;
}

function parseItemKey(key) {
  const [itemType, laneStr, itemStr] = key.split(":");
  return { itemType, laneIndex: parseInt(laneStr, 10), itemIndex: parseInt(itemStr, 10) };
}

function classToken(value) {
  return String(value ?? "event")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, "-");
}

function createChipContent(text) {
  const content = document.createElement("span");
  content.className = "chip-content";
  content.textContent = text;
  return content;
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

function appendWordChipContent(chip, word, includeBadges) {
  const prosodyLevel = prosodyLevelForWord(word);
  if (prosodyLevel !== null) {
    chip.append(createProsodyStrip(prosodyLevel));
  }
  const latestRevision = word._revisions?.[word._revisions.length - 1];
  if (latestRevision?.fromText) {
    const ghost = document.createElement("span");
    ghost.className = "revision-ghost";
    ghost.textContent = latestRevision.fromText;
    chip.append(ghost);
  }
  chip.append(createChipContent(word.text));
  if (includeBadges) {
    chip.append(createChipBadges([describeShortCommitment(word.commitment)]));
  }
}

function createProsodyStrip(level) {
  const strip = document.createElement("span");
  strip.className = "prosody-strip";
  strip.setAttribute("aria-hidden", "true");
  const fill = document.createElement("span");
  fill.className = "prosody-fill";
  fill.style.transform = `scaleX(${level})`;
  strip.append(fill);
  return strip;
}

function appendAlignmentHandles(chip, word) {
  if (!word.timing) {
    return;
  }
  const startHandle = document.createElement("span");
  startHandle.className = "alignment-handle alignment-start";
  startHandle.setAttribute("aria-hidden", "true");
  const endHandle = document.createElement("span");
  endHandle.className = "alignment-handle alignment-end";
  endHandle.setAttribute("aria-hidden", "true");
  chip.append(startHandle, endHandle);
}

function createTokenDensityBar(word) {
  const bar = document.createElement("span");
  bar.className = "token-density-bar";
  bar.dataset.commitment = commitmentClass(word.commitment);
  return bar;
}

function createChipBadges(labels) {
  const badges = document.createElement("span");
  badges.className = "chip-badges";
  labels.filter(Boolean).forEach((label) => {
    const badge = document.createElement("span");
    badge.className = "chip-badge";
    badge.textContent = label;
    badges.append(badge);
  });
  return badges;
}

function describeShortCommitment(commitment) {
  switch (commitment) {
    case "Hypothetical": return "Hyp";
    case "StableText": return "Stable";
    case "Prepared": return "Prep";
    case "Playable": return "Ready";
    case "Played": return "Played";
    case "Final": return "Final";
    case "Cancelled": return "Cancel";
    default: return null;
  }
}

function shortKindLabel(kind) {
  switch (kind) {
    case "speech_started": return "speech";
    case "asr_started": return "asr";
    case "llm_generation_started": return "llm";
    case "first_llm_token": return "token";
    case "playback_started": return "play";
    case "breath_group": return "breath";
    case "self_hearing_suppression_started": return "self";
    default: return labelForKind(kind).split(" ").slice(0, 2).join(" ");
  }
}

function prosodyLevelForWord(word) {
  const candidates = [
    word.prosody?.energy,
    word.prosody?.intensity,
    word.prosody?.volume,
    word.prosody?.stress,
    word.energy,
    word.intensity,
    word.volume,
    word.stress,
  ];
  const value = candidates.find((candidate) => Number.isFinite(candidate));
  if (value == null) {
    return null;
  }
  return Number(value) > 1 ? Math.min(1, Number(value) / 100) : Math.max(0, Math.min(1, Number(value)));
}

function isOverlapKind(kind) {
  return String(kind ?? "").includes("overlap") || String(kind ?? "").includes("barge");
}

function isInterruptionKind(kind) {
  const token = String(kind ?? "");
  return token.includes("interrupt") || token.includes("yield") || token.includes("cancelled");
}

function isTokenKind(kind) {
  return String(kind ?? "").includes("token");
}

function isEmotionKind(kind) {
  const token = String(kind ?? "");
  return token.includes("emotion") || token.includes("affect") || token.includes("face") || token.includes("expression");
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function normalizeEventAudioRef(entry, fallbackStartMs, fallbackEndMs) {
  const candidate =
    entry?.audio_ref ??
    entry?.metadata?.audio_ref ??
    entry?.metadata?.artifact?.audio_ref ??
    entry?.metadata?.artifact?.audio ??
    entry?.metadata?.artifact ??
    null;
  if (!candidate || typeof candidate !== "object") {
    return null;
  }
  const url =
    normalizeAudioRefString(candidate.url) ??
    normalizeAudioRefString(candidate.audio_url) ??
    normalizeAudioRefString(candidate.path) ??
    normalizeAudioRefString(candidate.audio_path);
  if (!url) {
    return null;
  }

  const startMs = normalizeAudioRefMs(
    candidate.start_ms ?? candidate.clip_start_ms ?? candidate.at_ms,
    fallbackStartMs,
  );
  const endMs = normalizeAudioRefMs(candidate.end_ms ?? candidate.clip_end_ms, fallbackEndMs);

  return {
    url,
    start_ms: startMs,
    end_ms: Math.max(startMs, endMs),
  };
}

function firstFiniteNumber(...values) {
  for (const value of values) {
    const number = Number(value);
    if (Number.isFinite(number)) {
      return number;
    }
  }
  return null;
}

function eventClocksFromEntry(entry, spanStartMs, spanEndMs, audioRef) {
  const metadata = entry?.metadata ?? {};
  const event = metadata?.event ?? {};
  const receivedMs = firstFiniteNumber(
    entry?.received_ms,
    metadata?.received_ms,
    event?.received_ms,
  );
  const eventMs = firstFiniteNumber(
    metadata?.start_event_ms,
    entry?.event_ms,
    entry?.at_ms,
    metadata?.event_ms,
    metadata?.elapsed_ms,
    event?.elapsed_ms,
    spanStartMs,
  );
  const endEventMs = firstFiniteNumber(
    metadata?.end_event_ms,
    metadata?.event?.elapsed_ms,
    entry?.end_event_ms,
    spanEndMs,
  );

  return {
    event_ms: eventMs,
    end_event_ms: endEventMs,
    span_start_ms: spanStartMs,
    span_end_ms: spanEndMs,
    audio_start_ms: audioRef?.start_ms ?? null,
    audio_end_ms: audioRef?.end_ms ?? null,
    received_ms: receivedMs,
  };
}

function formatClockValue(value) {
  return Number.isFinite(value) ? `${Math.round(value)} ms` : "n/a";
}

function eventClockLines(event) {
  const clocks = event?.clocks ?? {};
  return [
    `event_ms: ${formatClockValue(clocks.event_ms)}`,
    `span_start_ms: ${formatClockValue(clocks.span_start_ms)}`,
    `span_end_ms: ${formatClockValue(clocks.span_end_ms)}`,
    `audio_start_ms: ${formatClockValue(clocks.audio_start_ms)}`,
    `audio_end_ms: ${formatClockValue(clocks.audio_end_ms)}`,
    `received_ms: ${formatClockValue(clocks.received_ms)}`,
  ];
}

function eventClockSummary(event) {
  return eventClockLines(event).join(" · ");
}

function eventVisualBounds(event, fallbackStartMs, fallbackEndMs) {
  const clocks = event?.clocks ?? {};
  const useAudioClock = !isSuppressionKind(event?.kind);
  const candidates = [
    fallbackStartMs,
    fallbackEndMs,
    clocks.event_ms,
    useAudioClock ? clocks.audio_start_ms : null,
    useAudioClock ? clocks.audio_end_ms : null,
  ].filter(Number.isFinite);
  if (!candidates.length) {
    return { startMs: fallbackStartMs, endMs: fallbackEndMs };
  }
  const startMs = Math.min(...candidates);
  const endMs = Math.max(...candidates, startMs + 1);
  return { startMs, endMs };
}

function percentWithin(startMs, endMs, valueMs) {
  if (!Number.isFinite(valueMs) || endMs <= startMs) {
    return null;
  }
  const pct = ((valueMs - startMs) / (endMs - startMs)) * 100;
  return Math.max(0, Math.min(100, pct));
}

function eventHasClockDivergence(event) {
  const clocks = event?.clocks ?? {};
  const values = [
    clocks.event_ms,
    clocks.span_start_ms,
    !isSuppressionKind(event?.kind) ? clocks.audio_start_ms : null,
  ].filter(Number.isFinite);
  if (values.length < 2) {
    return false;
  }
  const min = Math.min(...values);
  const max = Math.max(...values);
  return max - min >= 1;
}

function appendEventClockOverlays(chip, event, visualStartMs, visualEndMs) {
  const clocks = event?.clocks ?? {};
  const useAudioClock = !isSuppressionKind(event?.kind);
  const visualDuration = visualEndMs - visualStartMs;
  if (visualDuration <= 0) {
    return;
  }

  const leadStart = firstFiniteNumber(useAudioClock ? clocks.audio_start_ms : null, clocks.span_start_ms);
  const leadEnd = clocks.event_ms;
  if (Number.isFinite(leadStart) && Number.isFinite(leadEnd) && leadEnd > leadStart) {
    const leadLeft = percentWithin(visualStartMs, visualEndMs, leadStart);
    const leadRight = percentWithin(visualStartMs, visualEndMs, leadEnd);
    if (leadLeft != null && leadRight != null && leadRight > leadLeft) {
      const lead = document.createElement("span");
      lead.className = "event-derived-lead";
      lead.style.left = `${leadLeft}%`;
      lead.style.width = `${leadRight - leadLeft}%`;
      chip.append(lead);
    }
  }

  if (useAudioClock && Number.isFinite(clocks.audio_start_ms) && Number.isFinite(clocks.audio_end_ms)) {
    const audioLeft = percentWithin(visualStartMs, visualEndMs, clocks.audio_start_ms);
    const audioRight = percentWithin(visualStartMs, visualEndMs, clocks.audio_end_ms);
    if (audioLeft != null && audioRight != null && audioRight > audioLeft) {
      const audio = document.createElement("span");
      audio.className = "event-audio-duration";
      audio.style.left = `${audioLeft}%`;
      audio.style.width = `${audioRight - audioLeft}%`;
      chip.append(audio);
    }
  }

  const tickLeft = percentWithin(visualStartMs, visualEndMs, clocks.event_ms);
  if (tickLeft != null) {
    const tick = document.createElement("span");
    tick.className = "event-emission-tick";
    tick.style.left = `${tickLeft}%`;
    chip.append(tick);
  }
}

function isSuppressionKind(kind) {
  return String(kind ?? "").includes("suppression");
}

function normalizeAudioRefString(value) {
  if (typeof value !== "string") {
    return null;
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

function normalizeAudioRefMs(value, fallbackValue) {
  return Math.max(0, Math.round(Number.isFinite(value) ? value : fallbackValue));
}

function playSelectedClip() {
  if (state.selectedItem?.type !== "event") {
    return;
  }
  const event = state.lanes[state.selectedItem.laneIndex]?.events?.[state.selectedItem.itemIndex];
  if (!event?.audio_ref?.url) {
    return;
  }
  playAudioClip(event.audio_ref, event.start_ms, event.end_ms, true);
}

function playAudioClip(audioRef, fallbackStartMs, fallbackEndMs, autoplay) {
  const startMs = normalizeAudioRefMs(audioRef?.start_ms, fallbackStartMs);
  const endMs = normalizeAudioRefMs(audioRef?.end_ms, fallbackEndMs);
  const targetUrl = audioRef?.url;
  if (!targetUrl) {
    return;
  }

  const seekAndMaybePlay = () => {
    audio.currentTime = startMs / 1000;
    setPlaybackStop(startMs, endMs);
    if (autoplay) {
      void audio.play();
    }
    refreshPlaybackState();
  };

  if (audio.src !== targetUrl) {
    audio.src = targetUrl;
    audio.addEventListener("loadedmetadata", seekAndMaybePlay, { once: true });
    uiState.statusMessage = `Loaded clip reference ${targetUrl}.`;
    renderShell();
    return;
  }
  seekAndMaybePlay();
}

function setPlaybackStop(startMs, endMs) {
  state.stopAtMs = endMs > startMs ? endMs : null;
}

function clearPlaybackStop() {
  state.stopAtMs = null;
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

function coerceMs(value, label) {
  if (!Number.isFinite(value)) {
    throw new Error(`${label} must be a finite number, received: ${String(value)}`);
  }
  return Math.max(0, Math.round(value));
}

function formatPlaybackTime() {
  return `${audio.currentTime.toFixed(3)}s / ${(state.maxDurationMs / 1000).toFixed(3)}s`;
}

function buildSelectionProjection() {
  if (!state.selectedItem) {
    return {
      canPlaySelectionClip: false,
      playSelectionClipLabel: "Play selected clip",
      summaryParts: [DEFAULT_SELECTION_MESSAGE],
      selectionJson: "{}",
      badge: null,
      revisions: [],
    };
  }

  if (state.selectedItem.type === "event") {
    const lane = state.lanes[state.selectedItem.laneIndex];
    const event = lane?.events?.[state.selectedItem.itemIndex];
    if (!lane || !event) {
      return {
        canPlaySelectionClip: false,
        playSelectionClipLabel: "Play selected clip",
        summaryParts: [DEFAULT_SELECTION_MESSAGE],
        selectionJson: "{}",
        badge: null,
        revisions: [],
      };
    }
    return {
      canPlaySelectionClip: Boolean(event.audio_ref?.url),
      playSelectionClipLabel: event.audio_ref?.url ? "Play event clip" : "Play selected clip",
      summaryParts: [
        h("strong", null, lane.label),
        h("br"),
        "Event ",
        h("strong", null, event.label),
        h("br"),
        `${event.start_ms}–${event.end_ms} ms · kind `,
        h("strong", null, event.kind),
        h("br"),
        eventClockSummary(event),
      ],
      selectionJson: JSON.stringify(
        {
          lane: lane.label,
          laneType: "event",
          id: event.id,
          kind: event.kind,
          label: event.label,
          start_ms: event.start_ms,
          end_ms: event.end_ms,
          duration_ms: Math.max(0, event.end_ms - event.start_ms),
          clocks: event.clocks,
          audioRef: event.audio_ref,
          metadata: event.metadata,
          original: event.original,
        },
        null,
        2,
      ),
      badge: null,
      revisions: [],
    };
  }

  const lane = state.lanes[state.selectedItem.laneIndex];
  const word = lane?.words?.[state.selectedItem.itemIndex];
  if (!lane || !word) {
    return {
      canPlaySelectionClip: false,
      playSelectionClipLabel: "Play selected clip",
      summaryParts: [DEFAULT_SELECTION_MESSAGE],
      selectionJson: "{}",
      badge: null,
      revisions: [],
    };
  }
  return {
    canPlaySelectionClip: false,
    playSelectionClipLabel: "Play selected clip",
    summaryParts: [
      h("strong", null, lane.label),
      h("br"),
      "Word ",
      h("strong", null, word.text),
      h("br"),
      `${word.resolvedTiming.start_ms}–${word.resolvedTiming.end_ms} ms · confidence ${word.timing_confidence ?? "n/a"}`,
      h("br"),
      "Timing source: ",
      h("strong", null, word.timingSourceDetail),
    ],
    selectionJson: JSON.stringify(
      {
        lane: lane.label,
        source: lane.stream.source,
        streamId: lane.stream.id,
        wordId: word.id,
        text: word.text,
        timing: word.timing,
        resolvedTiming: word.resolvedTiming,
        timingResolution: word.timingResolution,
        timingSourceDetail: word.timingSourceDetail,
        confidence: word.timing_confidence,
        commitment: word.commitment,
        spanState: describeSpanState(word.commitment),
        boundarySource: word.boundary_source,
        lexicalSpan: word.lexical_span,
        audioRef: word.audio_ref,
        revisions: word._revisions ?? [],
      },
      null,
      2,
    ),
    badge: word.commitment
      ? {
          className: `inspector-span-state commit-${commitmentClass(word.commitment)}`,
          text: describeSpanState(word.commitment),
        }
      : null,
    revisions: (word._revisions ?? []).map((rev) => ({
      atMs: rev.at_ms,
      fromText: rev.fromText,
      toText: word.text,
    })),
  };
}

// ── Span state helpers ────────────────────────────────────────────────────

/**
 * Return a human-readable description of a WordCommitment span state.
 * Matches the values of the Rust `WordCommitment` enum (serialized as PascalCase strings).
 */
function describeSpanState(commitment) {
  switch (commitment) {
    case "Hypothetical": return "Hypothesis — text may change";
    case "StableText":   return "Stable — text locked, not yet synthesised";
    case "Prepared":     return "Prepared — queued for synthesis";
    case "Playable":     return "Playable — audio ready, playback imminent";
    case "Played":       return "Played — currently being spoken";
    case "Final":        return "Committed — played and confirmed";
    case "Cancelled":    return "Cancelled — abandoned before playback";
    default:             return commitment ?? "Unknown";
  }
}

/**
 * Build an HTML fragment for the revision history of a word span.
 * Returns an empty string when there are no revisions.
 */
function buildRevisionHistoryHtml(word) {
  const revisions = word._revisions;
  if (!revisions?.length) {
    return "";
  }
  const rows = revisions
    .map((rev) => {
      const reason = rev.reason ? `<div class="inspector-revision-reason">${escapeHtml(rev.reason)}</div>` : "";
      return `<div class="inspector-revision-entry">
        <span>at ${rev.at_ms}ms:</span>
        <del>${escapeHtml(rev.fromText)}</del>
        <span>→</span>
        <span class="revision-new">${escapeHtml(word.text)}</span>
        ${reason}
      </div>`;
    })
    .join("");
  return `<div class="inspector-revision-history">
    <strong>↩ Retroactive revision</strong>
    ${rows}
  </div>`;
}
