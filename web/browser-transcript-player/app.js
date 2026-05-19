import { Fragment, h, render as preactRender } from "https://esm.sh/preact@10.26.9";
import cytoscape from "https://esm.sh/cytoscape@3.30.2";
import {
  createTimeScale,
  createTimelineViewport,
  renderedCanvasXToTimelinePx,
} from "/assets/timeline-viewport.mjs";
import {
  AlignmentKind,
  SpanModality,
  alignWordSpansToParentSpan,
  buildSharedSpanModel,
  createAlignment,
  createSpan,
  projectTimedWordsToSpans,
} from "/assets/shared-span-model.mjs";
import {
  LIVE_EVENT_LANE,
  SPAN_PAIRS,
  END_TO_START,
  isLlmTextEvent as isGeneratedSpeechEventKind,
} from "/assets/shared/events/schema.mjs";
import {
  normalizeSemanticText,
  normalizedId,
  speechUnitIdFromEvent,
  transcriptCandidateText as _transcriptCandidateText,
  wordStreamText as _wordStreamText,
  textContent,
} from "/assets/shared/events/reducers.mjs";
import {
  buildEnergyEnvelopeFromAudioBuffer,
  detectEnergyLandmarks,
  refineWordTimingsWithEnergy,
} from "/assets/energy-timing.mjs";

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
const LIVE_WAVEFORM_REFRESH_MS = 2_000;
const LIVE_WAVEFORM_GROWTH_REFRESH_MS = 1_000;
const GRAPH_MAX_RENDER_NODES = 180;
const GRAPH_MAX_RENDER_EDGES = 260;
// Central waveform panel layout constants
const WAVEFORM_PANEL_MAX_SPAN_ROWS = 4;
const WAVEFORM_SPAN_ROW_HEIGHT_PX = 20;
const WAVEFORM_SPAN_ROW_GAP_PX = 2;
const WAVEFORM_SPAN_ROW_STRIDE_PX = WAVEFORM_SPAN_ROW_HEIGHT_PX + WAVEFORM_SPAN_ROW_GAP_PX;
const WAVEFORM_SPAN_ROW_MARGIN_PX = 4;

// Serialize an open-span map key as [lane, turn, startKind].
function openSpanKey(lane, turn, startKind) {
  return JSON.stringify([lane, turn ?? null, startKind]);
}


// Accumulated live trace events (kept for error recovery / diagnostics).
const liveEvents = [];
let liveRenderScheduled = false;
let lastLiveWaveformRefreshAt = 0;
let liveWaveformRetryTimer = null;
let waveformRefreshInFlightUrl = null;
let nextWaveformPeakVersion = 1;
const waveformPeakVersions = new WeakMap();
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
  waveform: {
    url: null,
    peaks: null,
    durationMs: 0,
    status: "idle",
    energyEnvelope: null,
    energyLandmarks: null,
  },
  suppressTimelineClick: false,
  itemTimingByKey: new Map(),   // itemKey → {startMs, endMs}
  chipElementByKey: new Map(),  // itemKey → DOM element
  waveformChipElements: new Map(), // itemKey → waveform panel span element
  playbackCursorElements: [],
  surfaceMode: "timeline",
  graphFilters: {
    modality: "all",
    turn: "all",
    timeWindow: "all",
    commitment: "all",
    revisionsOnly: false,
    neighborhood: true,
  },
};

const graphState = {
  cy: null,
  bound: false,
};

const uiState = {
  liveMode: false,
  connectionStatusClass: "live-status-connecting",
  connectionStatusText: "connecting…",
  statusMessage: "Connecting to live event stream…",
  diagnosticsExpanded: false,
};

const sourceLabels = {
  RecordedAudio: "Recorded audio",
  LiveAsr: "Live ASR",
  GeneratedText: "Generated text",
  SyntheticSpeech: "Synthetic speech",
};

const DEFAULT_SELECTION_MESSAGE = "Select a word or event to inspect timing and metadata.";

function RibbonToken({ token }) {
  const className = [
    token.className,
    token.selection ? "transcript-token-button" : "",
    token.selected ? "selected" : "",
  ]
    .filter(Boolean)
    .join(" ");
  if (token.selection) {
    return h(
      "button",
      {
        type: "button",
        className,
        title: token.title ?? `Play "${token.text}"`,
        "aria-label": `Play transcript word: ${token.text}`,
        "aria-pressed": Boolean(token.selected),
        onClick: () => selectTranscriptToken(token),
      },
      token.text,
    );
  }
  return h(
    "span",
    {
      className,
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
      { className: "toolbar view-toolbar", id: "view-toolbar", "aria-label": "WaveDeck surface mode" },
      h("button", {
        id: "mode-timeline",
        type: "button",
        className: projection.surfaceMode === "timeline" ? "active-toggle" : "",
        "aria-pressed": projection.surfaceMode === "timeline",
        onClick: () => setSurfaceMode("timeline"),
      }, "Timeline"),
      h("button", {
        id: "mode-graph",
        type: "button",
        className: projection.surfaceMode === "graph" ? "active-toggle" : "",
        "aria-pressed": projection.surfaceMode === "graph",
        onClick: () => setSurfaceMode("graph"),
      }, "Graph"),
      h("button", {
        id: "diagnostics-toggle",
        type: "button",
        className: projection.diagnosticsExpanded ? "active-toggle" : "",
        "aria-pressed": projection.diagnosticsExpanded,
        disabled: !projection.diagnosticLaneCount,
        onClick: () => toggleDiagnostics(),
      }, projection.diagnosticsExpanded ? "Hide Diagnostics" : `Diagnostics (${projection.diagnosticLaneCount})`),
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
      h("span", { className: "span-legend-item span-state-confirmed" }, "Confirmed"),
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
    surfaceMode: state.surfaceMode,
    diagnosticsExpanded: uiState.diagnosticsExpanded,
    diagnosticLaneCount: state.lanes.filter((lane) => lane.type === "event").length,
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
  if (document.body.dataset.mode === "replay") {
    // Replay mode: expose the replay API and let replay.js control initialization.
    window.wavedeckReplay = {
      addLiveEvent,
      applyPayload,
      renderShell,
      resetLiveSession,
      uiState,
    };
    renderShell();
    return;
  }
  enterLiveMode();
}

// Reset the live session back to the initial empty state (used by replay.js for seek).
function resetLiveSession() {
  liveSession.turns.clear();
  liveSession.openSpans.clear();
  liveSession.viewerEvents.length = 0;
  liveSession.viewerMarkers.length = 0;
  liveSession.debugLog.length = 0;
  liveSession.maxElapsedMs = 0;
  liveSession.receivedOriginMs = performance.now();
  liveEvents.length = 0;
  applyLiveEvents();
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
    refinedTranscript: null,
    receivedOriginMs: performance.now(),
  };
}

function sessionGetOrCreateTurn(session, turnId) {
  if (!session.turns.has(turnId)) {
    session.turns.set(turnId, {
      id: turnId,
      transcriptCandidate: null,
      finalTranscript: null,
      finalTranscriptElapsedMs: null,
      latestWordStream: null,
      latestTtsWordStream: null,
      wordStreamTimeOffsetMs: null,
      wordRevisions: new Map(), // stableWordKey → [{fromText, at_ms, provenance, approximate}]
      generatedText: null,
      generatedSpeechFragments: [],
      speechUnitsById: new Map(),
      generatedSpeechUnitOrder: [],
    });
  }
  return session.turns.get(turnId);
}

function openSpanRecord(startMs, label = null, speechUnitId = null) {
  return { start_ms: startMs, label, speech_unit_id: speechUnitId };
}

function openSpanStartMs(record) {
  return typeof record === "number" ? record : record?.start_ms;
}

function openSpanLabel(record) {
  return typeof record === "number" ? null : record?.label ?? null;
}

function openSpanSpeechUnitId(record) {
  return typeof record === "number" ? null : record?.speech_unit_id ?? null;
}

// Returns a stable string key for a word, preferring:
//   1. stable WordId  2. lexical span bounds  3. array-index fallback
function stableWordKey(word, index) {
  if (word.span_id != null) {
    return `span-id:${word.span_id}`;
  }
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
  // 1. Stable SpanId
  if (newWord.span_id != null) {
    const found = prevWords.find((w) => w.span_id != null && w.span_id === newWord.span_id);
    if (found) {
      return { word: found, approximate: false };
    }
  }
  // 2. Stable WordId
  if (newWord.id != null) {
    const found = prevWords.find((w) => w.id != null && w.id === newWord.id);
    if (found) {
      return { word: found, approximate: false };
    }
  }
  // 3. Lexical span / text-offset overlap
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
  // 4. Array index fallback (approximate — provenance is marked accordingly)
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

function transcriptWords(text, commitment = "Confirmed") {
  return textContent(text).match(/\S+/g)?.map((word) => ({ text: word, commitment })) ?? [];
}

function committedAsrWordsThrough(session, elapsedMs) {
  const words = [];
  const turns = [...session.turns.values()].sort((left, right) => left.id - right.id);
  for (const turn of turns) {
    if (turn.id === 0 || turn.finalTranscript == null) {
      continue;
    }
    if (turn.finalTranscriptElapsedMs != null && turn.finalTranscriptElapsedMs > elapsedMs) {
      continue;
    }
    if (turn.latestWordStream?.words?.length) {
      for (let wordIndex = 0; wordIndex < turn.latestWordStream.words.length; wordIndex++) {
        const word = turn.latestWordStream.words[wordIndex];
        words.push({
          ...word,
          _turn: turn.id,
          _streamWordKey: stableWordKey(word, wordIndex),
          _streamWordIndex: wordIndex,
        });
      }
    } else {
      words.push(...transcriptWords(turn.finalTranscript, "Final"));
    }
  }
  return words;
}

function confirmedTranscriptWords(text, previousWords, elapsedMs) {
  return transcriptWords(text, "Confirmed").map((word, index) => {
    const previous = previousWords[index];
    const selectionMetadata = previous
      ? {
          _turn: previous._turn,
          _streamWordKey: previous._streamWordKey,
          _streamWordIndex: previous._streamWordIndex,
        }
      : {};
    if (!previous || previous.text === word.text) {
      return { ...word, ...selectionMetadata };
    }
    return {
      ...word,
      ...selectionMetadata,
      _revisions: [
        ...(previous._revisions ?? []),
        {
          fromText: previous.text,
          at_ms: elapsedMs,
          provenance: "confirmed by broader ASR context",
          approximate: true,
        },
      ],
    };
  });
}

function applyConfirmedTranscript(session, event) {
  const text = textContent(event.text);
  if (!text) {
    return;
  }
  const previousWords = committedAsrWordsThrough(session, event.elapsed_ms);
  session.refinedTranscript = {
    text,
    elapsedMs: event.elapsed_ms ?? 0,
    source: textContent(event.artifact?.source) || "refinement",
    segmentCount: Number.isFinite(event.artifact?.segment_count) ? event.artifact.segment_count : null,
    words: confirmedTranscriptWords(text, previousWords, event.elapsed_ms ?? 0),
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

  // ── tts_timed_word_stream_revision: update latest generated speech word stream
  if (event.kind === "tts_timed_word_stream_revision" && event.artifact?.words) {
    const liveTurn = sessionGetOrCreateTurn(session, turn);
    liveTurn.latestTtsWordStream = event.artifact;
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

  if ((event.kind === "transcript_proposition" || event.kind === "transcript_confirmed") && event.text) {
    applyConfirmedTranscript(session, event);
    log("stable", `Confirmed transcript: "${event.text}"`);
    // Fall through to emit as a lane marker below.
  }

  // ── transcript: commit final text for this turn
  if (event.kind === "transcript" && event.text) {
    const liveTurn = sessionGetOrCreateTurn(session, turn);
    liveTurn.finalTranscript = event.text;
    liveTurn.finalTranscriptElapsedMs = event.elapsed_ms ?? null;
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
        openSpanRecord(
          event.elapsed_ms,
          semanticSpanLabel(
            session,
            "breath_group_opened",
            turn,
            labelForKind("breath_group_opened"),
            event,
          ),
        ),
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
      openSpanRecord(
        event.elapsed_ms,
        semanticSpanLabel(session, event.kind, turn, labelForKind(event.kind), event),
        event.kind === "playback_started" ? speechUnitIdFromEvent(event) : null,
      ),
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
      label:
        openSpanLabel(openSpan) ??
        semanticSpanLabel(
          session,
          kind,
          spanTurn,
          `${labelForKind(kind)} (in progress)`,
          null,
          openSpan,
        ),
      start_ms: startMs,
      end_ms: session.maxElapsedMs,
      metadata: {
        in_progress: true,
        turn: spanTurn,
        ...(openSpanSpeechUnitId(openSpan) ? { speech_unit_id: openSpanSpeechUnitId(openSpan) } : {}),
      },
    });
  }

  // Live duplex ASR is one continuous timeline.  Keep reducer state per turn
  // for revision tracking, but project the latest ASR words onto one lane.
  const asrWords = [];
  const ttsWords = [];
  let nextWordId = 1;
  for (const [turnId, liveTurn] of [...session.turns.entries()].sort((a, b) => a[0] - b[0])) {
    if (liveTurn.latestWordStream?.words?.length > 0) {
      const offsetMs = liveTurn.wordStreamTimeOffsetMs ?? 0;
      for (let sourceWordIndex = 0; sourceWordIndex < liveTurn.latestWordStream.words.length; sourceWordIndex++) {
        const word = liveTurn.latestWordStream.words[sourceWordIndex];
        asrWords.push({
          ...wordWithTimeOffset(word, offsetMs),
          id: nextWordId++,
          _turn: turnId,
          _streamWordKey: stableWordKey(word, sourceWordIndex),
          _streamWordIndex: sourceWordIndex,
        });
      }
    }
    if (liveTurn.latestTtsWordStream?.words?.length > 0) {
      for (let sourceWordIndex = 0; sourceWordIndex < liveTurn.latestTtsWordStream.words.length; sourceWordIndex++) {
        const word = liveTurn.latestTtsWordStream.words[sourceWordIndex];
        ttsWords.push({
          ...word,
          id: nextWordId++,
          _turn: turnId,
          _streamWordKey: stableWordKey(word, sourceWordIndex),
          _streamWordIndex: sourceWordIndex,
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
        label: "Transcript Words",
        stream: {
          id: 1,
          source: "LiveAsr",
          words: asrWords,
        },
      }]
    : [];
  if (ttsWords.length > 0) {
    wordStreamLanes.push({
      label: "TTS",
      stream: {
        id: wordStreamLanes.length + 1,
        source: "SyntheticSpeech",
        words: ttsWords,
      },
    });
  }

  const transcriptEvents = derivedTranscriptEventsFromSession(session);
  const projectedViewerEvents = session.viewerEvents.map((event) => projectLiveViewerEvent(session, event));
  const playbackEvents = projectedViewerEvents.filter((event) => event.kind === "playback_started");

  const asrWordSpans = projectTimedWordsToSpans({
    words: asrWords,
    idPrefix: "asr-word",
    modality: SpanModality.Word,
    stream: "asr_timed_word_stream",
  });
  const ttsWordSpans = projectTimedWordsToSpans({
    words: ttsWords,
    idPrefix: "tts-word",
    modality: SpanModality.Word,
    stream: "tts_timed_word_stream_revision",
  });
  const transcriptSpans = transcriptEvents
    .map((event) =>
      createSpan({
        id: event.id,
        start_ms: event.start_ms,
        end_ms: event.end_ms,
        modality: SpanModality.Transcript,
        metadata: {
          lane: event.lane,
          kind: event.kind,
          turn: event.metadata?.turn ?? null,
          text: event.label ?? null,
          source: event.metadata?.source ?? null,
        },
      }))
    .filter(Boolean);
  const playbackSpans = playbackEvents
    .map((event, index) =>
      createSpan({
        id: event.id ?? `playback-span:${index + 1}`,
        start_ms: event.start_ms,
        end_ms: event.end_ms,
        modality: SpanModality.Playback,
        metadata: {
          lane: event.lane,
          kind: event.kind,
          turn: event.metadata?.turn ?? null,
          speech_unit_id: event.metadata?.event?.speech_unit_id ?? event.metadata?.speech_unit_id ?? null,
          text: event.label ?? null,
          source: "playback_started",
        },
      }))
    .filter(Boolean);

  const asrAlignments = [];
  for (const transcriptSpan of transcriptSpans) {
    const turnId = transcriptSpan.metadata?.turn ?? null;
    const turnWordSpans = asrWordSpans.filter((span) => (span.metadata?.turn ?? null) === turnId);
    asrAlignments.push(
      ...alignWordSpansToParentSpan(
        turnWordSpans,
        transcriptSpan,
        AlignmentKind.AlignedTo,
        0.9,
      ),
    );
  }
  const ttsAlignments = [];
  for (const playbackSpan of playbackSpans) {
    const turnId = playbackSpan.metadata?.turn ?? null;
    const turnWordSpans = ttsWordSpans.filter((span) => (span.metadata?.turn ?? null) === turnId);
    ttsAlignments.push(
      ...alignWordSpansToParentSpan(
        turnWordSpans,
        playbackSpan,
        AlignmentKind.PlayedAs,
        0.85,
      ),
    );
  }
  const streamAlignments = [
    ...asrWordSpans.map((span) =>
      createAlignment({
        source: span.id,
        target: "stream:asr",
        kind: AlignmentKind.DerivedFrom,
        confidence: 1,
      })),
    ...ttsWordSpans.map((span) =>
      createAlignment({
        source: span.id,
        target: "stream:tts",
        kind: AlignmentKind.DerivedFrom,
        confidence: 1,
      })),
  ].filter(Boolean);
  const streamSpans = [
    asrWords.length
      ? createSpan({
          id: "stream:asr",
          start_ms: asrWords[0]?.timing?.start_ms ?? 0,
          end_ms: asrWords[asrWords.length - 1]?.timing?.end_ms ?? session.maxElapsedMs,
          modality: SpanModality.Audio,
          metadata: { stream: "asr_timed_word_stream", lane: "Transcript Words" },
        })
      : null,
    ttsWords.length
      ? createSpan({
          id: "stream:tts",
          start_ms: ttsWords[0]?.timing?.start_ms ?? 0,
          end_ms: ttsWords[ttsWords.length - 1]?.timing?.end_ms ?? session.maxElapsedMs,
          modality: SpanModality.Audio,
          metadata: { stream: "tts_timed_word_stream_revision", lane: "TTS" },
        })
      : null,
  ].filter(Boolean);

  return {
    title: "Live — Listenbury",
    audio: uiState.liveMode
      ? {
          url: "/api/live-session-audio.wav",
          duration_ms: Math.max(session.maxElapsedMs, 1000),
        }
      : null,
    streams: wordStreamLanes,
    events: [
      ...transcriptEvents,
      ...projectedViewerEvents,
      ...inProgressEvents,
    ],
    markers: session.viewerMarkers,
    shared_span_model: buildSharedSpanModel({
      spans: [
        ...streamSpans,
        ...asrWordSpans,
        ...ttsWordSpans,
        ...transcriptSpans,
        ...playbackSpans,
      ].filter(Boolean),
      alignments: [...asrAlignments, ...ttsAlignments, ...streamAlignments],
    }),
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
    label: semanticSpanLabel(session, event.kind, turn, event.label, event.metadata?.event),
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
  const refined = session.refinedTranscript;

  if (refined?.words?.length) {
    for (const word of refined.words) {
      const selection = transcriptTokenSelectionFromWordMetadata(word);
      tokens.push({
        className: `transcript-token commit-confirmed${word._revisions?.length ? " was-revised" : ""}`,
        text: word.text,
        title: formatRevisionTooltip(word) || "confirmed by broader ASR context",
        selection,
        selected: isTranscriptTokenSelected(selection),
      });
    }
  } else if (refined?.text) {
    tokens.push({
      className: "transcript-token span-state-confirmed",
      text: refined.text,
      title: "confirmed by broader ASR context",
    });
  }

  for (const [, liveTurn] of sortedTurns) {
    const supersededByRefinement =
      refined?.elapsedMs != null &&
      liveTurn.finalTranscriptElapsedMs != null &&
      liveTurn.finalTranscriptElapsedMs <= refined.elapsedMs;
    if (liveTurn.id === 0 || supersededByRefinement) {
      continue;
    }

    if (liveTurn.finalTranscript != null) {
      // Committed turn: use word-level commitment states when available.
      const wordStream = liveTurn.latestWordStream;
      if (wordStream?.words?.length > 0) {
        for (let wordIndex = 0; wordIndex < wordStream.words.length; wordIndex++) {
          const word = wordStream.words[wordIndex];
          const selection = transcriptTokenSelection(liveTurn.id, word, wordIndex, "Transcript Words");
          const commitClass = `commit-${commitmentClass(word.commitment)}`;
          tokens.push({
            className: `transcript-token ${commitClass}${word._revisions?.length ? " was-revised" : ""}`,
            text: word.text,
            title: formatRevisionTooltip(word) || null,
            selection,
            selected: isTranscriptTokenSelected(selection),
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
      const wordStream = liveTurn.latestWordStream;
      if (wordStream?.words?.length > 0) {
        for (let wordIndex = 0; wordIndex < wordStream.words.length; wordIndex++) {
          const word = wordStream.words[wordIndex];
          const selection = transcriptTokenSelection(liveTurn.id, word, wordIndex, "Transcript Words");
          const commitClass = `commit-${commitmentClass(word.commitment)}`;
          tokens.push({
            className: `transcript-token ${commitClass}`,
            text: word.text,
            title: null,
            selection,
            selected: isTranscriptTokenSelected(selection),
          });
        }
      } else {
        // In-progress fallback: stable prefix + unstable tail from transcript_candidate.
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
      }
    } else if (liveTurn.latestWordStream?.words?.length > 0) {
      // Word-stream fallback when no transcript_candidate is available.
      for (let wordIndex = 0; wordIndex < liveTurn.latestWordStream.words.length; wordIndex++) {
        const word = liveTurn.latestWordStream.words[wordIndex];
        const selection = transcriptTokenSelection(liveTurn.id, word, wordIndex, "Transcript Words");
        const commitClass = `commit-${commitmentClass(word.commitment)}`;
        tokens.push({
          className: `transcript-token ${commitClass}`,
          text: word.text,
          title: null,
          selection,
          selected: isTranscriptTokenSelected(selection),
        });
      }
    }
  }
  return tokens;
}

function transcriptTokenSelection(turnId, word, wordIndex, laneLabel) {
  return {
    turn: turnId,
    wordKey: stableWordKey(word, wordIndex),
    wordIndex,
    laneLabel,
  };
}

function transcriptTokenSelectionFromWordMetadata(word) {
  if (word?._turn == null || !word?._streamWordKey || !Number.isFinite(word?._streamWordIndex)) {
    return null;
  }
  return {
    turn: word._turn,
    wordKey: word._streamWordKey,
    wordIndex: word._streamWordIndex,
    laneLabel: "Transcript Words",
  };
}

function selectTranscriptToken(token) {
  const resolved = resolveTranscriptTokenSelection(token.selection);
  if (!resolved) {
    return;
  }
  selectWord(resolved.laneIndex, resolved.wordIndex, false);
  autoplayWordClip(resolved.laneIndex, resolved.wordIndex);
}

function isTranscriptTokenSelected(selection) {
  const resolved = resolveTranscriptTokenSelection(selection);
  return Boolean(
    resolved &&
      state.selectedItem?.type === "word" &&
      state.selectedItem.laneIndex === resolved.laneIndex &&
      state.selectedItem.itemIndex === resolved.wordIndex,
  );
}

function resolveTranscriptTokenSelection(selection) {
  if (!selection) {
    return null;
  }
  for (let laneIndex = 0; laneIndex < state.lanes.length; laneIndex++) {
    const lane = state.lanes[laneIndex];
    if (lane?.type !== "word" || (selection.laneLabel && lane.label !== selection.laneLabel)) {
      continue;
    }
    const words = lane.words ?? [];
    const keyedIndex = words.findIndex(
      (word) =>
        word._turn === selection.turn &&
        word._streamWordKey === selection.wordKey,
    );
    if (keyedIndex !== -1) {
      return { laneIndex, wordIndex: keyedIndex };
    }
    const fallbackIndex = words.findIndex(
      (word) =>
        word._turn === selection.turn &&
        word._streamWordIndex === selection.wordIndex,
    );
    if (fallbackIndex !== -1) {
      return { laneIndex, wordIndex: fallbackIndex };
    }
  }
  return null;
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

function updateGeneratedSpeechText(liveTurn, event) {
  if (!event.text || !isGeneratedSpeechEventKind(event.kind)) {
    return;
  }
  const speechUnitId = speechUnitIdFromEvent(event);
  if (speechUnitId) {
    liveTurn.speechUnitsById.set(speechUnitId, event.text);
  }

  if (event.kind === "tts_enqueue_started" || event.kind === "speech_unit_committed") {
    if (speechUnitId) {
      if (!liveTurn.generatedSpeechUnitOrder.includes(speechUnitId)) {
        liveTurn.generatedSpeechUnitOrder.push(speechUnitId);
      }
      const fragments = liveTurn.generatedSpeechUnitOrder
        .map((id) => liveTurn.speechUnitsById.get(id))
        .filter((text) => typeof text === "string" && text.trim());
      liveTurn.generatedSpeechFragments = fragments;
      liveTurn.generatedText = fragments.join(" ");
      return;
    }
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
      const speechUnitId =
        openSpanSpeechUnitId(openSpan) ?? speechUnitIdFromEvent(sourceEvent);
      if (speechUnitId) {
        const speechUnitText = liveTurn?.speechUnitsById?.get(speechUnitId);
        if (speechUnitText) {
          return speechUnitText;
        }
      }
      return liveTurn?.generatedText ?? fallback;
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

// Null-returning wrappers over shared text utilities.
// app.js uses these in `??` chains where an absent value must short-circuit
// to the next candidate; the shared versions return "" for absent.
function transcriptCandidateText(candidate) {
  return normalizeSemanticText(_transcriptCandidateText(candidate));
}

function wordStreamText(words) {
  return normalizeSemanticText(_wordStreamText(words));
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

function reprojectWordTimingAgainstWaveform() {
  if (!state.payload) {
    return;
  }
  const previousSelection = state.selectedItem;
  state.lanes = buildLanes(state.payload);
  state.selectedItem = validSelection(previousSelection) ? previousSelection : firstItemSelection();
  syncMaxDurationWithAudio();
}

function normalizePayload(rawPayload) {
  const sharedSpanProjection = projectSharedSpanModel(rawPayload?.shared_span_model);
  const hasExplicitTimelinePayload =
    Array.isArray(rawPayload?.streams) || Array.isArray(rawPayload?.events) || Array.isArray(rawPayload?.markers);
  if (
    rawPayload &&
    (hasExplicitTimelinePayload || sharedSpanProjection)
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
      : sharedSpanProjection?.streams ?? [];

    return {
      title: rawPayload.title ?? VIEWER_NAME,
      audio: rawPayload.audio ?? null,
      streams,
      events: normalizeEvents(
        rawPayload.events ?? sharedSpanProjection?.events ?? [],
        rawPayload.markers ?? sharedSpanProjection?.markers ?? [],
      ),
    };
  }

  throw new Error("payload must be an object with streams/events");
}

function projectSharedSpanModel(sharedSpanModel) {
  if (!sharedSpanModel || !Array.isArray(sharedSpanModel.spans)) {
    return null;
  }
  const streamsByLabel = new Map();
  const events = [];

  for (const span of sharedSpanModel.spans) {
    const startMs = Number(span?.start_ms);
    const endMs = Number(span?.end_ms);
    if (!Number.isFinite(startMs) || !Number.isFinite(endMs)) {
      continue;
    }
    const modality = String(span?.modality ?? "");
    const lane = span?.metadata?.lane ?? (modality || "Events");
    if (modality === SpanModality.Word) {
      const streamLabel = streamLabelForWordSpan(lane, span?.metadata?.stream);
      if (!streamsByLabel.has(streamLabel)) {
        streamsByLabel.set(streamLabel, []);
      }
      streamsByLabel.get(streamLabel).push({
        id: span?.metadata?.word_id ?? streamsByLabel.get(streamLabel).length + 1,
        text: span?.metadata?.text ?? "",
        timing: {
          start_ms: Math.max(0, Math.round(startMs)),
          end_ms: Math.max(Math.round(startMs) + 1, Math.round(endMs)),
        },
        commitment: "StableText",
      });
      continue;
    }
    events.push({
      id: span.id,
      lane,
      kind: span?.metadata?.kind ?? String(modality || "span").toLowerCase(),
      label: span?.metadata?.text ?? span?.id ?? "span",
      start_ms: Math.max(0, Math.round(startMs)),
      end_ms: Math.max(Math.round(startMs) + 1, Math.round(endMs)),
      metadata: {
        ...(span?.metadata ?? {}),
        shared_span_id: span?.id ?? null,
      },
    });
  }

  const streams = [...streamsByLabel.entries()].map(([label, words], index) => ({
    label,
    stream: {
      id: index + 1,
      source: label === "Transcript Words" ? "LiveAsr" : "SyntheticSpeech",
      words,
    },
  }));
  return { streams, events, markers: [] };
}

function streamLabelForWordSpan(lane, streamName) {
  if (lane === "ASR" || lane === "Transcript Words" || streamName === "asr_timed_word_stream") {
    return "Transcript Words";
  }
  if (lane === "TTS" || streamName === "tts_timed_word_stream_revision") {
    return "TTS";
  }
  return "Word";
}

function normalizeWordLane(lane) {
  const totalWords = lane.stream.words.length || 1;
  const baseWords = lane.stream.words.map((word, wordIndex) => {
    const hasMeasuredTiming = Boolean(word.timing);
    const timing = hasMeasuredTiming
      ? word.timing
      : fallbackTiming(totalWords, wordIndex, inferPayloadDuration(lane.stream));
    return {
      ...word,
      whisperTiming: hasMeasuredTiming ? word.timing : null,
      energyTiming: null,
      energySnapConfidence: null,
      resolvedTiming: timing,
      timingResolution: hasMeasuredTiming ? "word.timing" : "fallback-layout",
      timingSourceDetail: describeTimingSource(
        {
          ...word,
          timingResolution: hasMeasuredTiming ? "word.timing" : "fallback-layout",
          energyTiming: null,
        },
        hasMeasuredTiming,
      ),
    };
  });

  const canRefineAgainstEnergy =
    lane.stream?.source === "RecordedAudio" &&
    state.waveform.status === "ready" &&
    state.waveform.energyEnvelope?.frames?.length;
  if (!canRefineAgainstEnergy) {
    return {
      ...lane,
      type: "word",
      words: baseWords,
    };
  }

  const refinements = refineWordTimingsWithEnergy(
    lane.stream.words,
    state.waveform.energyEnvelope,
    state.waveform.energyLandmarks,
  );
  const words = baseWords.map((word, wordIndex) => {
    const refinement = refinements[wordIndex];
    if (!refinement?.whisperTiming) {
      return word;
    }
    const next = {
      ...word,
      whisperTiming: refinement.whisperTiming,
      energyTiming: refinement.energyTiming,
      energySnapConfidence: refinement.energySnapConfidence ?? null,
      resolvedTiming: refinement.resolvedTiming ?? refinement.whisperTiming,
      timingResolution: refinement.timingResolution ?? "word.timing",
    };
    next.timingSourceDetail = describeTimingSource(next, true);
    return next;
  });

  return {
    ...lane,
    type: "word",
    words,
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
    const laneLabel = displayLaneLabel(event.lane ?? "Events");
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
  return displayLaneLabel(explicitLane ?? "Events");
}

function displayLaneLabel(label) {
  switch (label) {
    case "ASR":
      return "ASR Events";
    case "Mic":
    case "MIC":
      return "Mic / VAD";
    default:
      return label;
  }
}

function inferPayloadDuration(stream) {
  const timedEnd = Math.max(0, ...stream.words.map((word) => word.timing?.end_ms ?? 0));
  return timedEnd || 1000;
}

function configureAudio(audioConfig) {
  if (!audioConfig?.url) {
    clearLiveWaveformRetry();
    state.waveform = {
      url: null,
      peaks: null,
      durationMs: 0,
      status: "idle",
      energyEnvelope: null,
      energyLandmarks: null,
    };
    return;
  }

  const audioUrl = new URL(audioConfig.url, window.location.href);
  if (canonicalAudioUrl(audio.src) !== audioUrl.href) {
    audio.src = audioUrl.href;
  }

  const isLiveAudio = uiState.liveMode && audioUrl.pathname === "/api/live-session-audio.wav";
  if (!isLiveAudio) {
    void loadWaveform(audioConfig.url);
    return;
  }

  const expectedDurationMs = audioConfig.duration_ms ?? 0;
  const now = Date.now();
  const needsInitialLoad = state.waveform.url !== audioConfig.url || state.waveform.status === "idle";
  const needsErrorRetry = state.waveform.status === "error" && now - lastLiveWaveformRefreshAt >= LIVE_WAVEFORM_REFRESH_MS;
  const needsGrowthRefresh =
    state.waveform.status === "ready" &&
    expectedDurationMs > state.waveform.durationMs + LIVE_WAVEFORM_GROWTH_REFRESH_MS &&
    now - lastLiveWaveformRefreshAt >= LIVE_WAVEFORM_REFRESH_MS;

  if (needsInitialLoad || needsErrorRetry || needsGrowthRefresh) {
    lastLiveWaveformRefreshAt = now;
    const fetchUrl = new URL(audioConfig.url, window.location.href);
    fetchUrl.searchParams.set("waveform_rev", String(now));
    void loadWaveform(audioConfig.url, { force: true, fetchUrl: fetchUrl.href });
  } else if (state.waveform.status === "error") {
    scheduleLiveWaveformRetry();
  }
}

function canonicalAudioUrl(url) {
  if (!url) {
    return "";
  }
  const parsed = new URL(url, window.location.href);
  if (parsed.pathname === "/api/live-session-audio.wav") {
    parsed.search = "";
  }
  return parsed.href;
}

function liveAudioPlaybackUrl() {
  const url = new URL("/api/live-session-audio.wav", window.location.href);
  url.searchParams.set("audio_rev", String(Date.now()));
  return url.href;
}

function audioDurationMs() {
  return Number.isFinite(audio.duration) ? Math.round(audio.duration * 1000) : 0;
}

async function seekSessionAudioToMs(startMs, options = {}) {
  if (!audio.src) {
    return false;
  }

  const targetMs = Math.max(0, startMs);
  const stopAtMs = options.stopAtMs ?? null;
  const autoplay = options.autoplay ?? false;
  audio.pause();
  clearPlaybackStop();

  const currentAudioUrl = new URL(audio.src, window.location.href);
  const isLiveAudio = currentAudioUrl.pathname === "/api/live-session-audio.wav";
  const needsFreshLiveSnapshot = isLiveAudio && audioDurationMs() < targetMs + 50;
  if (needsFreshLiveSnapshot) {
    const loaded = waitForAudioEvent(["loadedmetadata", "canplay", "canplaythrough"], 2_000);
    audio.src = liveAudioPlaybackUrl();
    audio.load();
    await loaded;
  } else if (audio.readyState < 1) {
    await waitForAudioEvent(["loadedmetadata", "canplay", "canplaythrough"], 2_000);
  }

  if (audioDurationMs() > 0 && targetMs > audioDurationMs() + 50) {
    uiState.statusMessage = "Audio for that word is still loading.";
    renderShell();
    return false;
  }

  const targetSeconds = targetMs / 1000;
  try {
    if (typeof audio.fastSeek === "function") {
      audio.fastSeek(targetSeconds);
    } else {
      audio.currentTime = targetSeconds;
    }
  } catch (err) {
    console.warn("Unable to seek session audio:", err);
    return false;
  }

  const didSeek = await waitForAudioSeek(targetSeconds, 1_000);
  if (!didSeek) {
    console.warn(
      `Session audio did not seek to ${targetSeconds.toFixed(3)}s; current time is ${audio.currentTime.toFixed(3)}s`,
    );
    return false;
  }

  if (stopAtMs !== null) {
    setPlaybackStop(targetMs, stopAtMs);
  }

  if (autoplay) {
    try {
      await audio.play();
    } catch (err) {
      console.warn("Unable to play session audio:", err);
    }
  }
  refreshPlaybackState();
  return true;
}

function waitForAudioEvent(eventNames, timeoutMs) {
  return new Promise((resolve) => {
    let settled = false;
    const cleanup = () => {
      for (const name of eventNames) {
        audio.removeEventListener(name, onEvent);
      }
      window.clearTimeout(timer);
    };
    const finish = () => {
      if (settled) {
        return;
      }
      settled = true;
      cleanup();
      resolve();
    };
    const onEvent = () => finish();
    for (const name of eventNames) {
      audio.addEventListener(name, onEvent, { once: true });
    }
    const timer = window.setTimeout(finish, timeoutMs);
  });
}

function waitForAudioSeek(targetSeconds, timeoutMs) {
  if (Math.abs(audio.currentTime - targetSeconds) <= 0.02 && !audio.seeking) {
    return Promise.resolve(true);
  }
  return new Promise((resolve) => {
    let settled = false;
    const cleanup = () => {
      audio.removeEventListener("seeked", onEvent);
      window.clearTimeout(timer);
    };
    const finish = (didSeek) => {
      if (settled) {
        return;
      }
      settled = true;
      cleanup();
      resolve(didSeek);
    };
    const onEvent = () => {
      if (Math.abs(audio.currentTime - targetSeconds) <= 0.05) {
        finish(true);
      }
    };
    audio.addEventListener("seeked", onEvent);
    const timer = window.setTimeout(() => {
      finish(Math.abs(audio.currentTime - targetSeconds) <= 0.05);
    }, timeoutMs);
  });
}

async function loadWaveform(url, options = {}) {
  const { force = false, fetchUrl = url } = options;
  if (
    !url ||
    (!force &&
      state.waveform.url === url &&
      (state.waveform.status === "ready" || state.waveform.status === "loading"))
  ) {
    return;
  }
  if (force && state.waveform.url === url && state.waveform.status === "loading") {
    return;
  }
  if (force && waveformRefreshInFlightUrl === url) {
    return;
  }
  const previousWaveform = state.waveform;
  const preserveCurrentWaveform =
    force &&
    previousWaveform.url === url &&
    previousWaveform.status === "ready" &&
    previousWaveform.peaks?.length;
  if (force) {
    waveformRefreshInFlightUrl = url;
  }
  if (!preserveCurrentWaveform) {
    state.waveform = {
      url,
      peaks: null,
      durationMs: 0,
      status: "loading",
      energyEnvelope: null,
      energyLandmarks: null,
    };
  }
  try {
    const response = await fetch(fetchUrl, { cache: "no-store" });
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
    const energyEnvelope = buildEnergyEnvelopeFromAudioBuffer(audioBuffer);
    const energyLandmarks = detectEnergyLandmarks(energyEnvelope);
    const peaks = waveformPeaksFromAudioBuffer(audioBuffer);
    const durationMs = Math.round(audioBuffer.duration * 1000);
    const waveformUnchanged =
      preserveCurrentWaveform &&
      previousWaveform.durationMs === durationMs &&
      waveformPeaksEqual(previousWaveform.peaks, peaks);
    state.waveform = waveformUnchanged ? previousWaveform : {
      url,
      peaks,
      durationMs,
      status: "ready",
      energyEnvelope,
      energyLandmarks,
    };
    clearLiveWaveformRetry();
    if (!waveformUnchanged) {
      reprojectWordTimingAgainstWaveform();
      // Re-render full shell so inspector/selection metadata reflects updated resolved timings.
      render();
    }
  } catch (err) {
    console.warn("Unable to build waveform overlay:", err);
    if (!preserveCurrentWaveform) {
      state.waveform = {
        url,
        peaks: null,
        durationMs: 0,
        status: "error",
        energyEnvelope: null,
        energyLandmarks: null,
      };
    }
    const failedUrl = new URL(url, window.location.href);
    if (uiState.liveMode && failedUrl.pathname === "/api/live-session-audio.wav") {
      scheduleLiveWaveformRetry();
    }
  } finally {
    if (force && waveformRefreshInFlightUrl === url) {
      waveformRefreshInFlightUrl = null;
    }
  }
}

function scheduleLiveWaveformRetry() {
  if (liveWaveformRetryTimer !== null) {
    return;
  }
  liveWaveformRetryTimer = window.setTimeout(() => {
    liveWaveformRetryTimer = null;
    const audioConfig = state.payload?.audio;
    if (uiState.liveMode && audioConfig?.url) {
      configureAudio(audioConfig);
    }
  }, LIVE_WAVEFORM_REFRESH_MS);
}

function clearLiveWaveformRetry() {
  if (liveWaveformRetryTimer === null) {
    return;
  }
  window.clearTimeout(liveWaveformRetryTimer);
  liveWaveformRetryTimer = null;
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
  if (state.surfaceMode === "graph") {
    renderGraphInspectionMode();
  } else {
    renderCustomTimeline();
  }
  renderSelection();
  renderShell();
}

function setSurfaceMode(mode) {
  if (mode !== "timeline" && mode !== "graph") {
    return;
  }
  if (state.surfaceMode === mode) {
    return;
  }
  state.surfaceMode = mode;
  render();
}

function toggleDiagnostics() {
  uiState.diagnosticsExpanded = !uiState.diagnosticsExpanded;
  render();
}

// ── Custom timeline renderer ───────────────────────────────────────────────

let _programmaticScroll = false;

function pxPerMs() {
  return currentTimeScale().pxPerMs;
}

function pxForMs(ms) {
  return currentTimeScale().msToPx(ms);
}

function msForPx(px) {
  return currentTimeScale().pxToMs(px);
}

function currentTimeScale() {
  return createTimeScale({
    pxPerSecond: state.zoomPxPerSecond,
    durationMs: state.maxDurationMs,
  });
}

function currentTimelineViewport() {
  const col = getScrollContainer();
  return createTimelineViewport({
    pxPerSecond: state.zoomPxPerSecond,
    durationMs: state.maxDurationMs,
    viewportWidthPx: col ? col.clientWidth : 600,
    scrollLeftPx: col ? col.scrollLeft : 0,
  });
}

function intervalToCss(startMs, endMs, minWidthPx = 0) {
  const { leftPx, widthPx } = currentTimeScale().intervalToPx({ startMs, endMs, minWidthPx });
  return { left: `${leftPx}px`, width: `${widthPx}px`, widthPx };
}

function clampZoom(pxPerSec) {
  return Math.max(MIN_ZOOM_PX_PER_SECOND, Math.min(MAX_ZOOM_PX_PER_SECOND, pxPerSec));
}

function getScrollContainer() {
  return document.getElementById("timeline-tracks-col");
}

function getTrackContentWidth() {
  return currentTimelineViewport().contentWidthPx();
}

function getScrollViewport() {
  return currentTimelineViewport().visibleRangeMs({ minDurationMs: MIN_VIEW_DURATION_MS });
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
  const reusableCentralWaveformCanvas = document.getElementById("central-waveform-canvas");

  labelsCol.innerHTML = "";
  scrollContent.innerHTML = "";
  scrollContent.style.width = `${trackContentWidth}px`;

  state.chipElementByKey = new Map();
  state.itemTimingByKey = new Map();
  state.waveformChipElements = new Map();
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

  // Central waveform/oscilloscope panel — shared timebase anchor for all lanes
  appendCentralWaveformPanel(labelsCol, scrollContent, trackContentWidth, nowMs, {
    reusableCanvas: reusableCentralWaveformCanvas,
  });

  const eventLanes = state.lanes.filter((lane) => lane.type === "event");
  if (eventLanes.length > 0) {
    appendDiagnosticsHeader(labelsCol, scrollContent, trackContentWidth, eventLanes);
  }

  // Lane rows
  state.lanes.forEach((lane, laneIndex) => {
    if (lane.type === "word") {
      return;
    }
    if (lane.type === "event" && !uiState.diagnosticsExpanded) {
      return;
    }
    // Label entry
    const labelEntryEl = document.createElement("div");
    labelEntryEl.className = `lane-label-entry event-lane diagnostic-lane-label`;
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
      "event-track",
      "diagnostic-track",
      `lane-${classToken(lane.label)}`,
    ]
      .filter(Boolean)
      .join(" ");
    trackEl.dataset.laneIndex = String(laneIndex);

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
        const wordCss = intervalToCss(startMs, endMs, 2);
        const widthPx = wordCss.widthPx;
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

        const chip = document.createElement("button");
        chip.type = "button";
        chip.className = [baseClass, isActive ? "active" : "", isSelected ? "selected" : ""]
          .filter(Boolean)
          .join(" ");
        chip.dataset.baseClass = baseClass;
        chip.dataset.itemKey = key;
        chip.dataset.laneIndex = String(laneIndex);
        chip.dataset.itemIndex = String(wordIndex);
        chip.dataset.itemType = "word";
        chip.style.left = wordCss.left;
        chip.style.width = wordCss.width;
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
        const isRelated = isRelatedDiagnosticEvent("event", laneIndex, eventIndex, {
          startMs: visualStartMs,
          endMs: visualEndMs,
        });

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

        const chip = document.createElement("button");
        chip.type = "button";
        chip.className = [
          baseClass,
          isRelated ? "related-to-selection" : "",
          isActive ? "active" : "",
          isSelected ? "selected" : "",
        ]
          .filter(Boolean)
          .join(" ");
        chip.dataset.baseClass = baseClass;
        chip.dataset.itemKey = key;
        chip.dataset.laneIndex = String(laneIndex);
        chip.dataset.itemIndex = String(eventIndex);
        chip.dataset.itemType = "event";
        const eventCss = intervalToCss(visualStartMs, visualEndMs, isMarker ? 10 : 12);
        chip.style.left = eventCss.left;
        chip.style.width = eventCss.width;
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

// ── Graph inspection mode ───────────────────────────────────────────────────

function renderGraphInspectionMode() {
  const host = ensureGraphModeHost();
  if (viewer.firstElementChild !== host) {
    viewer.replaceChildren(host);
  }
  bindGraphControls();
  const model = buildAlignmentGraphModel();
  syncGraphControlOptions(model);
  renderGraphSummary(model);
  const cy = ensureAlignmentGraph();
  updateAlignmentGraph(cy, model);
}

function ensureGraphModeHost() {
  let host = document.getElementById("graph-mode-host");
  if (host) {
    return host;
  }
  host = document.createElement("section");
  host.id = "graph-mode-host";
  host.className = "graph-mode-host";
  host.innerHTML = `
    <header class="graph-toolbar">
      <strong>Span / Alignment Graph</strong>
      <span id="graph-summary" class="graph-summary"></span>
      <label>Modality
        <select id="graph-filter-modality"></select>
      </label>
      <label>Turn
        <select id="graph-filter-turn"></select>
      </label>
      <label>Window
        <select id="graph-filter-window">
          <option value="all">All time</option>
          <option value="viewport">Timeline viewport</option>
          <option value="selection">Selection focus</option>
        </select>
      </label>
      <label>Commitment
        <select id="graph-filter-commitment"></select>
      </label>
      <label class="graph-toggle">
        <input id="graph-filter-revisions" type="checkbox" />
        Revisions only
      </label>
      <label class="graph-toggle">
        <input id="graph-filter-neighborhood" type="checkbox" />
        Focus neighborhood
      </label>
    </header>
    <div id="alignment-graph" class="alignment-graph" role="img" aria-label="Span and alignment graph"></div>
    <p class="graph-help">
      Click a graph node to jump to timeline context. Word revisions render as lineage chains.
    </p>
  `;
  return host;
}

function bindGraphControls() {
  const modality = document.getElementById("graph-filter-modality");
  const turn = document.getElementById("graph-filter-turn");
  const windowFilter = document.getElementById("graph-filter-window");
  const commitment = document.getElementById("graph-filter-commitment");
  const revisions = document.getElementById("graph-filter-revisions");
  const neighborhood = document.getElementById("graph-filter-neighborhood");
  if (!modality || !turn || !windowFilter || !commitment || !revisions || !neighborhood) {
    return;
  }
  modality.onchange = () => {
    state.graphFilters.modality = modality.value;
    renderGraphInspectionMode();
  };
  turn.onchange = () => {
    state.graphFilters.turn = turn.value;
    renderGraphInspectionMode();
  };
  windowFilter.onchange = () => {
    state.graphFilters.timeWindow = windowFilter.value;
    renderGraphInspectionMode();
  };
  commitment.onchange = () => {
    state.graphFilters.commitment = commitment.value;
    renderGraphInspectionMode();
  };
  revisions.onchange = () => {
    state.graphFilters.revisionsOnly = revisions.checked;
    renderGraphInspectionMode();
  };
  neighborhood.onchange = () => {
    state.graphFilters.neighborhood = neighborhood.checked;
    renderGraphInspectionMode();
  };
}

function syncGraphControlOptions(model) {
  syncSelectWithOptions(
    document.getElementById("graph-filter-modality"),
    [{ value: "all", label: "All" }, ...model.modalities.map((entry) => ({ value: entry, label: entry }))],
    state.graphFilters.modality,
  );
  syncSelectWithOptions(
    document.getElementById("graph-filter-turn"),
    [{ value: "all", label: "All" }, ...model.turns.map((entry) => ({ value: String(entry), label: `Turn ${entry}` }))],
    state.graphFilters.turn,
  );
  syncSelectWithOptions(
    document.getElementById("graph-filter-commitment"),
    [{ value: "all", label: "All" }, ...model.commitments.map((entry) => ({ value: entry, label: entry }))],
    state.graphFilters.commitment,
  );
  const windowFilter = document.getElementById("graph-filter-window");
  if (windowFilter && windowFilter.value !== state.graphFilters.timeWindow) {
    windowFilter.value = state.graphFilters.timeWindow;
  }
  const revisions = document.getElementById("graph-filter-revisions");
  if (revisions) {
    revisions.checked = state.graphFilters.revisionsOnly;
  }
  const neighborhood = document.getElementById("graph-filter-neighborhood");
  if (neighborhood) {
    neighborhood.checked = state.graphFilters.neighborhood;
  }
}

function syncSelectWithOptions(selectEl, options, currentValue) {
  if (!selectEl) {
    return;
  }
  const rendered = options.map((opt) => `<option value="${escapeHtml(opt.value)}">${escapeHtml(opt.label)}</option>`).join("");
  selectEl.innerHTML = rendered;
  selectEl.value = options.some((opt) => opt.value === currentValue) ? currentValue : options[0]?.value ?? "all";
  if (currentValue !== selectEl.value) {
    if (selectEl.id === "graph-filter-modality") {
      state.graphFilters.modality = selectEl.value;
    } else if (selectEl.id === "graph-filter-turn") {
      state.graphFilters.turn = selectEl.value;
    } else if (selectEl.id === "graph-filter-commitment") {
      state.graphFilters.commitment = selectEl.value;
    }
  }
}

function renderGraphSummary(model) {
  const summary = document.getElementById("graph-summary");
  if (!summary) {
    return;
  }
  summary.textContent = `${model.nodes.length} nodes · ${model.edges.length} edges`;
}

function ensureAlignmentGraph() {
  const container = document.getElementById("alignment-graph");
  if (!container) {
    return null;
  }
  if (graphState.cy) {
    if (graphState.cy.container() === container) {
      return graphState.cy;
    }
    graphState.cy.destroy();
    graphState.cy = null;
    graphState.bound = false;
  }
  graphState.cy = cytoscape({
    container,
    minZoom: 0.2,
    maxZoom: 3,
    style: [
      { selector: "node", style: { label: "data(label)", color: "#eef3f7", "font-size": 11, "text-wrap": "wrap", "text-max-width": 160, "text-valign": "center", "text-halign": "center", "background-color": "#36414f", width: 28, height: 28 } },
      { selector: "edge", style: { width: 1.4, "line-color": "#5c6b7a", "target-arrow-color": "#5c6b7a", "target-arrow-shape": "triangle", "curve-style": "bezier", label: "data(label)", "font-size": 9, color: "#9dabba", "text-background-color": "#151a1f", "text-background-opacity": 0.9, "text-background-padding": "2px", "text-rotation": "autorotate" } },
      { selector: "node[nodeType = 'word']", style: { "background-color": "#3d7ca0", shape: "round-rectangle" } },
      { selector: "node[nodeType = 'event']", style: { "background-color": "#697f50" } },
      { selector: "node[nodeType = 'revision']", style: { "background-color": "#9a6e2b", shape: "diamond" } },
      { selector: "node[nodeType = 'audio']", style: { "background-color": "#4b5a8f", shape: "hexagon" } },
      { selector: "node[nodeType = 'turn']", style: { "background-color": "#6b4a89", shape: "round-hexagon" } },
      { selector: "node[nodeType = 'lexical']", style: { "background-color": "#2b7f72", shape: "tag" } },
      { selector: "node[nodeType = 'phoneme']", style: { "background-color": "#7d3c74", shape: "rectangle" } },
      { selector: "edge[edgeType = 'revision']", style: { "line-color": "#ffd166", "target-arrow-color": "#ffd166" } },
      { selector: "edge[edgeType = 'alignment']", style: { "line-style": "dashed", "line-color": "#63d2ff", "target-arrow-color": "#63d2ff" } },
      { selector: "edge[edgeType = 'contains']", style: { "line-color": "#8ee6a8", "target-arrow-color": "#8ee6a8" } },
      { selector: ".selected", style: { "border-width": 2, "border-color": "#ff7a90" } },
      { selector: ".related", style: { "line-color": "#ff7a90", "target-arrow-color": "#ff7a90", width: 2.1 } },
      { selector: ".faded", style: { opacity: 0.28 } },
    ],
    elements: [],
  });
  if (!graphState.bound) {
    graphState.bound = true;
    graphState.cy.on("tap", "node", (event) => {
      focusFromGraphNode(event.target);
    });
  }
  return graphState.cy;
}

function updateAlignmentGraph(cy, model) {
  if (!cy) {
    return;
  }
  const nextElements = [
    ...model.nodes.map((node) => ({ data: node })),
    ...model.edges.map((edge) => ({ data: edge })),
  ];
  const nextIds = new Set(nextElements.map((element) => element.data.id));
  cy.elements().forEach((element) => {
    if (!nextIds.has(element.id())) {
      element.remove();
    }
  });
  const existing = new Set(cy.elements().map((element) => element.id()));
  for (const element of nextElements) {
    if (existing.has(element.data.id)) {
      const current = cy.getElementById(element.data.id);
      current.data(element.data);
    } else {
      cy.add(element);
    }
  }
  cy.elements().removeClass("selected related faded");
  const selectedId = selectedGraphNodeId();
  if (selectedId) {
    const selected = cy.getElementById(selectedId);
    if (selected.nonempty()) {
      selected.addClass("selected");
      const neighborhood = selected.closedNeighborhood();
      neighborhood.connectedEdges().addClass("related");
      cy.elements().difference(neighborhood).addClass("faded");
    }
  }
  const layout = cy.layout({
    name: model.focusNodeId ? "breadthfirst" : "cose",
    animate: false,
    fit: true,
    padding: 24,
    roots: model.focusNodeId ? [model.focusNodeId] : undefined,
    nodeDimensionsIncludeLabels: true,
  });
  layout.run();
}

function focusFromGraphNode(node) {
  const selectionType = node.data("selectionType");
  const laneIndex = parseInt(node.data("laneIndex"), 10);
  const itemIndex = parseInt(node.data("itemIndex"), 10);
  if (selectionType === "word" && Number.isFinite(laneIndex) && Number.isFinite(itemIndex)) {
    state.surfaceMode = "timeline";
    render();
    selectWord(laneIndex, itemIndex, true);
    return;
  }
  if (selectionType === "event" && Number.isFinite(laneIndex) && Number.isFinite(itemIndex)) {
    state.surfaceMode = "timeline";
    render();
    selectEvent(laneIndex, itemIndex, true);
    return;
  }
  const turn = node.data("turn");
  if (Number.isFinite(turn)) {
    state.graphFilters.turn = String(turn);
    renderGraphInspectionMode();
  }
}

function buildAlignmentGraphModel() {
  const modalities = new Set();
  const turns = new Set();
  const commitments = new Set();
  const nodes = [];
  const edges = [];
  const nodeIds = new Set();
  const edgeIds = new Set();
  const words = [];
  const events = [];
  const focusNodeId = selectedGraphNodeId();
  const windowSelection = graphTimeWindow();

  state.lanes.forEach((lane, laneIndex) => {
    modalities.add(lane.label);
    if (lane.type === "word") {
      lane.words.forEach((word, wordIndex) => {
        const turn = Number.isFinite(word._turn) ? word._turn : null;
        if (turn !== null) turns.add(turn);
        if (word.commitment) commitments.add(word.commitment);
        words.push({ lane, laneIndex, word, wordIndex, turn });
      });
      return;
    }
    lane.events.forEach((event, eventIndex) => {
      const turn = Number.isFinite(event.metadata?.turn) ? event.metadata.turn : null;
      if (turn !== null) turns.add(turn);
      events.push({ lane, laneIndex, event, eventIndex, turn });
    });
  });

  const passesWindow = (startMs, endMs) => {
    if (!windowSelection) {
      return true;
    }
    return endMs >= windowSelection.startMs && startMs <= windowSelection.endMs;
  };

  for (const item of words) {
    const startMs = item.word.resolvedTiming.start_ms;
    const endMs = Math.max(item.word.resolvedTiming.end_ms, startMs + 1);
    if (!passesWindow(startMs, endMs)) {
      continue;
    }
    if (state.graphFilters.modality !== "all" && state.graphFilters.modality !== item.lane.label) {
      continue;
    }
    if (state.graphFilters.turn !== "all" && String(item.turn ?? "none") !== state.graphFilters.turn) {
      continue;
    }
    if (state.graphFilters.commitment !== "all" && item.word.commitment !== state.graphFilters.commitment) {
      continue;
    }
    if (state.graphFilters.revisionsOnly && !(item.word._revisions?.length > 0)) {
      continue;
    }
    addGraphWord(nodes, edges, nodeIds, edgeIds, item);
  }

  for (const item of events) {
    const startMs = item.event.start_ms;
    const endMs = Math.max(item.event.end_ms, startMs + 1);
    if (!passesWindow(startMs, endMs)) {
      continue;
    }
    if (state.graphFilters.modality !== "all" && state.graphFilters.modality !== item.lane.label) {
      continue;
    }
    if (state.graphFilters.turn !== "all" && String(item.turn ?? "none") !== state.graphFilters.turn) {
      continue;
    }
    if (state.graphFilters.revisionsOnly) {
      continue;
    }
    addGraphEvent(nodes, edges, nodeIds, edgeIds, item);
  }

  addOverlapEdges(edges, edgeIds, words, events, nodeIds, passesWindow);
  let filtered = { nodes, edges };
  if (state.graphFilters.neighborhood && focusNodeId) {
    filtered = graphNeighborhood(nodes, edges, focusNodeId);
  }
  filtered = capGraphSize(filtered.nodes, filtered.edges, focusNodeId, windowSelection);
  return {
    nodes: filtered.nodes,
    edges: filtered.edges,
    modalities: [...modalities].sort(),
    turns: [...turns].sort((left, right) => left - right),
    commitments: [...commitments].sort(),
    focusNodeId,
  };
}

function addGraphWord(nodes, edges, nodeIds, edgeIds, item) {
  const wordNodeId = `word:${item.laneIndex}:${item.wordIndex}`;
  const wordLabel = item.word.text || "(word)";
  pushGraphNode(nodes, nodeIds, {
    id: wordNodeId,
    label: wordLabel,
    nodeType: "word",
    modality: item.lane.label,
    turn: item.turn,
    commitment: item.word.commitment ?? null,
    selectionType: "word",
    laneIndex: item.laneIndex,
    itemIndex: item.wordIndex,
    startMs: item.word.resolvedTiming.start_ms,
    endMs: item.word.resolvedTiming.end_ms,
  });
  if (item.turn !== null) {
    const turnNodeId = `turn:${item.turn}`;
    pushGraphNode(nodes, nodeIds, {
      id: turnNodeId,
      label: `Turn ${item.turn}`,
      nodeType: "turn",
      turn: item.turn,
      selectionType: "turn",
    });
    pushGraphEdge(edges, edgeIds, {
      id: `edge:${turnNodeId}:${wordNodeId}:contains`,
      source: turnNodeId,
      target: wordNodeId,
      edgeType: "contains",
      label: "contains",
    });
  }
  if (item.word.audio_ref?.url) {
    const startMs = item.word.audio_ref.start_ms ?? item.word.resolvedTiming.start_ms;
    const endMs = item.word.audio_ref.end_ms ?? item.word.resolvedTiming.end_ms;
    const audioNodeId = `audio:${item.word.audio_ref.url}:${startMs}:${endMs}`;
    pushGraphNode(nodes, nodeIds, {
      id: audioNodeId,
      label: `Audio ${formatRulerLabel(startMs)}–${formatRulerLabel(endMs)}`,
      nodeType: "audio",
      modality: item.lane.label,
      turn: item.turn,
    });
    pushGraphEdge(edges, edgeIds, {
      id: `edge:${wordNodeId}:${audioNodeId}:alignment`,
      source: wordNodeId,
      target: audioNodeId,
      edgeType: "alignment",
      label: "word↔audio",
    });
  }
  if (item.word.lexical_span) {
    const lexicalNodeId = `lexical:${item.word.lexical_span.start}:${item.word.lexical_span.end}`;
    pushGraphNode(nodes, nodeIds, {
      id: lexicalNodeId,
      label: `Text ${item.word.lexical_span.start}:${item.word.lexical_span.end}`,
      nodeType: "lexical",
      modality: item.lane.label,
      turn: item.turn,
    });
    pushGraphEdge(edges, edgeIds, {
      id: `edge:${lexicalNodeId}:${wordNodeId}:contains`,
      source: lexicalNodeId,
      target: wordNodeId,
      edgeType: "contains",
      label: "contains",
    });
  }
  const phonemes = Array.isArray(item.word.phonemes) ? item.word.phonemes : [];
  phonemes.forEach((phoneme, index) => {
    const phonemeNodeId = `phoneme:${item.laneIndex}:${item.wordIndex}:${index}`;
    pushGraphNode(nodes, nodeIds, {
      id: phonemeNodeId,
      label: String(phoneme?.label ?? phoneme?.text ?? phoneme ?? "phoneme"),
      nodeType: "phoneme",
      modality: item.lane.label,
      turn: item.turn,
    });
    pushGraphEdge(edges, edgeIds, {
      id: `edge:${phonemeNodeId}:${wordNodeId}:alignment`,
      source: phonemeNodeId,
      target: wordNodeId,
      edgeType: "alignment",
      label: "phoneme↔word",
    });
  });
  const revisions = item.word._revisions ?? [];
  let revisionSourceId = null;
  revisions.forEach((revision, index) => {
    const revisionNodeId = `revision:${item.laneIndex}:${item.wordIndex}:${index}`;
    pushGraphNode(nodes, nodeIds, {
      id: revisionNodeId,
      label: revision.fromText ?? "revised",
      nodeType: "revision",
      modality: item.lane.label,
      turn: item.turn,
      atMs: revision.at_ms,
    });
    if (revisionSourceId) {
      pushGraphEdge(edges, edgeIds, {
        id: `edge:${revisionSourceId}:${revisionNodeId}:revision`,
        source: revisionSourceId,
        target: revisionNodeId,
        edgeType: "revision",
        label: "revision",
      });
    }
    revisionSourceId = revisionNodeId;
  });
  if (revisionSourceId) {
    pushGraphEdge(edges, edgeIds, {
      id: `edge:${revisionSourceId}:${wordNodeId}:revision`,
      source: revisionSourceId,
      target: wordNodeId,
      edgeType: "revision",
      label: "revision",
    });
  }
}

function addGraphEvent(nodes, edges, nodeIds, edgeIds, item) {
  const eventNodeId = `event:${item.laneIndex}:${item.eventIndex}`;
  pushGraphNode(nodes, nodeIds, {
    id: eventNodeId,
    label: item.event.label || item.event.kind,
    nodeType: "event",
    modality: item.lane.label,
    turn: item.turn,
    selectionType: "event",
    laneIndex: item.laneIndex,
    itemIndex: item.eventIndex,
    startMs: item.event.start_ms,
    endMs: item.event.end_ms,
  });
  if (item.turn !== null) {
    const turnNodeId = `turn:${item.turn}`;
    pushGraphNode(nodes, nodeIds, {
      id: turnNodeId,
      label: `Turn ${item.turn}`,
      nodeType: "turn",
      turn: item.turn,
      selectionType: "turn",
    });
    pushGraphEdge(edges, edgeIds, {
      id: `edge:${turnNodeId}:${eventNodeId}:contains`,
      source: turnNodeId,
      target: eventNodeId,
      edgeType: "contains",
      label: "contains",
    });
  }
}

function addOverlapEdges(edges, edgeIds, words, events, nodeIds, passesWindow) {
  const activeWords = words.filter((item) => {
    const startMs = item.word.resolvedTiming.start_ms;
    const endMs = Math.max(item.word.resolvedTiming.end_ms, startMs + 1);
    return passesWindow(startMs, endMs);
  });
  const activeEvents = events.filter((item) => {
    const startMs = item.event.start_ms;
    const endMs = Math.max(item.event.end_ms, startMs + 1);
    return passesWindow(startMs, endMs);
  });
  for (const word of activeWords) {
    const wordNodeId = `word:${word.laneIndex}:${word.wordIndex}`;
    if (!nodeIds.has(wordNodeId)) {
      continue;
    }
    for (const event of activeEvents) {
      const eventNodeId = `event:${event.laneIndex}:${event.eventIndex}`;
      if (!nodeIds.has(eventNodeId)) {
        continue;
      }
      if (word.turn !== null && event.turn !== null && word.turn !== event.turn) {
        continue;
      }
      const overlap = rangesOverlap(
        word.word.resolvedTiming.start_ms,
        word.word.resolvedTiming.end_ms,
        event.event.start_ms,
        event.event.end_ms,
      );
      if (!overlap) {
        continue;
      }
      pushGraphEdge(edges, edgeIds, {
        id: `edge:${wordNodeId}:${eventNodeId}:overlap`,
        source: wordNodeId,
        target: eventNodeId,
        edgeType: "alignment",
        label: "overlap",
      });
    }
  }
}

function rangesOverlap(leftStart, leftEnd, rightStart, rightEnd) {
  return Math.max(leftStart, rightStart) <= Math.min(leftEnd, rightEnd);
}

function graphTimeWindow() {
  if (state.graphFilters.timeWindow === "selection") {
    return state.brushSelection ?? selectedItemTiming();
  }
  if (state.graphFilters.timeWindow === "viewport") {
    const col = getScrollContainer();
    if (!col) {
      return null;
    }
    return getScrollViewport();
  }
  return null;
}

function graphNeighborhood(nodes, edges, focusNodeId) {
  const adjacency = new Map();
  edges.forEach((edge) => {
    if (!adjacency.has(edge.source)) adjacency.set(edge.source, new Set());
    if (!adjacency.has(edge.target)) adjacency.set(edge.target, new Set());
    adjacency.get(edge.source).add(edge.target);
    adjacency.get(edge.target).add(edge.source);
  });
  const keep = new Set([focusNodeId]);
  const neighbors = adjacency.get(focusNodeId) ?? new Set();
  neighbors.forEach((nodeId) => keep.add(nodeId));
  const filteredNodes = nodes.filter((node) => keep.has(node.id));
  const filteredEdges = edges.filter((edge) => keep.has(edge.source) && keep.has(edge.target));
  return filteredNodes.length ? { nodes: filteredNodes, edges: filteredEdges } : { nodes, edges };
}

function capGraphSize(nodes, edges, focusNodeId, windowSelection) {
  const centerMs = windowSelection ? (windowSelection.startMs + windowSelection.endMs) / 2 : null;
  const rankedNodes = [...nodes].sort((left, right) => {
    if (left.id === focusNodeId) return -1;
    if (right.id === focusNodeId) return 1;
    if (centerMs == null) return 0;
    const leftMid = Number.isFinite(left.startMs) && Number.isFinite(left.endMs) ? (left.startMs + left.endMs) / 2 : centerMs;
    const rightMid = Number.isFinite(right.startMs) && Number.isFinite(right.endMs) ? (right.startMs + right.endMs) / 2 : centerMs;
    return Math.abs(leftMid - centerMs) - Math.abs(rightMid - centerMs);
  });
  const keptNodes = rankedNodes.slice(0, GRAPH_MAX_RENDER_NODES);
  const keptNodeIds = new Set(keptNodes.map((node) => node.id));
  const keptEdges = edges
    .filter((edge) => keptNodeIds.has(edge.source) && keptNodeIds.has(edge.target))
    .slice(0, GRAPH_MAX_RENDER_EDGES);
  return { nodes: keptNodes, edges: keptEdges };
}

function pushGraphNode(nodes, ids, node) {
  if (ids.has(node.id)) {
    return;
  }
  ids.add(node.id);
  nodes.push(node);
}

function pushGraphEdge(edges, ids, edge) {
  if (ids.has(edge.id)) {
    return;
  }
  ids.add(edge.id);
  edges.push(edge);
}

function selectedGraphNodeId() {
  if (!state.selectedItem) {
    return null;
  }
  return `${state.selectedItem.type}:${state.selectedItem.laneIndex}:${state.selectedItem.itemIndex}`;
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

function appendDiagnosticsHeader(labelsCol, scrollContent, trackContentWidth, eventLanes) {
  const labelEl = document.createElement("div");
  labelEl.className = "lane-label-entry diagnostics-section-label";
  const headerEl = document.createElement("div");
  headerEl.className = "lane-header diagnostics-section-header";
  const h2El = document.createElement("h2");
  h2El.textContent = "Diagnostics";
  const metaEl = document.createElement("div");
  metaEl.className = "lane-meta";
  const eventCount = eventLanes.reduce((sum, lane) => sum + lane.events.length, 0);
  metaEl.textContent = `${eventLanes.length} lane${eventLanes.length === 1 ? "" : "s"} · ${eventCount} event${eventCount === 1 ? "" : "s"}`;
  headerEl.append(h2El, metaEl);
  labelEl.append(headerEl);
  labelsCol.append(labelEl);

  const trackEl = document.createElement("div");
  trackEl.className = "lane-track diagnostics-section-track";
  trackEl.style.width = `${trackContentWidth}px`;
  const toggle = document.createElement("button");
  toggle.type = "button";
  toggle.className = "diagnostics-section-toggle";
  toggle.setAttribute("aria-expanded", String(uiState.diagnosticsExpanded));
  toggle.textContent = uiState.diagnosticsExpanded ? "Hide diagnostic event lanes" : "Show diagnostic event lanes";
  toggle.addEventListener("click", (event) => {
    event.stopPropagation();
    toggleDiagnostics();
  });
  trackEl.append(toggle);
  scrollContent.append(trackEl);
}

function waveformCanvasMetrics(canvas, trackContentWidth, fallbackHeightPx) {
  const rect = canvas.getBoundingClientRect();
  return {
    renderedWidthPx: Math.max(1, trackContentWidth),
    canvasWidthPx: Math.max(1, Math.round(Math.min(trackContentWidth, WAVEFORM_CANVAS_MAX_WIDTH_PX))),
    cssHeight: Math.max(1, rect.height || fallbackHeightPx),
  };
}

function waveformPeakAtCanvasX(x, canvasWidthPx, renderedWidthPx, peaks) {
  const timelinePx = renderedCanvasXToTimelinePx({
    canvasX: x,
    canvasWidthPx,
    renderedWidthPx,
  });
  const peakIndex = currentTimeScale().waveformPeakIndexAtPx(timelinePx, {
    audioDurationMs: state.waveform.durationMs,
    peakCount: peaks.length,
  });
  return peakIndex === null ? 0 : (peaks[peakIndex] ?? 0);
}

function drawWaveformOverlay(canvas, trackContentWidth) {
  const peaks = state.waveform.peaks;
  if (!peaks?.length) {
    return;
  }

  const { renderedWidthPx, canvasWidthPx, cssHeight } = waveformCanvasMetrics(canvas, trackContentWidth, 80);
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  canvas.width = Math.round(canvasWidthPx * dpr);
  canvas.height = Math.round(cssHeight * dpr);
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return;
  }

  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, canvasWidthPx, cssHeight);
  ctx.strokeStyle = "rgba(99, 210, 255, 0.18)";
  ctx.lineWidth = 1;
  const centerY = cssHeight / 2;
  const maxAmp = cssHeight * 0.34;

  ctx.beginPath();
  for (let x = 0; x < canvasWidthPx; x++) {
    const peak = waveformPeakAtCanvasX(x, canvasWidthPx, renderedWidthPx, peaks);
    const amp = peak <= 0 ? 0 : Math.max(1, peak * maxAmp);
    ctx.moveTo(x + 0.5, centerY - amp);
    ctx.lineTo(x + 0.5, centerY + amp);
  }
  ctx.stroke();
}

// ── Central waveform / oscilloscope panel ─────────────────────────────────────

/**
 * Append the central waveform panel to the timeline.  This panel is the shared
 * timebase anchor: it shows the amplitude waveform prominently and overlays
 * compact span chips for every word lane so ASR and TTS timing can be
 * correlated at a glance.
 */
function appendCentralWaveformPanel(labelsCol, scrollContent, trackContentWidth, nowMs, options = {}) {
  const wordLanes = state.lanes.filter((lane) => lane.type === "word");

  // Labels column entry
  const labelEl = document.createElement("div");
  labelEl.className = "lane-label-entry waveform-lane-label";
  const headerEl = document.createElement("div");
  headerEl.className = "lane-header waveform-panel-header";
  const h2El = document.createElement("h2");
  h2El.textContent = "Waveform";
  headerEl.append(h2El);
  const metaEl = document.createElement("div");
  metaEl.className = "lane-meta";
  if (state.waveform.status === "ready") {
    metaEl.textContent = `${(state.waveform.durationMs / 1000).toFixed(2)} s`;
  } else if (state.waveform.status === "loading") {
    metaEl.textContent = "Loading audio…";
  } else if (uiState.liveMode && state.payload?.audio?.url) {
    metaEl.textContent = "Waiting for live audio…";
  } else {
    metaEl.textContent = "No audio loaded";
  }
  headerEl.append(metaEl);
  labelEl.append(headerEl);
  labelsCol.append(labelEl);

  // Track element
  const trackEl = document.createElement("div");
  trackEl.className = "lane-track waveform-track";
  trackEl.id = "waveform-panel";

  // Waveform canvas — fills the full panel height as background
  const waveformCanvasWidth = centralWaveformCanvasWidth(trackContentWidth);
  const canvas = options.reusableCanvas ?? document.createElement("canvas");
  canvas.id = "central-waveform-canvas";
  canvas.className = "waveform-panel-canvas";
  canvas.setAttribute("aria-hidden", "true");
  canvas.style.width = `${waveformCanvasWidth}px`;
  trackEl.append(canvas);
  scheduleCentralWaveformDraw(canvas, waveformCanvasWidth);

  const wordDeck = document.createElement("div");
  wordDeck.className = "waveform-word-deck";
  wordDeck.style.width = `${trackContentWidth}px`;
  trackEl.append(wordDeck);

  // Word chips: one row per word lane, stacked directly beneath the waveform.
  // Up to WAVEFORM_PANEL_MAX_SPAN_ROWS lanes are shown to keep the panel readable.
  const maxRows = Math.min(wordLanes.length, WAVEFORM_PANEL_MAX_SPAN_ROWS);
  for (let relIdx = 0; relIdx < maxRows; relIdx++) {
    const lane = wordLanes[relIdx];
    const laneIndex = state.lanes.indexOf(lane);
    const topPx = WAVEFORM_SPAN_ROW_MARGIN_PX + relIdx * WAVEFORM_SPAN_ROW_STRIDE_PX;

    lane.words.forEach((word, wordIndex) => {
      const key = itemKey("word", laneIndex, wordIndex);
      const startMs = word.resolvedTiming.start_ms;
      const endMs = Math.max(word.resolvedTiming.end_ms, startMs + 1);
      state.itemTimingByKey.set(key, { startMs, endMs });
      const wordCss = intervalToCss(startMs, endMs, 2);

      const isSelected =
        state.selectedItem?.type === "word" &&
        state.selectedItem.laneIndex === laneIndex &&
        state.selectedItem.itemIndex === wordIndex;
      const isActive = nowMs >= startMs && nowMs <= endMs;

      const baseClass = [
        "timeline-chip",
        "waveform-span",
        `lane-${classToken(lane.label)}`,
        `source-${classToken(lane.stream.source)}`,
        `commit-${commitmentClass(word.commitment)}`,
        word._revisions?.length ? "was-revised" : "",
      ]
        .filter(Boolean)
        .join(" ");

      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = [baseClass, isActive ? "active" : "", isSelected ? "selected" : ""]
        .filter(Boolean)
        .join(" ");
      chip.dataset.baseClass = baseClass;
      chip.dataset.itemKey = key;
      chip.dataset.laneIndex = String(laneIndex);
      chip.dataset.itemIndex = String(wordIndex);
      chip.dataset.itemType = "word";
      chip.style.left = wordCss.left;
      chip.style.width = wordCss.width;
      chip.style.top = `${topPx}px`;
      chip.title = `${lane.label}: ${word.text} (${startMs}–${endMs} ms)`;
      chip.setAttribute("aria-label", `${lane.label}: ${word.text}`);
      chip.append(createChipContent(word.text));
      wordDeck.append(chip);
      state.waveformChipElements.set(key, chip);
    });
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

  const cursor = document.createElement("div");
  cursor.className = "playback-cursor";
  cursor.setAttribute("aria-hidden", "true");
  cursor.hidden = true;
  trackEl.append(cursor);
  state.playbackCursorElements.push(cursor);

  scrollContent.append(trackEl);
}

/**
 * Draw the central waveform onto its canvas.
 * Unlike the faint waveform-overlay on individual word lanes, this uses a
 * prominent filled envelope and stroked outline so it serves as the primary
 * visual timebase reference.
 */
function drawCentralWaveform(canvas, renderedWidthPx, signature = centralWaveformSignature(renderedWidthPx)) {
  const { canvasWidthPx, cssHeight } = waveformCanvasMetrics(canvas, renderedWidthPx, 120);
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  canvas.width = Math.round(canvasWidthPx * dpr);
  canvas.height = Math.round(cssHeight * dpr);
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    delete canvas.dataset.pendingWaveformSignature;
    return;
  }

  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, canvasWidthPx, cssHeight);

  const peaks = state.waveform.peaks;
  if (!peaks?.length) {
    // Flat zero-line placeholder when no audio is loaded
    ctx.strokeStyle = "rgba(157, 171, 186, 0.28)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, cssHeight / 2);
    ctx.lineTo(canvasWidthPx, cssHeight / 2);
    ctx.stroke();
    canvas.dataset.waveformSignature = signature;
    delete canvas.dataset.pendingWaveformSignature;
    return;
  }

  const centerY = cssHeight / 2;
  const maxAmp = cssHeight * 0.44;

  // Pre-compute upper/lower envelope y-values
  const topY = new Array(canvasWidthPx);
  const botY = new Array(canvasWidthPx);
  for (let x = 0; x < canvasWidthPx; x++) {
    const peak = waveformPeakAtCanvasX(x, canvasWidthPx, renderedWidthPx, peaks);
    const amp = Math.max(0, peak * maxAmp);
    topY[x] = centerY - amp;
    botY[x] = centerY + amp;
  }

  // Filled waveform envelope
  ctx.fillStyle = "rgba(99, 210, 255, 0.10)";
  ctx.beginPath();
  ctx.moveTo(0.5, topY[0]);
  for (let x = 1; x < canvasWidthPx; x++) {
    ctx.lineTo(x + 0.5, topY[x]);
  }
  for (let x = canvasWidthPx - 1; x >= 0; x--) {
    ctx.lineTo(x + 0.5, botY[x]);
  }
  ctx.closePath();
  ctx.fill();

  // Stroked upper outline
  ctx.strokeStyle = "rgba(99, 210, 255, 0.72)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0.5, topY[0]);
  for (let x = 1; x < canvasWidthPx; x++) {
    ctx.lineTo(x + 0.5, topY[x]);
  }
  ctx.stroke();

  // Stroked lower outline
  ctx.beginPath();
  ctx.moveTo(0.5, botY[0]);
  for (let x = 1; x < canvasWidthPx; x++) {
    ctx.lineTo(x + 0.5, botY[x]);
  }
  ctx.stroke();

  if (uiState.diagnosticsExpanded) {
    drawEnergyDebugOverlays(ctx, {
      cssHeight,
      canvasWidthPx,
      renderedWidthPx,
    });
  }
  canvas.dataset.waveformSignature = signature;
  delete canvas.dataset.pendingWaveformSignature;
}

function drawEnergyDebugOverlays(ctx, metrics) {
  const envelopeFrames = state.waveform.energyEnvelope?.frames;
  if (!Array.isArray(envelopeFrames) || envelopeFrames.length === 0) {
    return;
  }
  const landmarks = state.waveform.energyLandmarks ?? {
    onsets: [],
    offsets: [],
    valleys: [],
    silences: [],
    peaks: [],
  };
  const maxRms = Math.max(0.0001, ...envelopeFrames.map((frame) => frame.rms_energy ?? 0));
  const bottom = metrics.cssHeight - 1;
  const usableHeight = Math.max(20, metrics.cssHeight * 0.32);
  const top = bottom - usableHeight;

  ctx.save();
  ctx.strokeStyle = "rgba(255, 184, 77, 0.85)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let index = 0; index < envelopeFrames.length; index++) {
    const frame = envelopeFrames[index];
    const x = msToCanvasX(metrics, Math.round((frame.frame_start_ms + frame.frame_end_ms) / 2));
    const normalizedRms = Math.max(0, Math.min(1, (frame.rms_energy ?? 0) / maxRms));
    const y = bottom - normalizedRms * usableHeight;
    if (index === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  }
  ctx.stroke();

  drawDebugMarkers(ctx, metrics, landmarks.onsets, "rgba(103, 224, 116, 0.8)", top, bottom, 1);
  drawDebugMarkers(ctx, metrics, landmarks.offsets, "rgba(255, 118, 117, 0.8)", top, bottom, 1);
  drawDebugMarkers(ctx, metrics, landmarks.valleys, "rgba(255, 211, 74, 0.9)", top, bottom, 1.4);
  for (const silence of landmarks.silences ?? []) {
    const startX = msToCanvasX(metrics, silence.start_ms);
    const endX = msToCanvasX(metrics, silence.end_ms);
    ctx.fillStyle = "rgba(255, 255, 255, 0.06)";
    ctx.fillRect(Math.min(startX, endX), top, Math.abs(endX - startX), bottom - top);
  }

  const wordLanes = state.lanes.filter((lane) => lane.type === "word");
  const whisperBoundaries = [];
  const snappedBoundaries = [];
  for (const lane of wordLanes) {
    for (const word of lane.words) {
      if (word.whisperTiming) {
        whisperBoundaries.push(word.whisperTiming.start_ms, word.whisperTiming.end_ms);
      }
      if (
        word.timingResolution === "energy.snapped" &&
        word.resolvedTiming &&
        word.whisperTiming &&
        (word.resolvedTiming.start_ms !== word.whisperTiming.start_ms ||
          word.resolvedTiming.end_ms !== word.whisperTiming.end_ms)
      ) {
        snappedBoundaries.push(word.resolvedTiming.start_ms, word.resolvedTiming.end_ms);
      }
    }
  }
  drawDebugMarkers(ctx, metrics, whisperBoundaries, "rgba(201, 214, 226, 0.20)", top, bottom, 0.8);
  drawDebugMarkers(ctx, metrics, snappedBoundaries, "rgba(99, 210, 255, 0.85)", top, bottom, 1.2);
  ctx.restore();
}

function drawDebugMarkers(ctx, metrics, msValues, color, yTop, yBottom, width) {
  if (!Array.isArray(msValues) || msValues.length === 0) {
    return;
  }
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  for (const ms of msValues) {
    const x = msToCanvasX(metrics, ms);
    ctx.beginPath();
    ctx.moveTo(x, yTop);
    ctx.lineTo(x, yBottom);
    ctx.stroke();
  }
}

function msToCanvasX(metrics, ms) {
  const timelinePx = pxForMs(Math.max(0, Math.round(ms ?? 0)));
  const ratio = metrics.renderedWidthPx > 0 ? metrics.canvasWidthPx / metrics.renderedWidthPx : 1;
  return Math.max(0, Math.min(metrics.canvasWidthPx, timelinePx * ratio));
}

function waveformPeakVersion(peaks) {
  if (!peaks || typeof peaks !== "object") {
    return 0;
  }
  let version = waveformPeakVersions.get(peaks);
  if (!version) {
    version = nextWaveformPeakVersion++;
    waveformPeakVersions.set(peaks, version);
  }
  return version;
}

function waveformPeaksEqual(left, right) {
  if (!left || !right || left.length !== right.length) {
    return false;
  }
  for (let index = 0; index < left.length; index++) {
    if (Math.abs((left[index] ?? 0) - (right[index] ?? 0)) > 0.000001) {
      return false;
    }
  }
  return true;
}

function centralWaveformCanvasWidth(trackContentWidth) {
  if (state.waveform.status !== "ready" || state.waveform.durationMs <= 0) {
    return trackContentWidth;
  }
  return Math.max(1, Math.min(trackContentWidth, Math.ceil(pxForMs(state.waveform.durationMs))));
}

function centralWaveformSignature(renderedWidthPx) {
  const canvasWidthPx = Math.max(1, Math.round(Math.min(renderedWidthPx, WAVEFORM_CANVAS_MAX_WIDTH_PX)));
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const debugSignature = uiState.diagnosticsExpanded ? wordTimingDebugSignature() : "";
  return [
    state.waveform.status,
    state.waveform.url ?? "",
    state.waveform.durationMs,
    waveformPeakVersion(state.waveform.peaks),
    state.waveform.energyEnvelope?.frames?.length ?? 0,
    renderedWidthPx,
    canvasWidthPx,
    dpr,
    uiState.diagnosticsExpanded ? "debug" : "plain",
    debugSignature,
  ].join("|");
}

function wordTimingDebugSignature() {
  return state.lanes
    .filter((lane) => lane.type === "word")
    .flatMap((lane) =>
      lane.words.map((word) => {
        const timing = word.resolvedTiming ?? word.timing ?? {};
        const whisper = word.whisperTiming ?? {};
        return [
          timing.start_ms ?? "",
          timing.end_ms ?? "",
          whisper.start_ms ?? "",
          whisper.end_ms ?? "",
          word.timingResolution ?? "",
        ].join(":");
      }))
    .join(",");
}

function scheduleCentralWaveformDraw(canvas, renderedWidthPx) {
  const signature = centralWaveformSignature(renderedWidthPx);
  if (canvas.dataset.waveformSignature === signature || canvas.dataset.pendingWaveformSignature === signature) {
    return;
  }
  canvas.dataset.pendingWaveformSignature = signature;
  requestAnimationFrame(() => drawCentralWaveform(canvas, renderedWidthPx, signature));
}

// Update only active/selected classes on existing chips (no DOM rebuild).
function updateChipStates() {
  const nowMs = currentPlaybackTimeMs();
  function refreshChipMap(map, getTimingForKey) {
    for (const [key, chip] of map.entries()) {
      const timing = getTimingForKey(key);
      if (!timing) continue;
      const { itemType, laneIndex, itemIndex } = parseItemKey(key);
      const isMarker = timing.endMs <= timing.startMs;
      const activeEndMs = isMarker ? timing.startMs + MARKER_ACTIVE_DURATION_MS : timing.endMs;
      const isActive = nowMs >= timing.startMs && nowMs <= activeEndMs;
      const isSelected =
        state.selectedItem?.type === itemType &&
        state.selectedItem.laneIndex === laneIndex &&
        state.selectedItem.itemIndex === itemIndex;
      const isRelated = isRelatedDiagnosticEvent(itemType, laneIndex, itemIndex, timing);
      const baseClass = chip.dataset.baseClass ?? "";
      const newClass = [
        baseClass,
        isRelated ? "related-to-selection" : "",
        isActive ? "active" : "",
        isSelected ? "selected" : "",
      ]
        .filter(Boolean)
        .join(" ");
      if (chip.className !== newClass) {
        chip.className = newClass;
      }
    }
  }
  refreshChipMap(state.chipElementByKey, (key) => state.itemTimingByKey.get(key));
  // Waveform panel chips share itemTimingByKey for word chips
  refreshChipMap(state.waveformChipElements, (key) => state.itemTimingByKey.get(key));
}

function isRelatedDiagnosticEvent(itemType, laneIndex, itemIndex, timing) {
  if (itemType !== "event" || state.selectedItem?.type !== "word") {
    return false;
  }
  const selectedTiming = selectedItemTiming();
  if (!selectedTiming) {
    return false;
  }
  const lane = state.lanes[laneIndex];
  const event = lane?.events?.[itemIndex];
  if (!event) {
    return false;
  }
  const selectedWord = state.lanes[state.selectedItem.laneIndex]?.words?.[state.selectedItem.itemIndex];
  const selectedTurn = selectedWord?._turn ?? null;
  const eventTurn = event.metadata?.turn ?? event.metadata?.event?.turn ?? null;
  const overlaps =
    Math.max(selectedTiming.startMs, timing.startMs) <=
    Math.min(selectedTiming.endMs, timing.endMs);
  const sameTurn = selectedTurn !== null && eventTurn !== null && selectedTurn === eventTurn;
  return overlaps || sameTurn;
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
    selectWord(laneIndex, itemIndex, false);
    autoplayWordClip(laneIndex, itemIndex);
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
        word.whisperTiming
          ? `whisper ${word.whisperTiming.start_ms}–${word.whisperTiming.end_ms} ms`
          : null,
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
      if (state.surfaceMode === "graph") {
        renderGraphInspectionMode();
      }
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
    void seekSessionAudioToMs(word.resolvedTiming.start_ms);
  }

  clearPlaybackStop();
  ensureTimingVisible({
    startMs: word.resolvedTiming.start_ms,
    endMs: Math.max(word.resolvedTiming.end_ms, word.resolvedTiming.start_ms + 1),
  });
  updateChipStates();
  updateTimeRangeSelectionOverlays();
  renderSelection();
  if (state.surfaceMode === "graph") {
    renderGraphInspectionMode();
  }
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
      void seekSessionAudioToMs(event.start_ms);
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
  if (state.surfaceMode === "graph") {
    renderGraphInspectionMode();
  }
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

  if (word.timingResolution === "energy.snapped" && word.energyTiming) {
    const confidence = Number.isFinite(word.energyTiming.confidence)
      ? ` (confidence ${(word.energyTiming.confidence * 100).toFixed(0)}%)`
      : "";
    return `energy-snapped timing via ${word.energyTiming.method ?? "rms-valley-snap"}${confidence}`;
  }

  if (word.energyTiming && Number.isFinite(word.energyTiming.confidence)) {
    return `measured word.timing (energy evidence weak, confidence ${(word.energyTiming.confidence * 100).toFixed(0)}%)`;
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
    (lane?.label === "Mic" || lane?.label === "Mic / VAD") &&
    event?.kind === "speech_started" &&
    event?.metadata?.in_progress !== true &&
    endMs > startMs
  ) {
    return formatRulerLabel(endMs);
  }
  if (lane?.type === "event") {
    return shortKindLabel(event?.kind ?? "event");
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
    case "Confirmed": return "Confirmed";
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

function autoplayWordClip(laneIndex, wordIndex) {
  const word = state.lanes[laneIndex]?.words?.[wordIndex];
  if (!word || !audio.src) {
    return;
  }
  const startMs = word.resolvedTiming.start_ms;
  const endMs = Math.max(word.resolvedTiming.end_ms, startMs + 1);
  void seekSessionAudioToMs(startMs, { stopAtMs: endMs, autoplay: true });
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

  if (canonicalAudioUrl(audio.src) !== canonicalAudioUrl(targetUrl)) {
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
        h("strong", null, selectionKindLabel("event", lane, event)),
        h("br"),
        h("strong", null, lane.label),
        h("br"),
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
      badge: {
        className: `inspector-span-state diagnostic-kind-${classToken(selectionKindLabel("event", lane, event))}`,
        text: selectionKindLabel("event", lane, event),
      },
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
      h("strong", null, "Transcript word"),
      h("br"),
      h("strong", null, lane.label),
      h("br"),
      "Word ",
      h("strong", null, word.text),
      h("br"),
      `Whisper timing: ${(word.whisperTiming ?? word.timing)?.start_ms ?? "n/a"}–${(word.whisperTiming ?? word.timing)?.end_ms ?? "n/a"} ms`,
      h("br"),
      `Energy-snapped timing: ${word.energyTiming?.start_ms ?? "n/a"}–${word.energyTiming?.end_ms ?? "n/a"} ms`,
      h("br"),
      `Resolved timing: ${word.resolvedTiming.start_ms}–${word.resolvedTiming.end_ms} ms · confidence ${word.timing_confidence ?? "n/a"}`,
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
          whisperTiming: word.whisperTiming ?? word.timing ?? null,
          energyTiming: word.energyTiming ?? null,
          resolvedTiming: word.resolvedTiming,
          timingResolution: word.timingResolution,
          timingSourceDetail: word.timingSourceDetail,
          energySnapConfidence: word.energySnapConfidence,
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
          text: `Word · ${describeSpanState(word.commitment)}`,
        }
      : null,
    revisions: (word._revisions ?? []).map((rev) => ({
      atMs: rev.at_ms,
      fromText: rev.fromText,
      toText: word.text,
    })),
  };
}

function selectionKindLabel(itemType, lane, item) {
  if (itemType === "word") {
    return "Transcript word";
  }
  const kind = String(item?.kind ?? "");
  const laneLabel = lane?.label ?? "";
  if (laneLabel === "Mic / VAD" || kind.includes("speech") || kind.includes("breath")) {
    return "Mic / VAD event";
  }
  if (kind.includes("suppression")) {
    return "Suppression event";
  }
  if (kind.includes("playback")) {
    return "Playback event";
  }
  if (kind.includes("tts") || laneLabel === "TTS") {
    return "TTS event";
  }
  if (kind.includes("llm") || laneLabel === "LLM") {
    return "LLM event";
  }
  if (kind.includes("transcript") || kind.includes("asr") || laneLabel === "ASR Events") {
    return "ASR event";
  }
  return "Diagnostic event";
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
    case "Confirmed":    return "Confirmed — refined by broader ASR context";
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
