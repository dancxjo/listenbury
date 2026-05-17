const viewer = document.getElementById("viewer");
const viewerTitle = document.getElementById("viewer-title");
const statusMessage = document.getElementById("status-message");
const playbackTime = document.getElementById("playback-time");
const selectionSummary = document.getElementById("selection-summary");
const selectionJson = document.getElementById("selection-json");
const playPauseButton = document.getElementById("play-pause");
const jumpPrevButton = document.getElementById("jump-prev");
const jumpNextButton = document.getElementById("jump-next");
const playSelectionClipButton = document.getElementById("play-selection-clip");
const zoomOutButton = document.getElementById("zoom-out");
const zoomInButton = document.getElementById("zoom-in");
const audio = document.getElementById("audio");
const liveBanner = document.getElementById("live-banner");
const liveEventCount = document.getElementById("live-event-count");
const liveConnectionStatus = document.getElementById("live-connection-status");
const transcriptRibbon = document.getElementById("transcript-ribbon");
const transcriptRibbonText = document.getElementById("transcript-ribbon-text");
const spanDebugLog = document.getElementById("span-debug-log");
const VIEWER_NAME = "WaveDeck";
const MIN_VIEW_DURATION_MS = 100;
const MIN_SELECTION_VIEW_MS = 500;
const ZOOM_FACTOR = 1.8;
const WHEEL_ZOOM_FACTOR = 1.16;
const RANGE_SELECTION_DRAG_THRESHOLD_PX = 12;

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
  first_safe_speech_unit_emitted: "LLM",
  speech_unit_committed: "LLM",
  speech_unit_cancelled: "LLM",
  speculative_speech_updated: "LLM",
  first_tts_audio_frame_available: "Speaker",
  playback_started: "Speaker",
  playback_finished: "Speaker",
  self_hearing_suppression_started: "Speaker",
  self_hearing_suppression_ended: "Speaker",
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
  viewStartMs: null,
  viewEndMs: null,
  dragSelection: null,
  suppressTimelineClick: false,
};

const sourceLabels = {
  RecordedAudio: "Recorded audio",
  LiveAsr: "Live ASR",
  GeneratedText: "Generated text",
  SyntheticSpeech: "Synthetic speech",
};

playPauseButton.addEventListener("click", () => togglePlayback());
jumpPrevButton.addEventListener("click", () => jumpSelectedWord(-1));
jumpNextButton.addEventListener("click", () => jumpSelectedWord(1));
playSelectionClipButton.addEventListener("click", () => playSelectedClip());
zoomOutButton.addEventListener("click", () => zoomTimeline(1 / ZOOM_FACTOR));
zoomInButton.addEventListener("click", () => zoomTimeline(ZOOM_FACTOR));
viewer.addEventListener("pointerdown", startTimeRangeSelection);
viewer.addEventListener("pointermove", moveTimeRangeSelection);
viewer.addEventListener("pointerup", finishTimeRangeSelection);
viewer.addEventListener("pointercancel", cancelTimeRangeSelection);

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
  liveBanner.hidden = false;
  transcriptRibbon.hidden = false;
  document.title = "WaveDeck · Live";
  viewerTitle.textContent = "Live — Listenbury";
  statusMessage.textContent = "Connecting to live event stream…";

  connectLiveEvents();
}

function connectLiveEvents() {
  const source = new EventSource("/api/live-events");

  source.onopen = () => {
    liveConnectionStatus.textContent = "connected";
    liveConnectionStatus.className = "live-status-connected";
    statusMessage.textContent = "Listening for live events…";
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
    liveConnectionStatus.textContent = "unavailable";
    liveConnectionStatus.className = "live-status-error";
    statusMessage.textContent = message;
    source.close();
  });

  source.onerror = () => {
    liveConnectionStatus.textContent = "disconnected";
    liveConnectionStatus.className = "live-status-error";
    statusMessage.textContent = "Live event stream disconnected. Session may have ended.";
    source.close();
  };
}

function addLiveEvent(traceEvent) {
  liveEvents.push(traceEvent);
  reduceLiveEvent(liveSession, traceEvent);
  liveEventCount.textContent = `${liveEvents.length} event${liveEvents.length === 1 ? "" : "s"}`;

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
//   { id, transcriptCandidate, finalTranscript, latestWordStream, wordRevisions }

function createLiveSession() {
  return {
    turns: new Map(),      // turnId → LiveTurn
    openSpans: new Map(),  // openSpanKey → start_ms
    viewerEvents: [],      // accumulated closed span events
    viewerMarkers: [],     // accumulated point markers
    debugLog: [],          // debug entries, generated exactly once per input event
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
      wordRevisions: new Map(), // stableWordKey → [{fromText, at_ms, provenance, approximate}]
    });
  }
  return session.turns.get(turnId);
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

  // ── breath_group_opened → open span
  if (event.kind === "breath_group_opened") {
    const key = openSpanKey("Mic", turn, "breath_group_opened");
    session.openSpans.set(key, event.elapsed_ms);
    log("open", `Breath group opened (turn ${turn})`);
    return;
  }

  // ── breath_group_closed → close span
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

  // ── Span start event → record open span
  if (SPAN_PAIRS[event.kind]) {
    const lane = LIVE_EVENT_LANE[event.kind] ?? "Events";
    const spanKey = openSpanKey(lane, turn, event.kind);
    session.openSpans.set(spanKey, event.elapsed_ms);
    if (event.kind === "asr_started") {
      log("open", `ASR span opened (turn ${turn}) [Hypothesis]`);
    }
    return;
  }

  // ── All other events → point marker
  const lane = LIVE_EVENT_LANE[event.kind] ?? "Events";
  const label = event.text ? event.text.slice(0, 60) : labelForKind(event.kind);
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
  for (const [key, startMs] of session.openSpans.entries()) {
    const [spanLane, spanTurn, kind] = JSON.parse(key);
    inProgressEvents.push({
      lane: spanLane,
      kind,
      label: `${labelForKind(kind)} (in progress)`,
      start_ms: startMs,
      end_ms: session.maxElapsedMs,
      metadata: { in_progress: true, turn: spanTurn },
    });
  }

  // Per-turn word stream lanes (sorted by turn id for stable ordering).
  const wordStreamLanes = [];
  for (const [turnId, liveTurn] of [...session.turns.entries()].sort((a, b) => a[0] - b[0])) {
    if (liveTurn.latestWordStream?.words?.length > 0) {
      wordStreamLanes.push({
        label: `ASR turn ${turnId}`,
        stream: {
          id: turnId,
          source: "LiveAsr",
          words: liveTurn.latestWordStream.words,
        },
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

// ── Span debug log ─────────────────────────────────────────────────────────

function renderSpanDebugLog(session) {
  if (!spanDebugLog || !session.debugLog.length) {
    return;
  }
  // Show the 40 most-recent entries, newest first.
  const entries = session.debugLog.slice(-40).reverse();
  const fragment = document.createDocumentFragment();
  for (const entry of entries) {
    const el = document.createElement("div");
    el.className = `span-debug-entry entry-${entry.type}`;
    const time = document.createElement("span");
    time.className = "span-debug-time";
    time.textContent = `${(entry.elapsedMs / 1000).toFixed(3)}s`;
    const msg = document.createElement("span");
    msg.className = "span-debug-msg";
    msg.textContent = entry.message;
    el.append(time, msg);
    fragment.append(el);
  }
  spanDebugLog.replaceChildren(fragment);
}

// ── Transcript ribbon ──────────────────────────────────────────────────────

// Render the live transcript ribbon from the durable LiveSession state.
// Both past turns (via finalTranscript + latestWordStream) and the current
// in-progress turn (via transcriptCandidate or latestWordStream) are driven
// by the session model, not by raw event lists.
function renderTranscriptRibbon(session) {
  if (!transcriptRibbonText) {
    return;
  }

  const fragment = document.createDocumentFragment();
  const sortedTurns = [...session.turns.entries()].sort((a, b) => a[0] - b[0]);

  for (const [, liveTurn] of sortedTurns) {
    if (liveTurn.finalTranscript != null) {
      // Committed turn: use word-level commitment states when available.
      const wordStream = liveTurn.latestWordStream;
      if (wordStream?.words?.length > 0) {
        for (const word of wordStream.words) {
          const token = document.createElement("span");
          const commitClass = `commit-${commitmentClass(word.commitment)}`;
          token.className = `transcript-token ${commitClass}${word._revisions?.length ? " was-revised" : ""}`;
          token.textContent = word.text;
          const revTooltip = formatRevisionTooltip(word);
          if (revTooltip) {
            token.title = revTooltip;
          }
          fragment.append(token, " ");
        }
      } else {
        // Fall back to plain committed text.
        const token = document.createElement("span");
        token.className = "transcript-token span-state-committed";
        token.textContent = liveTurn.finalTranscript;
        fragment.append(token, " ");
      }
    } else if (liveTurn.transcriptCandidate) {
      // In-progress turn: stable prefix + unstable tail from transcript_candidate.
      const { stable_text, unstable_text } = liveTurn.transcriptCandidate;
      if (stable_text) {
        const stableToken = document.createElement("span");
        stableToken.className = "transcript-token span-state-stable";
        stableToken.textContent = stable_text;
        fragment.append(stableToken, " ");
      }
      if (unstable_text) {
        const provisionalToken = document.createElement("span");
        provisionalToken.className = "transcript-token span-state-hypothetical";
        provisionalToken.textContent = unstable_text;
        fragment.append(provisionalToken);
      }
    } else if (liveTurn.latestWordStream?.words?.length > 0) {
      // Word-stream fallback when no transcript_candidate is available.
      for (const word of liveTurn.latestWordStream.words) {
        const token = document.createElement("span");
        const commitClass = `commit-${commitmentClass(word.commitment)}`;
        token.className = `transcript-token ${commitClass}`;
        token.textContent = word.text;
        fragment.append(token, " ");
      }
    }
  }

  transcriptRibbonText.replaceChildren(fragment);
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
  applyPayload(payload, { preserveSelection: true, preserveViewport: true });
  viewerTitle.textContent = "Live — Listenbury";
  renderTranscriptRibbon(liveSession);
  renderSpanDebugLog(liveSession);
}

function labelForKind(kind) {
  return kind.replace(/_/g, " ");
}

function togglePlayback() {
  if (!audio.src) {
    statusMessage.textContent = "No audio loaded for this selection.";
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
  const previousViewport = options.preserveViewport ? captureViewportSnapshot() : null;
  if (!options.preserveViewport) {
    state.viewStartMs = null;
    state.viewEndMs = null;
  }
  state.dragSelection = null;

  const normalized = normalizePayload(rawPayload);
  const wordLanes = normalized.streams.map((lane) => normalizeWordLane(lane));
  const eventLanes = normalizeEventLanes(normalized.events);

  state.payload = normalized;
  state.lanes = [...wordLanes, ...eventLanes].map((lane, laneIndex) => {
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
  state.selectedItem = validSelection(previousSelection) ? previousSelection : firstItemSelection();
  clearPlaybackStop();
  configureAudio(normalized.audio);
  syncMaxDurationWithAudio();
  restoreViewportSnapshot(previousViewport);
  render();
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

    normalizedEvents.push({
      id: entry.id ?? `event-${index + 1}`,
      lane: entry.lane ?? "Events",
      kind: entry.kind ?? "event",
      label: entry.label ?? entry.kind ?? `Event ${index + 1}`,
      start_ms: startMs,
      end_ms: endMs,
      metadata: entry.metadata ?? null,
      audio_ref: normalizeEventAudioRef(entry, startMs, endMs),
      style: endMs > startMs ? "span" : "marker",
      original: entry,
    });
  });

  markers.forEach((entry, index) => {
    if (!entry || typeof entry !== "object") {
      throw new Error(`marker entry ${index} must be an object`);
    }

    const atMs = coerceMs(entry.at_ms ?? entry.start_ms, `marker ${index} at_ms`);

    normalizedEvents.push({
      id: entry.id ?? `marker-${index + 1}`,
      lane: entry.lane ?? "Markers",
      kind: entry.kind ?? "marker",
      label: entry.label ?? entry.kind ?? `Marker ${index + 1}`,
      start_ms: atMs,
      end_ms: atMs,
      metadata: entry.metadata ?? null,
      audio_ref: normalizeEventAudioRef(entry, atMs, atMs),
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

function inferPayloadDuration(stream) {
  const timedEnd = Math.max(0, ...stream.words.map((word) => word.timing?.end_ms ?? 0));
  return timedEnd || 1000;
}

function configureAudio(audioConfig) {
  if (!audioConfig?.url) {
    return;
  }

  audio.src = audioConfig.url;
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
  clampViewport();
}

function render() {
  viewerTitle.textContent = state.payload?.title ?? "No stream loaded";
  playbackTime.textContent = formatPlaybackTime();
  playPauseButton.textContent = audio.paused ? "Play" : "Pause";
  updateZoomControls();

  if (!state.lanes.length) {
    viewer.className = "viewer empty";
    viewer.innerHTML = "<p>No streams or events loaded yet.</p>";
    renderSelection();
    return;
  }

  viewer.className = "viewer";
  viewer.replaceChildren(renderTimelineRuler(), ...state.lanes.map(renderLane));
  refreshPlaybackState();
  renderSelection();
}

function renderLane(lane) {
  return lane.type === "event" ? renderEventLane(lane) : renderWordLane(lane);
}

function renderTimelineRuler() {
  const viewport = getViewport();
  const ruler = document.createElement("div");
  ruler.className = "timeline-ruler";
  appendTimeRangeSelection(ruler, viewport);

  const ticks = buildRulerTicks(viewport);
  ticks.forEach((tickMs) => {
    const tick = document.createElement("span");
    tick.className = "ruler-tick";
    tick.style.left = `${msToViewportPercent(tickMs, viewport)}%`;

    const label = document.createElement("span");
    label.className = "ruler-label";
    label.style.left = `${msToViewportPercent(tickMs, viewport)}%`;
    label.textContent = formatRulerLabel(tickMs);

    ruler.append(tick, label);
  });

  ruler.addEventListener("click", (event) => {
    if (state.suppressTimelineClick) {
      state.suppressTimelineClick = false;
      return;
    }
    if (!audio.src) {
      return;
    }
    const rect = ruler.getBoundingClientRect();
    const ratio = Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width));
    audio.currentTime = (viewport.startMs + ratio * viewport.durationMs) / 1000;
    refreshPlaybackState();
  });
  ruler.addEventListener("wheel", (event) => zoomTimelineFromWheel(event, ruler), { passive: false });

  return ruler;
}

function renderWordLane(lane) {
  const viewport = getViewport();
  const section = document.createElement("section");
  section.className = "lane";

  const header = document.createElement("div");
  header.className = "lane-header";

  const title = document.createElement("h2");
  title.textContent = lane.label;

  const meta = document.createElement("div");
  meta.className = "lane-meta";
  meta.textContent = `${sourceLabels[lane.stream.source] ?? lane.stream.source} · ${lane.words.length} words`;

  header.append(title, meta);
  section.append(header);

  const track = document.createElement("div");
  track.className = "lane-track";
  appendTimeRangeSelection(track, viewport);

  lane.words.forEach((word) => {
    const visibleRange = visibleItemRange(
      word.resolvedTiming.start_ms,
      word.resolvedTiming.end_ms,
      viewport,
    );
    if (!visibleRange) {
      return;
    }

    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "timeline-chip word-chip";
    if (word.timingResolution === "fallback-layout") {
      chip.classList.add("fallback-timing");
    }
    // Span state: style by WordCommitment (Hypothetical / StableText / Final / Cancelled …)
    if (word.commitment) {
      chip.classList.add(`commit-${commitmentClass(word.commitment)}`);
    }
    // Retroactive revision: word text was corrected in a later ASR pass.
    if (word._revisions?.length > 0) {
      chip.classList.add("was-revised");
    }
    chip.dataset.key = itemKey("word", word.laneIndex, word.wordIndex);
    chip.dataset.itemType = "word";
    chip.dataset.startMs = String(word.resolvedTiming.start_ms);
    chip.dataset.endMs = String(word.resolvedTiming.end_ms);
    chip.textContent = word.text;
    chip.title = `${word.text} (${word.resolvedTiming.start_ms}–${word.resolvedTiming.end_ms} ms) · ${word.timingSourceDetail}`;

    const left = msToViewportPercent(visibleRange.startMs, viewport);
    const width = Math.max(
      6,
      durationToViewportPercent(visibleRange.endMs - visibleRange.startMs, viewport),
    );

    const clampedLeft = Math.min(left, 99);
    chip.style.left = `${clampedLeft}%`;
    chip.style.width = `${Math.min(width, 100 - clampedLeft)}%`;
    chip.addEventListener("click", () => selectWord(word.laneIndex, word.wordIndex, true));
    track.append(chip);
  });

  section.append(track);
  return section;
}

function renderEventLane(lane) {
  const viewport = getViewport();
  const section = document.createElement("section");
  section.className = "lane event-lane";

  const header = document.createElement("div");
  header.className = "lane-header";

  const title = document.createElement("h2");
  title.textContent = lane.label;

  const meta = document.createElement("div");
  meta.className = "lane-meta";
  meta.textContent = `${lane.events.length} events`;

  header.append(title, meta);
  section.append(header);

  const track = document.createElement("div");
  track.className = "lane-track event-track";
  appendTimeRangeSelection(track, viewport);

  lane.events.forEach((event) => {
    const visibleRange = visibleItemRange(event.start_ms, event.end_ms, viewport);
    if (!visibleRange) {
      return;
    }

    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = `timeline-chip event-chip ${event.style} kind-${classToken(event.kind)}`;
    chip.dataset.key = itemKey("event", event.laneIndex, event.eventIndex);
    chip.dataset.itemType = "event";
    chip.dataset.startMs = String(event.start_ms);
    chip.dataset.endMs = String(event.end_ms);
    chip.textContent = event.label;
    chip.title = `${event.kind} (${event.start_ms}–${event.end_ms} ms)`;

    const left = msToViewportPercent(visibleRange.startMs, viewport);
    const width =
      event.style === "marker"
        ? 2
        : Math.max(3, durationToViewportPercent(visibleRange.endMs - visibleRange.startMs, viewport));

    const clampedLeft = Math.min(left, 99);
    chip.style.left = `${clampedLeft}%`;
    chip.style.width = `${Math.min(width, 100 - clampedLeft)}%`;
    chip.addEventListener("click", () => selectEvent(event.laneIndex, event.eventIndex, true));
    track.append(chip);
  });

  section.append(track);
  return section;
}

function refreshPlaybackState() {
  playbackTime.textContent = formatPlaybackTime();
  playPauseButton.textContent = audio.paused ? "Play" : "Pause";
  const nowMs = Math.round(audio.currentTime * 1000);

  document.querySelectorAll(".timeline-chip").forEach((chip) => {
    const startMs = Number(chip.dataset.startMs);
    const endMs = Number(chip.dataset.endMs);
    const isMarker = chip.dataset.itemType === "event" && endMs <= startMs;
    const activeEndMs = isMarker ? startMs + 120 : endMs;
    chip.classList.toggle("active", nowMs >= startMs && nowMs <= activeEndMs);

    chip.classList.toggle(
      "selected",
      state.selectedItem !== null &&
        chip.dataset.key === itemKey(state.selectedItem.type, state.selectedItem.laneIndex, state.selectedItem.itemIndex),
    );
  });
}

function renderSelection() {
  updateZoomControls();
  playSelectionClipButton.disabled = true;
  playSelectionClipButton.textContent = "Play selected clip";
  if (!state.selectedItem) {
    selectionSummary.textContent = "Select a word or event to inspect timing and metadata.";
    selectionJson.textContent = "{}";
    return;
  }

  if (state.selectedItem.type === "event") {
    const lane = state.lanes[state.selectedItem.laneIndex];
    const event = lane?.events?.[state.selectedItem.itemIndex];
    if (!lane || !event) {
      selectionSummary.textContent = "Select a word or event to inspect timing and metadata.";
      selectionJson.textContent = "{}";
      return;
    }

    selectionSummary.innerHTML = `
      <strong>${lane.label}</strong><br />
      Event <strong>${event.label}</strong><br />
      ${event.start_ms}–${event.end_ms} ms · kind <strong>${event.kind}</strong>
    `;
    if (event.audio_ref?.url) {
      playSelectionClipButton.disabled = false;
      playSelectionClipButton.textContent = "Play event clip";
    }

    selectionJson.textContent = JSON.stringify(
      {
        lane: lane.label,
        laneType: "event",
        id: event.id,
        kind: event.kind,
        label: event.label,
        start_ms: event.start_ms,
        end_ms: event.end_ms,
        duration_ms: Math.max(0, event.end_ms - event.start_ms),
        audioRef: event.audio_ref,
        metadata: event.metadata,
        original: event.original,
      },
      null,
      2,
    );
    return;
  }

  const lane = state.lanes[state.selectedItem.laneIndex];
  const word = lane?.words?.[state.selectedItem.itemIndex];
  if (!lane || !word) {
    selectionSummary.textContent = "Select a word or event to inspect timing and metadata.";
    selectionJson.textContent = "{}";
    return;
  }

  selectionSummary.innerHTML = `
    <strong>${lane.label}</strong><br />
    Word <strong>${word.text}</strong><br />
    ${word.resolvedTiming.start_ms}–${word.resolvedTiming.end_ms} ms · confidence ${
      word.timing_confidence ?? "n/a"
    }<br />
    Timing source: <strong>${word.timingSourceDetail}</strong>
  `;

  const revisionBlock = buildRevisionHistoryHtml(word);

  selectionJson.textContent = JSON.stringify(
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
  );

  // Show commitment state badge + revision history in the summary panel.
  if (word.commitment) {
    const badge = document.createElement("span");
    badge.className = `inspector-span-state commit-${commitmentClass(word.commitment)}`;
    badge.textContent = describeSpanState(word.commitment);
    selectionSummary.prepend(badge);
  }
  if (revisionBlock) {
    selectionSummary.insertAdjacentHTML("beforeend", revisionBlock);
  }
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
  refreshPlaybackState();
  renderSelection();
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
  refreshPlaybackState();
  renderSelection();
}

function zoomTimeline(factor, focusMsOverride = null) {
  if (!state.lanes.length) {
    return;
  }

  const viewport = getViewport();
  const focusMs = Number.isFinite(focusMsOverride) ? focusMsOverride : viewportFocusMs(viewport);
  const nextDuration = viewport.durationMs / factor;
  setViewportAround(focusMs, nextDuration);
  render();
}

function zoomTimelineFromWheel(event, surface) {
  if (!state.lanes.length) {
    return;
  }

  const wheelDelta = Math.abs(event.deltaY) >= Math.abs(event.deltaX) ? event.deltaY : event.deltaX;
  if (!wheelDelta) {
    return;
  }

  const focusMs = timeMsAtClientX(event.clientX, surface);
  if (focusMs === null) {
    return;
  }

  event.preventDefault();
  zoomTimeline(wheelDelta < 0 ? WHEEL_ZOOM_FACTOR : 1 / WHEEL_ZOOM_FACTOR, focusMs);
}

function zoomToTimeSelection(selection) {
  const timing = clampTimeSelection(selection);
  if (!timing) {
    return;
  }

  setViewportRange(timing.startMs, timing.endMs);
  render();
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

function viewportFocusMs(viewport) {
  const selection = selectedItemTiming();
  if (selection) {
    const selectionCenter = (selection.startMs + selection.endMs) / 2;
    if (selectionCenter >= viewport.startMs && selectionCenter <= viewport.endMs) {
      return selectionCenter;
    }
  }

  const playheadMs = audio.currentTime * 1000;
  if (Number.isFinite(playheadMs) && playheadMs >= viewport.startMs && playheadMs <= viewport.endMs) {
    return playheadMs;
  }

  return viewport.startMs + viewport.durationMs / 2;
}

function setViewportAround(focusMs, requestedDurationMs) {
  const nextDuration = Math.max(
    MIN_VIEW_DURATION_MS,
    Math.min(state.maxDurationMs, requestedDurationMs),
  );

  if (nextDuration >= state.maxDurationMs) {
    state.viewStartMs = null;
    state.viewEndMs = null;
    return;
  }

  const unclampedStart = focusMs - nextDuration / 2;
  const startMs = Math.max(0, Math.min(state.maxDurationMs - nextDuration, unclampedStart));
  state.viewStartMs = startMs;
  state.viewEndMs = startMs + nextDuration;
}

function setViewportRange(startMs, endMs) {
  const selectionDuration = Math.max(MIN_SELECTION_VIEW_MS, endMs - startMs);
  const centerMs = startMs + (endMs - startMs) / 2;
  setViewportAround(centerMs, selectionDuration);
}

function captureViewportSnapshot() {
  const viewport = getViewport();
  return {
    startMs: viewport.startMs,
    endMs: viewport.endMs,
    durationMs: viewport.durationMs,
    followsTimelineEnd: Math.abs(viewport.endMs - state.maxDurationMs) < 1,
  };
}

function restoreViewportSnapshot(snapshot) {
  if (!snapshot) {
    return;
  }

  const durationMs = Math.max(
    MIN_VIEW_DURATION_MS,
    Math.min(state.maxDurationMs, snapshot.durationMs),
  );

  if (durationMs >= state.maxDurationMs) {
    state.viewStartMs = null;
    state.viewEndMs = null;
    return;
  }

  const requestedStartMs = snapshot.followsTimelineEnd
    ? state.maxDurationMs - durationMs
    : snapshot.startMs;
  const startMs = Math.max(0, Math.min(state.maxDurationMs - durationMs, requestedStartMs));
  state.viewStartMs = startMs;
  state.viewEndMs = startMs + durationMs;
}

function clampViewport() {
  if (state.viewStartMs === null || state.viewEndMs === null) {
    state.viewStartMs = null;
    state.viewEndMs = null;
    return;
  }

  const durationMs = state.viewEndMs - state.viewStartMs;
  if (durationMs >= state.maxDurationMs || durationMs < MIN_VIEW_DURATION_MS) {
    state.viewStartMs = null;
    state.viewEndMs = null;
    return;
  }

  const startMs = Math.max(0, Math.min(state.maxDurationMs - durationMs, state.viewStartMs));
  state.viewStartMs = startMs;
  state.viewEndMs = startMs + durationMs;
}

function getViewport() {
  const hasCustomViewport =
    state.viewStartMs !== null &&
    state.viewEndMs !== null &&
    state.viewEndMs > state.viewStartMs;
  const startMs = hasCustomViewport ? state.viewStartMs : 0;
  const endMs = hasCustomViewport ? state.viewEndMs : state.maxDurationMs;
  return {
    startMs,
    endMs,
    durationMs: Math.max(MIN_VIEW_DURATION_MS, endMs - startMs),
    isFullTimeline: !hasCustomViewport,
  };
}

function visibleItemRange(startMs, endMs, viewport) {
  const itemStartMs = Math.max(0, startMs);
  const itemEndMs = Math.max(itemStartMs + 1, endMs);
  if (itemEndMs < viewport.startMs || itemStartMs > viewport.endMs) {
    return null;
  }
  return {
    startMs: Math.max(itemStartMs, viewport.startMs),
    endMs: Math.min(itemEndMs, viewport.endMs),
  };
}

function msToViewportPercent(ms, viewport) {
  return ((ms - viewport.startMs) / viewport.durationMs) * 100;
}

function durationToViewportPercent(durationMs, viewport) {
  return (durationMs / viewport.durationMs) * 100;
}

function appendTimeRangeSelection(container, viewport) {
  const overlay = document.createElement("div");
  overlay.className = "time-range-selection";
  overlay.setAttribute("aria-hidden", "true");
  container.append(overlay);
  updateTimeRangeSelectionOverlay(overlay, viewport);
}

function updateTimeRangeSelectionOverlays() {
  const viewport = getViewport();
  document.querySelectorAll(".time-range-selection").forEach((overlay) => {
    updateTimeRangeSelectionOverlay(overlay, viewport);
  });
  updateZoomControls();
}

function updateTimeRangeSelectionOverlay(overlay, viewport) {
  const selection = activeTimeRangeSelection();
  if (!selection) {
    overlay.hidden = true;
    overlay.style.width = "0";
    return;
  }

  const visibleRange = visibleItemRange(selection.startMs, selection.endMs, viewport);
  if (!visibleRange) {
    overlay.hidden = true;
    overlay.style.width = "0";
    return;
  }

  const left = Math.max(0, Math.min(100, msToViewportPercent(visibleRange.startMs, viewport)));
  const right = Math.max(0, Math.min(100, msToViewportPercent(visibleRange.endMs, viewport)));
  overlay.hidden = false;
  overlay.style.left = `${left}%`;
  overlay.style.width = `${Math.max(0, right - left)}%`;
}

function activeTimeRangeSelection() {
  if (state.dragSelection) {
    return normalizeTimeSelection(state.dragSelection.startMs, state.dragSelection.endMs);
  }
  return null;
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

  const startMs = timeMsAtClientX(event.clientX, surface);
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
  viewer.setPointerCapture(event.pointerId);
  updateTimeRangeSelectionOverlays();
}

function moveTimeRangeSelection(event) {
  if (!state.dragSelection || state.dragSelection.pointerId !== event.pointerId) {
    return;
  }

  const endMs = timeMsAtClientX(event.clientX, state.dragSelection.surface);
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
  const endMs = timeMsAtClientX(event.clientX, dragSelection.surface);
  if (endMs !== null) {
    dragSelection.endMs = endMs;
  }

  state.dragSelection = null;
  if (viewer.hasPointerCapture(event.pointerId)) {
    viewer.releasePointerCapture(event.pointerId);
  }

  const delta = Math.abs(event.clientX - dragSelection.startClientX);
  if (delta < RANGE_SELECTION_DRAG_THRESHOLD_PX) {
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
  zoomToTimeSelection(selection);
}

function cancelTimeRangeSelection(event) {
  if (!state.dragSelection || state.dragSelection.pointerId !== event.pointerId) {
    return;
  }

  state.dragSelection = null;
  if (viewer.hasPointerCapture(event.pointerId)) {
    viewer.releasePointerCapture(event.pointerId);
  }
  updateTimeRangeSelectionOverlays();
}

function timeMsAtClientX(clientX, surface) {
  const viewport = getViewport();
  const rect = surface.getBoundingClientRect();
  if (!rect.width) {
    return null;
  }
  const ratio = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
  return viewport.startMs + ratio * viewport.durationMs;
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
  const viewport = getViewport();
  const canZoom = state.lanes.length > 0;
  zoomInButton.disabled = !canZoom || viewport.durationMs <= MIN_VIEW_DURATION_MS;
  zoomOutButton.disabled = !canZoom || viewport.isFullTimeline;
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

function classToken(value) {
  return String(value ?? "event")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, "-");
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
    statusMessage.textContent = `Loaded clip reference ${targetUrl}.`;
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

function buildRulerTicks(viewport) {
  const safeDuration = Math.max(MIN_VIEW_DURATION_MS, viewport.durationMs);
  const targetSegments = 10;
  const preferredSteps = [25, 50, 100, 250, 500, 1000, 2000, 5000, 10_000, 15_000, 30_000, 60_000];
  const desiredStep = safeDuration / targetSegments;
  const stepMs = preferredSteps.find((step) => step >= desiredStep) ?? 120_000;
  const ticks = [];
  const firstTick = Math.ceil(viewport.startMs / stepMs) * stepMs;
  if (viewport.startMs > 0) {
    ticks.push(viewport.startMs);
  }
  for (let at = firstTick; at <= viewport.endMs; at += stepMs) {
    ticks.push(at);
  }
  if (ticks[ticks.length - 1] !== viewport.endMs) {
    ticks.push(viewport.endMs);
  }
  return [...new Set(ticks)];
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

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
