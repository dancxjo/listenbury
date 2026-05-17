const viewer = document.getElementById("viewer");
const viewerTitle = document.getElementById("viewer-title");
const statusMessage = document.getElementById("status-message");
const playbackTime = document.getElementById("playback-time");
const selectionSummary = document.getElementById("selection-summary");
const selectionJson = document.getElementById("selection-json");
const loadDemoButton = document.getElementById("load-demo");
const jsonFileInput = document.getElementById("json-file");
const audioFileInput = document.getElementById("audio-file");
const playPauseButton = document.getElementById("play-pause");
const jumpPrevButton = document.getElementById("jump-prev");
const jumpNextButton = document.getElementById("jump-next");
const playSelectionClipButton = document.getElementById("play-selection-clip");
const audio = document.getElementById("audio");
const liveBanner = document.getElementById("live-banner");
const liveEventCount = document.getElementById("live-event-count");
const liveConnectionStatus = document.getElementById("live-connection-status");
const queryParams = new URLSearchParams(window.location.search);
const VIEWER_NAME = "WaveDeck";

// Live mode is activated by ?live=1 in the URL.
const isLiveMode = queryParams.get("live") === "1";

// Lane assignment for live trace event kinds.
const LIVE_EVENT_LANE = {
  capture_started: "Mic",
  speech_started: "Mic",
  breath_group_opened: "Mic",
  breath_group_closed: "Mic",
  asr_started: "ASR",
  asr_finished: "ASR",
  transcript: "ASR",
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

// Accumulated live trace events.
const liveEvents = [];
let liveRenderScheduled = false;
// Debounce interval for live re-renders (ms). Balances UI responsiveness vs. render cost.
const LIVE_RENDER_DEBOUNCE_MS = 80;

const state = {
  payload: null,
  lanes: [],
  selectedItem: null,
  maxDurationMs: 1000,
  stopAtMs: null,
};

const sourceLabels = {
  RecordedAudio: "Recorded audio",
  LiveAsr: "Live ASR",
  GeneratedText: "Generated text",
  SyntheticSpeech: "Synthetic speech",
};

loadDemoButton.addEventListener("click", () => loadDemo());
jsonFileInput.addEventListener("change", (event) => readJsonFile(event.target.files?.[0]));
audioFileInput.addEventListener("change", (event) => readAudioFile(event.target.files?.[0]));
playPauseButton.addEventListener("click", () => togglePlayback());
jumpPrevButton.addEventListener("click", () => jumpSelectedWord(-1));
jumpNextButton.addEventListener("click", () => jumpSelectedWord(1));
playSelectionClipButton.addEventListener("click", () => playSelectedClip());

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
  if (isLiveMode) {
    enterLiveMode();
    return;
  }

  const payloadMode = queryParams.get("payload");
  if (payloadMode === "demo") {
    await loadPayloadFromUrls(["/api/demo-payload", "./demo.json"], "Loaded bundled demo.");
    return;
  }
  if (payloadMode === "provided") {
    const loaded = await loadPayloadFromUrls(["/api/payload"], "Loaded --payload JSON.");
    if (loaded) {
      return;
    }
  }
  if (payloadMode === "trace") {
    const loaded = await loadPayloadFromUrls(
      ["/api/trace-viewer-payload"],
      "Loaded --trace viewer payload conversion.",
    );
    if (loaded) {
      return;
    }
  }
  await loadDemo();
}

function enterLiveMode() {
  document.body.classList.add("live-mode");
  liveBanner.hidden = false;
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

  source.onerror = () => {
    liveConnectionStatus.textContent = "disconnected";
    liveConnectionStatus.className = "live-status-error";
    statusMessage.textContent = "Live event stream disconnected. Session may have ended.";
    source.close();
  };
}

function addLiveEvent(traceEvent) {
  liveEvents.push(traceEvent);
  liveEventCount.textContent = `${liveEvents.length} event${liveEvents.length === 1 ? "" : "s"}`;

  if (!liveRenderScheduled) {
    liveRenderScheduled = true;
    setTimeout(() => {
      liveRenderScheduled = false;
      applyLiveEvents();
    }, LIVE_RENDER_DEBOUNCE_MS);
  }
}

function applyLiveEvents() {
  const payload = buildLivePayload(liveEvents);
  applyPayload(payload);
  viewerTitle.textContent = "Live — Listenbury";
}

function buildLivePayload(events) {
  // Group events by lane and convert to ViewerPayload-compatible format.
  const viewerEvents = [];
  const viewerMarkers = [];

  // Track open spans (start events without a matching end).
  const openSpans = new Map(); // key: `${lane}:${turn}:${startKind}` → start_ms

  // Pairs: when we see the "end" event, close the span started by the "start" event.
  const spanPairs = {
    asr_started: { end: "asr_finished", lane: "ASR" },
    playback_started: { end: "playback_finished", lane: "Speaker" },
    llm_generation_started: { end: "playback_started", lane: "LLM" },
    self_hearing_suppression_started: { end: "self_hearing_suppression_ended", lane: "Speaker" },
  };
  const endToStart = {};
  for (const [startKind, info] of Object.entries(spanPairs)) {
    endToStart[info.end] = { startKind, lane: info.lane };
  }

  function openSpanKey(lane, turn, startKind) {
    return JSON.stringify([lane, turn ?? null, startKind]);
  }

  for (const event of events) {
    const lane = LIVE_EVENT_LANE[event.kind] ?? "Events";

    // Check if this closes a span.
    const startInfo = endToStart[event.kind];
    if (startInfo) {
      const openKey = openSpanKey(startInfo.lane, event.turn, startInfo.startKind);
      const spanStart = openSpans.get(openKey);
      if (spanStart !== undefined) {
        openSpans.delete(openKey);
        viewerEvents.push({
          lane: startInfo.lane,
          kind: startInfo.startKind,
          label: labelForKind(startInfo.startKind),
          start_ms: spanStart,
          end_ms: event.elapsed_ms,
          metadata: event,
        });
        continue;
      }
    }

    // Check if this opens a span.
    if (spanPairs[event.kind]) {
      const spanKey = openSpanKey(lane, event.turn, event.kind);
      openSpans.set(spanKey, event.elapsed_ms);
      // Don't emit yet – wait for the closing event.
      continue;
    }

    // Emit as a marker or event.
    const label = event.text ? event.text.slice(0, 60) : labelForKind(event.kind);
    viewerMarkers.push({
      lane,
      kind: event.kind,
      label,
      at_ms: event.elapsed_ms,
      metadata: { event },
    });
  }

  // Flush any unclosed spans as open-ended spans up to current max elapsed_ms.
  const maxMs = Math.max(0, ...events.map((e) => e.elapsed_ms));
  for (const [key, startMs] of openSpans.entries()) {
    const [lane, turn, kind] = JSON.parse(key);
    viewerEvents.push({
      lane,
      kind,
      label: `${labelForKind(kind)} (in progress)`,
      start_ms: startMs,
      end_ms: maxMs,
      metadata: {
        in_progress: true,
        turn: turn,
      },
    });
  }

  return {
    title: "Live — Listenbury",
    streams: [],
    events: viewerEvents,
    markers: viewerMarkers,
  };
}

function labelForKind(kind) {
  return kind.replace(/_/g, " ");
}


async function loadDemo() {
  await loadPayloadFromUrls(["/api/demo-payload", "./demo.json"], "Loaded bundled demo.");
}

async function loadPayloadFromUrls(urls, successMessage) {
  try {
    const failures = [];
    for (const url of urls) {
      const response = await fetch(url);
      if (!response.ok) {
        failures.push(`${url} (${response.status})`);
        continue;
      }
      const payload = await response.json();
      applyPayload(payload);
      statusMessage.textContent = successMessage;
      return true;
    }
    throw new Error(`failed to load payload from ${failures.join(", ") || urls.join(", ")}`);
  } catch (error) {
    statusMessage.textContent =
      `Unable to auto-load demo (${error?.message ?? "unknown error"}). Serve the repository over local HTTP or choose a JSON file manually.`;
    console.error(error);
    return false;
  }
}

async function readJsonFile(file) {
  if (!file) {
    return;
  }

  try {
    const payload = JSON.parse(await file.text());
    applyPayload(payload);
    statusMessage.textContent = `Loaded ${file.name}.`;
  } catch (error) {
    statusMessage.textContent = `Failed to parse ${file.name}.`;
    console.error(error);
  }
}

function readAudioFile(file) {
  if (!file) {
    return;
  }

  audio.src = URL.createObjectURL(file);
  statusMessage.textContent = `Loaded audio override ${file.name}.`;
}

function togglePlayback() {
  if (!audio.src) {
    statusMessage.textContent = "No audio loaded. Choose an audio file or load a payload with audio.url.";
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

function applyPayload(rawPayload) {
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
  state.selectedItem = firstItemSelection();
  clearPlaybackStop();
  configureAudio(normalized.audio);
  syncMaxDurationWithAudio();
  render();
}

function normalizePayload(rawPayload) {
  if (isTimedWordStream(rawPayload)) {
    return {
      title: VIEWER_NAME,
      audio: null,
      streams: [{ label: defaultLaneLabel(rawPayload, 0), stream: rawPayload }],
      events: [],
    };
  }

  if (Array.isArray(rawPayload)) {
    return {
      title: VIEWER_NAME,
      audio: null,
      streams: rawPayload.map((stream, index) => ({
        label: defaultLaneLabel(stream, index),
        stream,
      })),
      events: [],
    };
  }

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

  throw new Error(
    "payload must be a TimedWordStream, an array of TimedWordStreams, or an object with streams/events",
  );
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
}

function render() {
  viewerTitle.textContent = state.payload?.title ?? "No stream loaded";
  playbackTime.textContent = formatPlaybackTime();
  playPauseButton.textContent = audio.paused ? "Play" : "Pause";

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
  const ruler = document.createElement("div");
  ruler.className = "timeline-ruler";

  const ticks = buildRulerTicks(state.maxDurationMs);
  ticks.forEach((tickMs) => {
    const tick = document.createElement("span");
    tick.className = "ruler-tick";
    tick.style.left = `${(tickMs / state.maxDurationMs) * 100}%`;

    const label = document.createElement("span");
    label.className = "ruler-label";
    label.style.left = `${(tickMs / state.maxDurationMs) * 100}%`;
    label.textContent = formatRulerLabel(tickMs);

    ruler.append(tick, label);
  });

  ruler.addEventListener("click", (event) => {
    if (!audio.src) {
      return;
    }
    const rect = ruler.getBoundingClientRect();
    const ratio = Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width));
    audio.currentTime = (ratio * state.maxDurationMs) / 1000;
    refreshPlaybackState();
  });

  return ruler;
}

function renderWordLane(lane) {
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

  lane.words.forEach((word) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "timeline-chip word-chip";
    if (word.timingResolution === "fallback-layout") {
      chip.classList.add("fallback-timing");
    }
    chip.dataset.key = itemKey("word", word.laneIndex, word.wordIndex);
    chip.dataset.itemType = "word";
    chip.dataset.startMs = String(word.resolvedTiming.start_ms);
    chip.dataset.endMs = String(word.resolvedTiming.end_ms);
    chip.textContent = word.text;
    chip.title = `${word.text} (${word.resolvedTiming.start_ms}–${word.resolvedTiming.end_ms} ms) · ${word.timingSourceDetail}`;

    const left = (word.resolvedTiming.start_ms / state.maxDurationMs) * 100;
    const width = Math.max(
      6,
      ((Math.max(word.resolvedTiming.end_ms, word.resolvedTiming.start_ms + 1) - word.resolvedTiming.start_ms) /
        state.maxDurationMs) *
        100,
    );

    chip.style.left = `${Math.min(left, 96)}%`;
    chip.style.width = `${Math.min(width, 100 - left)}%`;
    chip.addEventListener("click", () => selectWord(word.laneIndex, word.wordIndex, true));
    track.append(chip);
  });

  section.append(track);
  return section;
}

function renderEventLane(lane) {
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

  lane.events.forEach((event) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = `timeline-chip event-chip ${event.style} kind-${classToken(event.kind)}`;
    chip.dataset.key = itemKey("event", event.laneIndex, event.eventIndex);
    chip.dataset.itemType = "event";
    chip.dataset.startMs = String(event.start_ms);
    chip.dataset.endMs = String(event.end_ms);
    chip.textContent = event.label;
    chip.title = `${event.kind} (${event.start_ms}–${event.end_ms} ms)`;

    const left = (event.start_ms / state.maxDurationMs) * 100;
    const width =
      event.style === "marker"
        ? 2
        : Math.max(3, ((Math.max(event.end_ms, event.start_ms + 1) - event.start_ms) / state.maxDurationMs) * 100);

    chip.style.left = `${Math.min(left, 99)}%`;
    chip.style.width = `${Math.min(width, 100 - left)}%`;
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
      boundarySource: word.boundary_source,
      lexicalSpan: word.lexical_span,
      audioRef: word.audio_ref,
    },
    null,
    2,
  );
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

function buildRulerTicks(durationMs) {
  const safeDuration = Math.max(1000, durationMs);
  const targetSegments = 10;
  const preferredSteps = [250, 500, 1000, 2000, 5000, 10_000, 15_000, 30_000, 60_000];
  const desiredStep = safeDuration / targetSegments;
  const stepMs = preferredSteps.find((step) => step >= desiredStep) ?? 120_000;
  const ticks = [];
  for (let at = 0; at <= safeDuration; at += stepMs) {
    ticks.push(at);
  }
  if (ticks[ticks.length - 1] !== safeDuration) {
    ticks.push(safeDuration);
  }
  return ticks;
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
