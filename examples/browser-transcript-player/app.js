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
const audio = document.getElementById("audio");
const queryParams = new URLSearchParams(window.location.search);
const VIEWER_NAME = "WaveDeck";

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

audio.addEventListener("timeupdate", () => {
  if (state.stopAtMs !== null && audio.currentTime * 1000 >= state.stopAtMs) {
    audio.pause();
    state.stopAtMs = null;
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
  state.stopAtMs = null;
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
  playbackTime.textContent = `${audio.currentTime.toFixed(3)}s / ${(state.maxDurationMs / 1000).toFixed(3)}s`;
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
  playbackTime.textContent = `${audio.currentTime.toFixed(3)}s / ${(state.maxDurationMs / 1000).toFixed(3)}s`;
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

  state.stopAtMs = null;
  refreshPlaybackState();
  renderSelection();
}

function selectEvent(laneIndex, eventIndex, seekAudio) {
  const event = state.lanes[laneIndex]?.events?.[eventIndex];
  if (!event) {
    return;
  }

  state.selectedItem = { type: "event", laneIndex, itemIndex: eventIndex };
  if (seekAudio && audio.src) {
    audio.currentTime = event.start_ms / 1000;
  }

  state.stopAtMs = null;
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
