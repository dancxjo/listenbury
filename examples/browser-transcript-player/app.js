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

const state = {
  payload: null,
  lanes: [],
  selectedWord: null,
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

void loadDemo();

async function loadDemo() {
  try {
    const response = await fetch("./demo.json");
    if (!response.ok) {
      throw new Error(`demo fetch failed (${response.status})`);
    }
    const payload = await response.json();
    applyPayload(payload);
    statusMessage.textContent = "Loaded bundled demo.";
  } catch (error) {
    statusMessage.textContent =
      "Unable to auto-load demo. Serve the repository over local HTTP or choose a JSON file manually.";
    console.error(error);
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
  if (state.selectedWord) {
    const selectedKey = wordKey(state.selectedWord.laneIndex, state.selectedWord.wordIndex);
    index = words.findIndex((word) => wordKey(word.laneIndex, word.wordIndex) === selectedKey);
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
  state.payload = normalized;
  state.lanes = normalized.streams.map((lane, laneIndex) => normalizeLane(lane, laneIndex));
  state.selectedWord = firstWordSelection();
  state.stopAtMs = null;
  configureAudio(normalized.audio);
  syncMaxDurationWithAudio();
  render();
}

function normalizePayload(rawPayload) {
  if (isTimedWordStream(rawPayload)) {
    return {
      title: "TimedWordStream viewer",
      audio: null,
      streams: [{ label: defaultLaneLabel(rawPayload, 0), stream: rawPayload }],
    };
  }

  if (Array.isArray(rawPayload)) {
    return {
      title: "TimedWordStream viewer",
      audio: null,
      streams: rawPayload.map((stream, index) => ({
        label: defaultLaneLabel(stream, index),
        stream,
      })),
    };
  }

  if (rawPayload && Array.isArray(rawPayload.streams)) {
    return {
      title: rawPayload.title ?? "TimedWordStream viewer",
      audio: rawPayload.audio ?? null,
      streams: rawPayload.streams.map((entry, index) => {
        if (isTimedWordStream(entry)) {
          return { label: defaultLaneLabel(entry, index), stream: entry };
        }
        if (entry?.stream && isTimedWordStream(entry.stream)) {
          return { label: entry.label ?? defaultLaneLabel(entry.stream, index), stream: entry.stream };
        }
        throw new Error(`stream entry ${index} is not a TimedWordStream`);
      }),
    };
  }

  throw new Error("payload must be a TimedWordStream, an array of TimedWordStreams, or an object with streams");
}

function normalizeLane(lane, laneIndex) {
  const totalWords = lane.stream.words.length || 1;
  return {
    ...lane,
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
        laneIndex,
        wordIndex,
      };
    }),
  };
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
    ...state.lanes.flatMap((lane) => lane.words.map((word) => word.resolvedTiming.end_ms)),
  );
  const fromAudio = Number.isFinite(audio.duration) ? Math.round(audio.duration * 1000) : 0;
  state.maxDurationMs = Math.max(fromPayload, fromStreams, fromAudio, 1000);
}

function render() {
  viewerTitle.textContent = state.payload?.title ?? "No stream loaded";
  playbackTime.textContent = `${audio.currentTime.toFixed(3)}s`;
  playPauseButton.textContent = audio.paused ? "Play" : "Pause";

  if (!state.lanes.length) {
    viewer.className = "viewer empty";
    viewer.innerHTML = "<p>No streams loaded yet.</p>";
    renderSelection();
    return;
  }

  viewer.className = "viewer";
  viewer.replaceChildren(...state.lanes.map(renderLane));
  refreshPlaybackState();
  renderSelection();
}

function renderLane(lane) {
  const section = document.createElement("section");
  section.className = "lane";

  const header = document.createElement("div");
  header.className = "lane-header";

  const title = document.createElement("h2");
  title.textContent = lane.label;

  const meta = document.createElement("div");
  meta.className = "lane-meta";
  meta.textContent = `${sourceLabels[lane.stream.source] ?? lane.stream.source} · ${lane.stream.words.length} words`;

  header.append(title, meta);
  section.append(header);

  const track = document.createElement("div");
  track.className = "lane-track";

  lane.words.forEach((word) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "word-chip";
    if (word.timingResolution === "fallback-layout") {
      chip.classList.add("fallback-timing");
    }
    chip.dataset.key = wordKey(word.laneIndex, word.wordIndex);
    chip.dataset.startMs = String(word.resolvedTiming.start_ms);
    chip.dataset.endMs = String(word.resolvedTiming.end_ms);
    chip.textContent = word.text;
    chip.title = `${word.text} (${word.resolvedTiming.start_ms}–${word.resolvedTiming.end_ms} ms) · ${word.timingSourceDetail}`;

    const left = (word.resolvedTiming.start_ms / state.maxDurationMs) * 100;
    const width = Math.max(
      6,
      ((Math.max(word.resolvedTiming.end_ms, word.resolvedTiming.start_ms + 1) -
        word.resolvedTiming.start_ms) /
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

function refreshPlaybackState() {
  playbackTime.textContent = `${audio.currentTime.toFixed(3)}s`;
  playPauseButton.textContent = audio.paused ? "Play" : "Pause";
  const nowMs = Math.round(audio.currentTime * 1000);

  document.querySelectorAll(".word-chip").forEach((chip) => {
    const startMs = Number(chip.dataset.startMs);
    const endMs = Number(chip.dataset.endMs);
    chip.classList.toggle("active", nowMs >= startMs && nowMs <= endMs);
    chip.classList.toggle(
      "selected",
      state.selectedWord !== null &&
        chip.dataset.key === wordKey(state.selectedWord.laneIndex, state.selectedWord.wordIndex),
    );
  });
}

function renderSelection() {
  if (!state.selectedWord) {
    selectionSummary.textContent =
      "Select a word to inspect timing, confidence, and stream metadata.";
    selectionJson.textContent = "{}";
    return;
  }

  const lane = state.lanes[state.selectedWord.laneIndex];
  const word = lane.words[state.selectedWord.wordIndex];
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
  state.selectedWord = { laneIndex, wordIndex };
  const word = state.lanes[laneIndex]?.words[wordIndex];
  if (!word) {
    return;
  }

  if (seekAudio && audio.src) {
    audio.currentTime = word.resolvedTiming.start_ms / 1000;
  }

  state.stopAtMs = null;
  refreshPlaybackState();
  renderSelection();
}

function firstWordSelection() {
  const firstLane = state.lanes.find((lane) => lane.words.length > 0);
  return firstLane ? { laneIndex: firstLane.words[0].laneIndex, wordIndex: 0 } : null;
}

function flattenWords() {
  return state.lanes
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

function wordKey(laneIndex, wordIndex) {
  return `${laneIndex}:${wordIndex}`;
}
