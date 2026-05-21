import {
  assembleNarrativeManuscript,
  buildNarrativeEpisode,
  createNarrativeSession,
  reduceNarrativeEvent,
} from "/assets/screenplay-model.mjs";
import { toTitleCase } from "/assets/scene-heading.mjs";
import {
  isTraceSessionEnvelope,
  parseTraceEventsJsonl,
  traceSessionLabel,
} from "/assets/trace-session.mjs";

const scriptRoot = document.getElementById("script");
const statusEl = document.getElementById("connection-status");
const eventCountEl = document.getElementById("event-count");
const dotEl = document.getElementById("connection-dot");
const micSourceEl = document.getElementById("mic-source");
const playButtonEl = document.getElementById("screenplay-play");
const recordButtonEl = document.getElementById("screenplay-record");
const micErrorEl = document.getElementById("screenplay-mic-error");
const playbackAudioEl = document.getElementById("screenplay-audio");

const RENDER_DEBOUNCE_MS = 60;
const MAX_DELETED_SNIPPETS = 8;
const PLACEHOLDER_SCENE_HEADING = "INT./EXT. UNKNOWN LOCATION - PRESENT";
const PLACEHOLDER_ACTION =
  "Scene headings and action are provisional until enough live context arrives to form scenes and an episode.";
const BROWSER_AUDIO_SAMPLE_RATE_HZ = 16_000;
const BROWSER_AUDIO_FRAME_SAMPLES = BROWSER_AUDIO_SAMPLE_RATE_HZ / 100;
const MIC_SOURCE_BROWSER = "browser";
const MIC_SOURCE_NATIVE = "native";

const session = createNarrativeSession();
const pageState = {
  connectionStatus: "connecting",
  connectionClass: "is-connecting",
  message: "Waiting for live transcription...",
};

let renderScheduled = false;
const inputUiState = {
  browserAudioAvailable: false,
  browserRecording: false,
  nativeMicAvailable: false,
  nativeMicEnabled: false,
  selectedMicSource: MIC_SOURCE_BROWSER,
  micControlUpdating: false,
  browserError: null,
};
const browserAudioState = {
  stream: null,
  audioContext: null,
  source: null,
  processor: null,
  pendingSamples: new Float32Array(0),
  sendChain: Promise.resolve(),
};

void initializeScreenplay();
render();

async function initializeScreenplay() {
  bindInputChrome();
  await refreshInputStatus();
  const loadedAttachedTrace = await loadAttachedTraceSession();
  if (!loadedAttachedTrace) {
    connectLiveEvents();
  }
}

function connectLiveEvents() {
  const source = new EventSource("/api/live-events");

  source.onopen = () => {
    pageState.connectionStatus = "connected";
    pageState.connectionClass = "is-connected";
    pageState.message = "Listening for live transcription...";
    scheduleRender();
  };

  source.onmessage = (event) => {
    try {
      reduceNarrativeEvent(session, JSON.parse(event.data));
      scheduleRender();
    } catch (error) {
      console.error("Failed to parse live event:", error, event.data);
    }
  };

  source.addEventListener("live-unavailable", (event) => {
    let message = "Live events are unavailable. Start a listen session with --web.";
    try {
      const payload = JSON.parse(event.data);
      message = payload.message || message;
    } catch (error) {
      console.error("Failed to parse live availability event:", error, event.data);
    }
    pageState.connectionStatus = "unavailable";
    pageState.connectionClass = "is-error";
    pageState.message = message;
    scheduleRender();
    source.close();
  });

  source.onerror = () => {
    pageState.connectionStatus = "disconnected";
    pageState.connectionClass = "is-error";
    pageState.message = "Live event stream disconnected.";
    scheduleRender();
    source.close();
  };
}

async function loadAttachedTraceSession() {
  try {
    const sessionResponse = await fetch("/api/trace-session");
    if (sessionResponse.ok) {
      const sessionEnvelope = await sessionResponse.json();
      if (isTraceSessionEnvelope(sessionEnvelope)) {
        applyRecordedEvents(sessionEnvelope.events);
        pageState.connectionStatus = "recorded";
        pageState.connectionClass = "is-connected";
        pageState.message = `Loaded recorded session: ${traceSessionLabel(sessionEnvelope, "attached trace")}`;
        scheduleRender();
        return true;
      }
    }
    const traceResponse = await fetch("/api/trace");
    if (!traceResponse.ok) {
      return false;
    }
    const { events } = parseTraceEventsJsonl(await traceResponse.text());
    if (!events.length) {
      return false;
    }
    applyRecordedEvents(events);
    pageState.connectionStatus = "recorded";
    pageState.connectionClass = "is-connected";
    pageState.message = "Loaded recorded trace.";
    scheduleRender();
    return true;
  } catch (error) {
    console.error("Failed to load recorded trace:", error);
  }
  return false;
}

function applyRecordedEvents(events) {
  for (const event of events) {
    reduceNarrativeEvent(session, event);
  }
}

function scheduleRender() {
  if (renderScheduled) {
    return;
  }
  renderScheduled = true;
  window.setTimeout(() => {
    renderScheduled = false;
    render();
  }, RENDER_DEBOUNCE_MS);
}

function render() {
  statusEl.textContent = pageState.connectionStatus;
  eventCountEl.textContent = `${session.eventCount} event${session.eventCount === 1 ? "" : "s"}`;
  dotEl.className = `connection-dot ${pageState.connectionClass}`;
  renderInputChrome();

  const episode = buildNarrativeEpisode(session, { id: "episode-live", episodeNumber: 1, sessionLabel: "Live screenplay" });
  const manuscript = assembleNarrativeManuscript([episode]);

  scriptRoot.replaceChildren();
  if (!episode.scenes.length) {
    scriptRoot.append(sceneHeading(PLACEHOLDER_SCENE_HEADING), actionLine(PLACEHOLDER_ACTION));
    appendPropositionBlock(scriptRoot);
    const empty = document.createElement("p");
    empty.className = "empty-page";
    empty.textContent = pageState.message;
    scriptRoot.append(empty);
    return;
  }

  appendPropositionBlock(scriptRoot);
  scriptRoot.append(episodeHeader(episode, manuscript));
  for (const scene of episode.scenes) {
    scriptRoot.append(sceneSection(scene));
  }

  window.requestAnimationFrame(() => {
    window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
  });
}

function episodeHeader(episode, manuscript) {
  const section = document.createElement("section");
  section.className = "episode-meta";

  const title = document.createElement("p");
  title.className = "status-line";
  title.textContent = episode.title;

  const summary = document.createElement("p");
  summary.className = "action";
  summary.textContent = episode.summary;

  const metadata = document.createElement("p");
  metadata.className = "soft-note";
  metadata.textContent = formatEpisodeMetadata(episode, manuscript);

  const displayPolicy = document.createElement("p");
  displayPolicy.className = "soft-note";
  displayPolicy.textContent = "Display policy: committed text is plain; prospective text stays tinted; deleted/cancelled text remains traceable.";

  const sceneList = document.createElement("ol");
  sceneList.className = "scene-list";
  for (const scene of episode.sceneList) {
    const item = document.createElement("li");
    const link = document.createElement("a");
    link.setAttribute("href", `#${scene.id}`);
    link.textContent = scene.heading;
    const detail = document.createElement("span");
    detail.textContent = ` — ${scene.summary}`;
    item.append(link, detail);
    sceneList.append(item);
  }

  section.append(title, summary, metadata, displayPolicy, sceneList);
  return section;
}

function sceneSection(scene) {
  const section = document.createElement("section");
  section.className = "scene";
  section.id = scene.id;

  section.append(sceneHeading(scene.heading), actionLine(scene.action));

  if (scene.topicLabel) {
    const topicNote = document.createElement("p");
    topicNote.className = "soft-note";
    topicNote.textContent = `Topic: ${toTitleCase(scene.topicLabel)}`;
    section.append(topicNote);
  }

  const summary = document.createElement("p");
  summary.className = "soft-note";
  summary.textContent = scene.summary;
  section.append(summary);

  for (const beat of scene.beats) {
    if (beat.role) {
      const block = document.createElement("section");
      block.className = "turn";
      block.append(characterCue(beat.role), dialogueLine(beat.segments ?? [{ text: beat.text }]));
      section.append(block);
    } else {
      const action = actionLine(beat.text);
      action.classList.add("soft-note");
      section.append(action);
    }
  }

  return section;
}

function sceneHeading(text) {
  const heading = document.createElement("p");
  heading.className = "scene-heading";
  heading.textContent = text;
  return heading;
}

function actionLine(text) {
  const action = document.createElement("p");
  action.className = "action";
  action.textContent = text;
  return action;
}

function characterCue(name) {
  const cue = document.createElement("p");
  cue.className = "character";
  cue.textContent = name;
  return cue;
}

function dialogueLine(segments) {
  const line = document.createElement("p");
  line.className = "dialogue";
  for (const segment of segments) {
    const span = document.createElement("span");
    if (segment.className) {
      span.className = segment.className;
    }
    span.textContent = segment.text;
    if (segment.spanMetadata?.length) {
      span.dataset.spanMetadata = JSON.stringify(segment.spanMetadata);
      span.title = `span metadata: ${segment.spanMetadata.length} item${segment.spanMetadata.length === 1 ? "" : "s"}`;
      span.setAttribute(
        "aria-label",
        `${segment.text} (span metadata: ${segment.spanMetadata.length} item${segment.spanMetadata.length === 1 ? "" : "s"})`,
      );
    }
    line.append(span);
  }
  return line;
}

function appendPropositionBlock(parent) {
  if (!session.proposition?.text) {
    return;
  }

  const block = document.createElement("section");
  block.className = "turn proposition-turn";
  block.append(characterCue("PROPOSITION"), dialogueLine(propositionSegments()));
  parent.append(block);
}

function propositionSegments() {
  const segments = [{ text: session.proposition.text, className: "prospective-asr" }];
  for (const item of session.propositionDeleted.slice(-MAX_DELETED_SNIPPETS)) {
    if (item.text) {
      segments.push({ text: item.text, className: "deleted" });
    }
  }
  return segments;
}

function formatEpisodeMetadata(episode, manuscript) {
  return [
    manuscript.chapters[0]?.title ?? "Chapter",
    `${episode.metadata.sessionTurns} turns`,
    `${episode.metadata.eventCount} events`,
    `${episode.metadata.startedAtMs}ms-${episode.metadata.endedAtMs}ms`,
  ].join(" • ");
}

function bindInputChrome() {
  micSourceEl?.addEventListener("change", async (event) => {
    const source = event.currentTarget.value;
    if (source !== MIC_SOURCE_BROWSER && source !== MIC_SOURCE_NATIVE) {
      return;
    }
    inputUiState.selectedMicSource = source;
    renderInputChrome();
    if (isMicRecordingActive()) {
      await applyMicRecordingState(true);
    }
  });
  playButtonEl?.addEventListener("click", () => {
    void toggleScreenplayPlayback();
  });
  recordButtonEl?.addEventListener("click", () => {
    void applyMicRecordingState(!isMicRecordingActive());
  });
  playbackAudioEl?.addEventListener("play", renderInputChrome);
  playbackAudioEl?.addEventListener("pause", renderInputChrome);
}

function renderInputChrome() {
  if (!micSourceEl || !recordButtonEl || !playButtonEl || !micErrorEl) {
    return;
  }
  ensureSelectedMicSource();
  micSourceEl.value = inputUiState.selectedMicSource;
  micSourceEl.disabled = inputUiState.micControlUpdating || !hasAvailableMicSource();
  for (const option of micSourceEl.options) {
    if (option.value === MIC_SOURCE_BROWSER) {
      option.disabled = !inputUiState.browserAudioAvailable;
    } else if (option.value === MIC_SOURCE_NATIVE) {
      option.disabled = !inputUiState.nativeMicAvailable;
    }
  }
  const recordingActive = isMicRecordingActive();
  const canRecord = canRecordSelectedMic();
  recordButtonEl.classList.toggle("record-active", recordingActive);
  recordButtonEl.setAttribute("aria-pressed", recordingActive ? "true" : "false");
  recordButtonEl.setAttribute("aria-label", recordingActive ? "Pause recording" : "Start recording");
  recordButtonEl.title = recordingActive ? "Pause recording" : "Start recording";
  recordButtonEl.disabled = inputUiState.micControlUpdating || !canRecord;
  playButtonEl.textContent = playbackAudioEl?.paused ? "▶︎" : "❚❚";
  micErrorEl.textContent = inputUiState.browserError ?? "";
  micErrorEl.hidden = !inputUiState.browserError;
}

function hasAvailableMicSource() {
  return inputUiState.browserAudioAvailable || inputUiState.nativeMicAvailable;
}

function canRecordSelectedMic() {
  return inputUiState.selectedMicSource === MIC_SOURCE_NATIVE
    ? inputUiState.nativeMicAvailable
    : inputUiState.browserAudioAvailable;
}

function isMicRecordingActive() {
  return inputUiState.selectedMicSource === MIC_SOURCE_NATIVE
    ? inputUiState.nativeMicEnabled
    : inputUiState.browserRecording;
}

function ensureSelectedMicSource() {
  if (inputUiState.selectedMicSource === MIC_SOURCE_NATIVE && inputUiState.nativeMicAvailable) {
    return;
  }
  if (inputUiState.selectedMicSource === MIC_SOURCE_BROWSER && inputUiState.browserAudioAvailable) {
    return;
  }
  if (inputUiState.nativeMicEnabled && inputUiState.nativeMicAvailable) {
    inputUiState.selectedMicSource = MIC_SOURCE_NATIVE;
    return;
  }
  if (inputUiState.browserAudioAvailable) {
    inputUiState.selectedMicSource = MIC_SOURCE_BROWSER;
    return;
  }
  if (inputUiState.nativeMicAvailable) {
    inputUiState.selectedMicSource = MIC_SOURCE_NATIVE;
  }
}

async function refreshInputStatus() {
  try {
    const response = await fetch("/api/input-status", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(await response.text());
    }
    const status = await response.json();
    inputUiState.browserAudioAvailable = Boolean(status.browserMic?.available);
    inputUiState.nativeMicAvailable = Boolean(status.nativeMic?.available);
    inputUiState.nativeMicEnabled = Boolean(status.nativeMic?.enabled);
    inputUiState.browserError = null;
  } catch (error) {
    inputUiState.browserAudioAvailable = false;
    inputUiState.nativeMicAvailable = false;
    inputUiState.nativeMicEnabled = false;
    inputUiState.browserError = conciseErrorMessage(error);
  } finally {
    ensureSelectedMicSource();
    renderInputChrome();
  }
}

async function applyMicRecordingState(enableRecording) {
  ensureSelectedMicSource();
  if (inputUiState.micControlUpdating) {
    return;
  }
  if (enableRecording && !canRecordSelectedMic()) {
    return;
  }
  inputUiState.micControlUpdating = true;
  renderInputChrome();
  try {
    if (inputUiState.selectedMicSource === MIC_SOURCE_NATIVE) {
      stopBrowserRecording({ render: false });
      await setBrowserMicEnabled(false);
      await setNativeMicEnabled(enableRecording);
    } else {
      await setNativeMicEnabled(false);
      await setBrowserMicEnabled(enableRecording);
      if (enableRecording) {
        await startBrowserRecording();
      } else {
        stopBrowserRecording({ render: false });
      }
    }
    inputUiState.browserError = null;
  } catch (error) {
    stopBrowserRecording({ render: false });
    inputUiState.browserError = conciseErrorMessage(error);
  } finally {
    inputUiState.micControlUpdating = false;
    ensureSelectedMicSource();
    renderInputChrome();
  }
}

async function setNativeMicEnabled(enabled) {
  if (!inputUiState.nativeMicAvailable) {
    return;
  }
  const response = await fetch("/api/native-mic", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ enabled }),
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  const status = await response.json();
  inputUiState.nativeMicAvailable = Boolean(status.nativeMic?.available);
  inputUiState.nativeMicEnabled = Boolean(status.nativeMic?.enabled);
  inputUiState.browserAudioAvailable = Boolean(status.browserMic?.available);
}

async function setBrowserMicEnabled(enabled) {
  if (!inputUiState.browserAudioAvailable) {
    return;
  }
  const response = await fetch("/api/browser-mic", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ enabled }),
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  const status = await response.json();
  inputUiState.browserAudioAvailable = Boolean(status.browserMic?.available);
  inputUiState.nativeMicAvailable = Boolean(status.nativeMic?.available);
  inputUiState.nativeMicEnabled = Boolean(status.nativeMic?.enabled);
}

async function startBrowserRecording() {
  if (inputUiState.browserRecording) {
    return;
  }
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("Browser microphone capture is not available.");
  }
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
  });
  const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
  const audioContext = new AudioContextCtor();
  const source = audioContext.createMediaStreamSource(stream);
  const processor = audioContext.createScriptProcessor(4096, source.channelCount || 1, 1);
  processor.onaudioprocess = (event) => {
    const samples = resampleInputBufferToMono(
      event.inputBuffer,
      audioContext.sampleRate,
      BROWSER_AUDIO_SAMPLE_RATE_HZ,
    );
    queueBrowserAudioSamples(samples);
  };
  source.connect(processor);
  processor.connect(audioContext.destination);
  browserAudioState.stream = stream;
  browserAudioState.audioContext = audioContext;
  browserAudioState.source = source;
  browserAudioState.processor = processor;
  inputUiState.browserRecording = true;
}

function stopBrowserRecording(options = {}) {
  const { render = true } = options;
  browserAudioState.processor?.disconnect();
  browserAudioState.source?.disconnect();
  for (const track of browserAudioState.stream?.getTracks?.() ?? []) {
    track.stop();
  }
  void browserAudioState.audioContext?.close?.();
  browserAudioState.stream = null;
  browserAudioState.audioContext = null;
  browserAudioState.source = null;
  browserAudioState.processor = null;
  flushBrowserAudioSamples();
  inputUiState.browserRecording = false;
  if (render) {
    renderInputChrome();
  }
}

function queueBrowserAudioSamples(samples) {
  if (!samples.length) {
    return;
  }
  const combined = new Float32Array(browserAudioState.pendingSamples.length + samples.length);
  combined.set(browserAudioState.pendingSamples, 0);
  combined.set(samples, browserAudioState.pendingSamples.length);
  let offset = 0;
  while (combined.length - offset >= BROWSER_AUDIO_FRAME_SAMPLES) {
    queueBrowserAudioChunk(combined.slice(offset, offset + BROWSER_AUDIO_FRAME_SAMPLES));
    offset += BROWSER_AUDIO_FRAME_SAMPLES;
  }
  browserAudioState.pendingSamples = combined.slice(offset);
}

function flushBrowserAudioSamples() {
  if (!browserAudioState.pendingSamples.length) {
    return;
  }
  const padded = new Float32Array(BROWSER_AUDIO_FRAME_SAMPLES);
  padded.set(browserAudioState.pendingSamples.slice(0, BROWSER_AUDIO_FRAME_SAMPLES));
  browserAudioState.pendingSamples = new Float32Array(0);
  queueBrowserAudioChunk(padded);
}

function queueBrowserAudioChunk(samples) {
  if (!samples.length) {
    return;
  }
  const body = samples.slice().buffer;
  browserAudioState.sendChain = browserAudioState.sendChain
    .catch(() => {})
    .then(async () => {
      const response = await fetch("/api/browser-audio", {
        method: "POST",
        headers: {
          "Content-Type": "application/octet-stream",
          "X-Sample-Rate-Hz": String(BROWSER_AUDIO_SAMPLE_RATE_HZ),
          "X-Channels": "1",
        },
        body,
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
    })
    .catch((error) => {
      inputUiState.browserError = conciseErrorMessage(error);
      renderInputChrome();
    });
}

function resampleInputBufferToMono(inputBuffer, fromSampleRate, toSampleRate) {
  const channelCount = inputBuffer.numberOfChannels || 1;
  const inputLength = inputBuffer.length;
  const ratio = fromSampleRate / toSampleRate;
  const outputLength = Math.max(1, Math.floor(inputLength / ratio));
  const output = new Float32Array(outputLength);
  const channels = Array.from({ length: channelCount }, (_, index) => inputBuffer.getChannelData(index));
  for (let outputIndex = 0; outputIndex < outputLength; outputIndex += 1) {
    const inputIndex = Math.min(inputLength - 1, Math.floor(outputIndex * ratio));
    let sum = 0;
    for (const channel of channels) {
      sum += channel[inputIndex] || 0;
    }
    output[outputIndex] = Math.max(-1, Math.min(1, sum / channels.length));
  }
  return output;
}

async function toggleScreenplayPlayback() {
  if (!playbackAudioEl) {
    return;
  }
  if (!playbackAudioEl.paused) {
    playbackAudioEl.pause();
    return;
  }
  playbackAudioEl.src = `/api/live-session-audio.wav?ts=${Date.now()}`;
  try {
    await playbackAudioEl.play();
  } catch (error) {
    inputUiState.browserError = conciseErrorMessage(error);
    renderInputChrome();
  }
}

function conciseErrorMessage(error) {
  if (!error) {
    return "Unknown error";
  }
  if (typeof error === "string") {
    return error;
  }
  if (typeof error.message === "string" && error.message.trim().length > 0) {
    return error.message.trim();
  }
  return String(error);
}
