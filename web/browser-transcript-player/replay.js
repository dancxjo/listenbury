// WaveDeck Trace Replay Controller
//
// This module drives the replay page (replay.html).  It waits for app.js to
// expose window.wavedeckReplay (set when data-mode="replay" is on <body>),
// then wires up the replay controls.
//
// Replay semantics:
//   • JSONL traces are replayed event-by-event with deterministic timing based
//     on each event's elapsed_ms field.
//   • JSON viewer-payload files are loaded as a static snapshot (no stepping).
//   • Pause/resume, speed control, step-forward, seek, and reset are all
//     supported for JSONL replays.

// ── Wait for app.js to register the replay API ───────────────────────────

function waitForReplayApi(resolve) {
  if (window.wavedeckReplay) {
    resolve(window.wavedeckReplay);
  } else {
    // app.js is a module; it runs after this module.  Poll until available.
    setTimeout(() => waitForReplayApi(resolve), 20);
  }
}

const replayApi = await new Promise(waitForReplayApi);
const { addLiveEvent, applyPayload, renderShell, resetLiveSession, uiState } = replayApi;

// ── Replay state ──────────────────────────────────────────────────────────

let replayEvents = [];        // parsed JSONL events in order
let replayIndex = 0;          // next event to dispatch
let replayPaused = true;
let replaySpeed = 1.0;        // current speed multiplier
let replayTimer = null;
let replayStartPerf = null;   // performance.now() when play started/resumed
let replayStartEventMs = null; // elapsed_ms of the first pending event when play started

// ── DOM references ────────────────────────────────────────────────────────

const fixtureSelect = document.getElementById("fixture-select");
const fileInput = document.getElementById("replay-file-input");
const playPauseBtn = document.getElementById("replay-play-pause");
const stepBtn = document.getElementById("replay-step");
const resetBtn = document.getElementById("replay-reset");
const speedRange = document.getElementById("replay-speed");
const speedLabel = document.getElementById("replay-speed-label");
const progressBar = document.getElementById("replay-progress-bar");
const statusEl = document.getElementById("replay-status");

// ── Event bindings ────────────────────────────────────────────────────────

fixtureSelect.addEventListener("change", () => {
  const option = fixtureSelect.options[fixtureSelect.selectedIndex];
  const url = option.value;
  const type = option.dataset.type;
  if (url) {
    void loadFromUrl(url, type || inferTypeFromUrl(url));
    fixtureSelect.value = "";
  }
});

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (!file) return;
  const type = inferTypeFromFilename(file.name);
  void loadFromFile(file, type);
  fileInput.value = "";
});

playPauseBtn.addEventListener("click", () => {
  if (replayPaused) {
    startReplay();
  } else {
    pauseReplay();
  }
});

stepBtn.addEventListener("click", stepForward);

resetBtn.addEventListener("click", resetReplay);

speedRange.addEventListener("input", () => {
  replaySpeed = parseFloat(speedRange.value);
  speedLabel.textContent = `${replaySpeed.toFixed(1)}×`;
  if (!replayPaused && replayIndex < replayEvents.length) {
    // Re-anchor timing so speed change takes effect immediately.
    replayStartPerf = performance.now();
    replayStartEventMs = replayEvents[replayIndex]?.elapsed_ms ?? 0;
  }
});

progressBar.addEventListener("input", () => {
  if (!replayEvents.length) return;
  const maxMs = replayEvents[replayEvents.length - 1]?.elapsed_ms ?? 0;
  const targetMs = (parseFloat(progressBar.value) / 100) * maxMs;
  seekToMs(targetMs);
});

// ── Loading ───────────────────────────────────────────────────────────────

function inferTypeFromUrl(url) {
  if (url.endsWith(".jsonl")) return "jsonl";
  return "payload";
}

function inferTypeFromFilename(name) {
  if (name.endsWith(".jsonl")) return "jsonl";
  return "payload";
}

async function loadFromUrl(url, type) {
  setStatus("Loading…");
  try {
    const response = await fetch(url);
    if (!response.ok) {
      setStatus(`Load failed: HTTP ${response.status}`);
      return;
    }
    const text = await response.text();
    processLoadedText(text, type, url.split("/").pop() ?? url);
  } catch (err) {
    setStatus(`Load error: ${err.message}`);
  }
}

async function loadFromFile(file, type) {
  setStatus("Reading file…");
  try {
    const text = await file.text();
    processLoadedText(text, type, file.name);
  } catch (err) {
    setStatus(`File read error: ${err.message}`);
  }
}

function processLoadedText(text, type, label) {
  if (type === "jsonl") {
    loadJsonlTrace(text, label);
  } else {
    loadStaticPayload(text, label);
  }
}

function loadJsonlTrace(text, label) {
  const lines = text.split("\n").filter((line) => line.trim());
  const events = [];
  let parseErrors = 0;
  for (const line of lines) {
    try {
      events.push(JSON.parse(line));
    } catch {
      parseErrors++;
    }
  }
  if (!events.length) {
    setStatus(`No valid events in ${label}`);
    return;
  }
  replayEvents = events;
  replayIndex = 0;
  replayPaused = true;
  replayStartPerf = null;
  replayStartEventMs = null;
  cancelScheduledReplay();
  resetLiveSession();
  setLiveMode(true, `Replay — ${label}`);
  enableControls(true);
  updateProgressBar();
  const durationS = ((events[events.length - 1]?.elapsed_ms ?? 0) / 1000).toFixed(1);
  const note = parseErrors > 0 ? ` (${parseErrors} parse error${parseErrors === 1 ? "" : "s"})` : "";
  setStatus(`${events.length} events · ${durationS}s${note} · paused`);
}

function loadStaticPayload(text, label) {
  let payload;
  try {
    payload = JSON.parse(text);
  } catch (err) {
    setStatus(`JSON parse error: ${err.message}`);
    return;
  }
  replayEvents = [];
  replayIndex = 0;
  replayPaused = true;
  cancelScheduledReplay();
  resetLiveSession();
  setLiveMode(false, `Snapshot — ${label}`);
  enableControls(false);
  applyPayload(payload);
  setStatus(`Static payload loaded: ${label}`);
}

// ── Playback control ──────────────────────────────────────────────────────

function startReplay() {
  if (!replayEvents.length || replayIndex >= replayEvents.length) return;
  replayPaused = false;
  replayStartPerf = performance.now();
  replayStartEventMs = replayEvents[replayIndex]?.elapsed_ms ?? 0;
  playPauseBtn.textContent = "⏸ Pause";
  scheduleNextEvent();
}

function pauseReplay() {
  replayPaused = true;
  cancelScheduledReplay();
  playPauseBtn.textContent = "▶ Play";
  setStatus(statusFromProgress());
}

function resetReplay() {
  pauseReplay();
  replayIndex = 0;
  resetLiveSession();
  updateProgressBar();
  playPauseBtn.textContent = "▶ Play";
  setStatus(`${replayEvents.length} events · reset`);
}

function stepForward() {
  if (!replayEvents.length || replayIndex >= replayEvents.length) return;
  if (!replayPaused) {
    pauseReplay();
  }
  addLiveEvent(replayEvents[replayIndex]);
  replayIndex++;
  updateProgressBar();
  if (replayIndex >= replayEvents.length) {
    setStatus("Replay complete");
    disablePlayPause();
  } else {
    setStatus(statusFromProgress());
  }
}

function seekToMs(targetMs) {
  const wasPlaying = !replayPaused;
  pauseReplay();
  cancelScheduledReplay();
  replayIndex = 0;
  resetLiveSession();

  // Dispatch all events up to targetMs instantaneously.
  while (replayIndex < replayEvents.length && replayEvents[replayIndex].elapsed_ms <= targetMs) {
    addLiveEvent(replayEvents[replayIndex]);
    replayIndex++;
  }
  updateProgressBar();
  setStatus(statusFromProgress());

  if (wasPlaying && replayIndex < replayEvents.length) {
    startReplay();
  }
}

// ── Scheduling ────────────────────────────────────────────────────────────

function scheduleNextEvent() {
  if (replayPaused || replayIndex >= replayEvents.length) {
    if (replayIndex >= replayEvents.length) {
      replayPaused = true;
      playPauseBtn.textContent = "▶ Play";
      setStatus("Replay complete");
      disablePlayPause();
    }
    return;
  }

  const nextEvent = replayEvents[replayIndex];
  const wallMsFromStart = (nextEvent.elapsed_ms - replayStartEventMs) / replaySpeed;
  const targetPerf = replayStartPerf + wallMsFromStart;
  const delay = Math.max(0, targetPerf - performance.now());

  replayTimer = setTimeout(() => {
    if (replayPaused || replayIndex >= replayEvents.length) return;
    addLiveEvent(replayEvents[replayIndex]);
    replayIndex++;
    updateProgressBar();
    scheduleNextEvent();
  }, delay);
}

function cancelScheduledReplay() {
  if (replayTimer !== null) {
    clearTimeout(replayTimer);
    replayTimer = null;
  }
}

// ── UI helpers ────────────────────────────────────────────────────────────

function setStatus(message) {
  if (statusEl) statusEl.textContent = message;
}

function statusFromProgress() {
  if (!replayEvents.length) return "No trace loaded";
  const atMs = replayEvents[replayIndex - 1]?.elapsed_ms ?? 0;
  const maxMs = replayEvents[replayEvents.length - 1]?.elapsed_ms ?? 0;
  const atS = (atMs / 1000).toFixed(1);
  const maxS = (maxMs / 1000).toFixed(1);
  return `Event ${replayIndex} / ${replayEvents.length} · ${atS}s / ${maxS}s`;
}

function updateProgressBar() {
  if (!progressBar || !replayEvents.length) return;
  const maxMs = replayEvents[replayEvents.length - 1]?.elapsed_ms ?? 1;
  const atMs = replayEvents[Math.max(0, replayIndex - 1)]?.elapsed_ms ?? 0;
  progressBar.value = maxMs > 0 ? String((atMs / maxMs) * 100) : "0";
}

function enableControls(hasJSONL) {
  playPauseBtn.disabled = !hasJSONL;
  stepBtn.disabled = !hasJSONL;
  resetBtn.disabled = false;
  progressBar.disabled = !hasJSONL;
  playPauseBtn.textContent = "▶ Play";
}

function disablePlayPause() {
  playPauseBtn.disabled = true;
  stepBtn.disabled = true;
}

function setLiveMode(live, title) {
  uiState.liveMode = live;
  uiState.statusMessage = title;
  uiState.connectionStatusClass = live ? "live-status-connected" : "";
  uiState.connectionStatusText = live ? "replay" : "";
  if (live) {
    document.title = `WaveDeck · ${title}`;
    document.body.classList.add("live-mode");
  } else {
    document.title = `WaveDeck · ${title}`;
    document.body.classList.remove("live-mode");
  }
  renderShell();
}
