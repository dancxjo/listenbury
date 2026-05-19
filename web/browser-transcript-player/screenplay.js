import {
  assembleNarrativeManuscript,
  buildNarrativeEpisode,
  createNarrativeSession,
  reduceNarrativeEvent,
} from "/assets/screenplay-model.mjs";
import { toTitleCase } from "/assets/scene-heading.mjs";

const scriptRoot = document.getElementById("script");
const statusEl = document.getElementById("connection-status");
const eventCountEl = document.getElementById("event-count");
const dotEl = document.getElementById("connection-dot");

const RENDER_DEBOUNCE_MS = 60;
const MAX_DELETED_SNIPPETS = 8;
const PLACEHOLDER_SCENE_HEADING = "INT./EXT. UNKNOWN LOCATION - PRESENT";
const PLACEHOLDER_ACTION =
  "Scene headings and action are provisional until enough live context arrives to form scenes and an episode.";

const session = createNarrativeSession();
const pageState = {
  connectionStatus: "connecting",
  connectionClass: "is-connecting",
  message: "Waiting for live transcription...",
};

let renderScheduled = false;

connectLiveEvents();
render();

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

  const sources = document.createElement("p");
  sources.className = "source-line";
  sources.textContent = `Source trace IDs: ${scene.sourceEventIds.join(", ")}`;
  section.append(sources);

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
