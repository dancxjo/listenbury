const scriptRoot = document.getElementById("script");
const statusEl = document.getElementById("connection-status");
const eventCountEl = document.getElementById("event-count");
const dotEl = document.getElementById("connection-dot");

const RENDER_DEBOUNCE_MS = 60;
const MAX_DELETED_SNIPPETS = 8;
const PLACEHOLDER_SCENE_HEADING = "INT./EXT. LISTENBURY RUNTIME - PRESENT";
const PLACEHOLDER_ACTION =
  "Scene heading and action are provisional. Future builds can replace this with LLM-labeled scenes and vision-derived interior/exterior context.";

const session = {
  turns: new Map(),
  eventCount: 0,
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
    session.connectionStatus = "connected";
    session.connectionClass = "is-connected";
    session.message = "Listening for live transcription...";
    scheduleRender();
  };

  source.onmessage = (event) => {
    try {
      reduceEvent(JSON.parse(event.data));
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
    session.connectionStatus = "unavailable";
    session.connectionClass = "is-error";
    session.message = message;
    scheduleRender();
    source.close();
  });

  source.onerror = () => {
    session.connectionStatus = "disconnected";
    session.connectionClass = "is-error";
    session.message = "Live event stream disconnected.";
    scheduleRender();
    source.close();
  };
}

function reduceEvent(event) {
  if (!event || typeof event !== "object") {
    return;
  }

  session.eventCount += 1;
  const turn = getTurn(event.turn ?? 0);

  if (event.kind === "transcript_candidate") {
    applyTranscriptCandidate(turn, event);
    return;
  }

  if (event.kind === "asr_timed_word_stream") {
    applyAsrWordStream(turn, event);
    return;
  }

  if (event.kind === "transcript" && textContent(event.text)) {
    const next = textContent(event.text);
    recordDeletedText(turn.userDeleted, deletedTextBetween(currentUserText(turn), next), event.elapsed_ms);
    turn.userFinal = next;
    turn.userStable = next;
    turn.userUnstable = "";
    turn.userCandidateText = next;
    return;
  }

  if (event.kind === "tts_timed_word_stream_revision") {
    applyTtsWordStreamRevision(turn, event);
    return;
  }

  if (isLlmTextEvent(event.kind) && textContent(event.text)) {
    applyLlmTextEvent(turn, event);
    return;
  }

  if (event.kind === "playback_started" && textContent(event.text)) {
    finalizeLlmText(turn, textContent(event.text));
    return;
  }

  if (event.kind === "playback_finished" && turn.llmProspective) {
    finalizeLlmText(turn, turn.llmProspective);
  }
}

function getTurn(turnId) {
  if (!session.turns.has(turnId)) {
    session.turns.set(turnId, {
      id: turnId,
      userFinal: "",
      userStable: "",
      userUnstable: "",
      userWords: [],
      userCandidateText: "",
      userDeleted: [],
      llmFinal: "",
      llmProspective: "",
      llmFragments: [],
      llmWords: [],
      llmDeleted: [],
    });
  }
  return session.turns.get(turnId);
}

function applyTranscriptCandidate(turn, event) {
  const artifact = event.artifact && typeof event.artifact === "object" ? event.artifact : null;
  if (!artifact) {
    if (String(event.text ?? "").includes("candidate_cancelled")) {
      recordDeletedText(turn.userDeleted, [turn.userCandidateText], event.elapsed_ms);
      turn.userStable = "";
      turn.userUnstable = "";
      turn.userCandidateText = "";
    }
    return;
  }

  const stable = textContent(artifact.stable_text);
  const unstable = textContent(artifact.unstable_text);
  const next = joinSemanticText(stable, unstable);
  recordDeletedText(turn.userDeleted, deletedTextBetween(turn.userCandidateText, next), event.elapsed_ms);
  turn.userStable = stable;
  turn.userUnstable = unstable;
  turn.userCandidateText = next;
}

function applyAsrWordStream(turn, event) {
  const words = Array.isArray(event.artifact?.words) ? event.artifact.words : [];
  if (!words.length) {
    return;
  }

  const next = joinWords(words.map((word) => word?.text));
  recordDeletedText(turn.userDeleted, deletedTextBetween(joinWords(turn.userWords.map((word) => word.text)), next), event.elapsed_ms);
  turn.userWords = words
    .filter((word) => textContent(word?.text))
    .map((word) => ({
      text: textContent(word.text),
      commitment: String(word.commitment ?? ""),
    }));
  turn.userCandidateText = turn.userFinal || next;
}

function applyTtsWordStreamRevision(turn, event) {
  const words = Array.isArray(event.artifact?.words) ? event.artifact.words : [];
  const text = joinWords(words.map((word) => word?.text));
  if (!text) {
    return;
  }

  const reason = String(event.reason ?? "").toLowerCase();
  const cancelled = reason.includes("cancel") || words.every((word) => String(word?.commitment ?? "") === "Cancelled");
  if (cancelled) {
    recordDeletedText(turn.llmDeleted, [text], event.elapsed_ms);
    removeLlmFragment(turn, text);
    turn.llmWords = [];
    return;
  }

  turn.llmWords = words
    .filter((word) => textContent(word?.text))
    .map((word) => ({
      text: textContent(word.text),
      commitment: String(word.commitment ?? ""),
    }));
  const playedPrefix = [];
  for (const word of turn.llmWords) {
    if (!isPlayedCommitment(word.commitment)) {
      break;
    }
    playedPrefix.push(word.text);
  }
  let committedRevision = false;
  if (playedPrefix.length) {
    commitLlmText(turn, joinWords(playedPrefix));
    committedRevision = true;
  } else if (reason.includes("committed")) {
    commitLlmText(turn, text);
    committedRevision = true;
  }
  if (!committedRevision) {
    addLlmFragment(turn, text);
  }
}

function applyLlmTextEvent(turn, event) {
  const text = textContent(event.text);
  if (!text) {
    return;
  }

  if (event.kind === "speech_unit_cancelled") {
    recordDeletedText(turn.llmDeleted, [text], event.elapsed_ms);
    removeLlmFragment(turn, text);
    return;
  }

  if (event.kind === "speculative_speech_updated") {
    setLlmProspective(turn, text);
    return;
  }

  if (event.kind === "tts_enqueue_started" || event.kind === "speech_unit_committed") {
    commitLlmText(turn, text);
    return;
  }

  addLlmFragment(turn, text);
}

function isLlmTextEvent(kind) {
  return [
    "first_safe_speech_unit_emitted",
    "speech_unit_committed",
    "speech_unit_cancelled",
    "speculative_speech_updated",
    "tts_enqueue_started",
  ].includes(kind);
}

function addLlmFragment(turn, text) {
  const cleaned = textContent(text);
  if (!cleaned) {
    return;
  }
  const last = turn.llmFragments[turn.llmFragments.length - 1];
  if (last !== cleaned && !turn.llmFragments.includes(cleaned)) {
    turn.llmFragments.push(cleaned);
  }
  turn.llmProspective = joinSemanticText(...turn.llmFragments);
}

function setLlmProspective(turn, text) {
  const cleaned = textContent(text);
  if (!cleaned) {
    return;
  }
  turn.llmProspective = cleaned;
  if (!turn.llmFragments.includes(cleaned)) {
    turn.llmFragments = [cleaned];
  }
}

function removeLlmFragment(turn, text) {
  const cleaned = textContent(text);
  turn.llmFragments = turn.llmFragments.filter((fragment) => fragment !== cleaned);
  turn.llmProspective = joinSemanticText(...turn.llmFragments);
}

function finalizeLlmText(turn, text) {
  commitLlmText(turn, text);
  turn.llmWords = [];
}

function commitLlmText(turn, text) {
  const next = textContent(text);
  if (!next) {
    return;
  }

  const current = textContent(turn.llmFinal);
  if (!current) {
    turn.llmFinal = next;
  } else if (current === next || current.endsWith(next)) {
    turn.llmFinal = current;
  } else if (next.startsWith(current)) {
    turn.llmFinal = next;
  } else {
    turn.llmFinal = joinSemanticText(current, next);
  }

  if (!turn.llmProspective || turn.llmFinal.startsWith(turn.llmProspective)) {
    turn.llmProspective = turn.llmFinal;
  } else if (!turn.llmProspective.startsWith(turn.llmFinal)) {
    turn.llmProspective = joinSemanticText(turn.llmFinal, turn.llmProspective);
  }
  turn.llmFragments = turn.llmProspective ? [turn.llmProspective] : [];
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
  statusEl.textContent = session.connectionStatus;
  eventCountEl.textContent = `${session.eventCount} event${session.eventCount === 1 ? "" : "s"}`;
  dotEl.className = `connection-dot ${session.connectionClass}`;

  const turns = [...session.turns.values()]
    .filter((turn) => turnHasDialogue(turn))
    .sort((left, right) => left.id - right.id);

  scriptRoot.replaceChildren();
  scriptRoot.append(sceneHeading(PLACEHOLDER_SCENE_HEADING), actionLine(PLACEHOLDER_ACTION));
  if (!turns.length) {
    const empty = document.createElement("p");
    empty.className = "empty-page";
    empty.textContent = session.message;
    scriptRoot.append(empty);
    return;
  }

  for (const turn of turns) {
    const block = document.createElement("section");
    block.className = "turn";

    if (turnHasUserDialogue(turn)) {
      block.append(characterCue("USER"), dialogueLine(userSegments(turn)));
    }
    if (turnHasLlmDialogue(turn)) {
      block.append(characterCue("PETE"), dialogueLine(llmSegments(turn)));
    }

    scriptRoot.append(block);
  }

  window.requestAnimationFrame(() => {
    window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
  });
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
  for (const segment of compactSegments(segments)) {
    const span = document.createElement("span");
    if (segment.className) {
      span.className = segment.className;
    }
    span.textContent = segment.text;
    line.append(span);
  }
  return line;
}

function userSegments(turn) {
  const segments = [];

  if (turn.userFinal) {
    segments.push({ text: turn.userFinal });
  } else if (turn.userWords.length) {
    for (const word of turn.userWords) {
      segments.push({
        text: word.text,
        className: isProspectiveCommitment(word.commitment) ? "prospective-asr" : "",
        word: true,
      });
    }
  } else {
    if (turn.userStable) {
      segments.push({ text: turn.userStable });
    }
    if (turn.userUnstable) {
      segments.push({ text: turn.userUnstable, className: "prospective-asr" });
    }
  }

  appendDeletedSegments(segments, turn.userDeleted);
  return segments;
}

function llmSegments(turn) {
  const segments = [];
  const finalText = turn.llmFinal;
  const prospectiveText = turn.llmProspective;

  if (finalText) {
    segments.push({ text: finalText });
    const wordStreamText = turn.llmWords.length ? joinWords(turn.llmWords.map((word) => word.text)) : "";
    const tail = prospectiveTail(finalText, prospectiveText) || prospectiveTail(finalText, wordStreamText);
    if (tail) {
      segments.push({ text: tail, className: "prospective-llm" });
    }
  } else if (turn.llmWords.length) {
    for (const word of turn.llmWords) {
      segments.push({
        text: word.text,
        className: isPlayedCommitment(word.commitment) ? "" : "prospective-llm",
        word: true,
      });
    }
  } else if (prospectiveText) {
    segments.push({ text: prospectiveText, className: "prospective-llm" });
  }

  appendDeletedSegments(segments, turn.llmDeleted);
  return segments;
}

function appendDeletedSegments(segments, deleted) {
  for (const item of deleted.slice(-MAX_DELETED_SNIPPETS)) {
    if (item.text) {
      segments.push({ text: item.text, className: "deleted" });
    }
  }
}

function compactSegments(segments) {
  const result = [];
  for (const segment of segments.filter((entry) => textContent(entry.text))) {
    const text = textContent(segment.text);
    const previous = result[result.length - 1];
    const needsSpace = previous && !/^\s|^[,.;:!?)]/.test(text) && !/\s$/.test(previous.text);
    const nextText = `${needsSpace ? " " : ""}${text}`;
    if (previous && previous.className === segment.className && segment.word) {
      previous.text += nextText;
    } else {
      result.push({ text: nextText, className: segment.className || "" });
    }
  }
  return result;
}

function turnHasDialogue(turn) {
  return turnHasUserDialogue(turn) || turnHasLlmDialogue(turn);
}

function turnHasUserDialogue(turn) {
  return Boolean(
    turn.userFinal ||
      turn.userStable ||
      turn.userUnstable ||
      turn.userWords.length ||
      turn.userDeleted.length,
  );
}

function turnHasLlmDialogue(turn) {
  return Boolean(turn.llmFinal || turn.llmProspective || turn.llmDeleted.length);
}

function currentUserText(turn) {
  return turn.userFinal || joinWords(turn.userWords.map((word) => word.text)) || turn.userCandidateText;
}

function isProspectiveCommitment(commitment) {
  return !["Final", "StableText", "Played"].includes(String(commitment ?? ""));
}

function isPlayedCommitment(commitment) {
  return ["Final", "Played"].includes(String(commitment ?? ""));
}

function prospectiveTail(finalText, prospectiveText) {
  const finalClean = textContent(finalText);
  const prospectiveClean = textContent(prospectiveText);
  if (!prospectiveClean || prospectiveClean === finalClean) {
    return "";
  }
  if (finalClean.endsWith(prospectiveClean)) {
    return "";
  }
  if (prospectiveClean.startsWith(finalClean)) {
    return prospectiveClean.slice(finalClean.length).trim();
  }
  return prospectiveClean;
}

function recordDeletedText(target, snippets, elapsedMs) {
  for (const snippet of snippets.map(textContent).filter(Boolean)) {
    if (target.some((entry) => entry.text === snippet)) {
      continue;
    }
    target.push({ text: snippet, elapsedMs: elapsedMs ?? 0 });
  }
  if (target.length > MAX_DELETED_SNIPPETS) {
    target.splice(0, target.length - MAX_DELETED_SNIPPETS);
  }
}

function deletedTextBetween(previous, next) {
  const prevTokens = tokenize(previous);
  const nextTokens = new Set(tokenize(next));
  if (!prevTokens.length) {
    return [];
  }

  const deleted = [];
  let current = [];
  for (const token of prevTokens) {
    if (nextTokens.has(token)) {
      if (current.length) {
        deleted.push(joinWords(current));
        current = [];
      }
    } else {
      current.push(token);
    }
  }
  if (current.length) {
    deleted.push(joinWords(current));
  }
  return deleted;
}

function tokenize(text) {
  return textContent(text).match(/\S+/g) ?? [];
}

function joinWords(words) {
  return words
    .map(textContent)
    .filter(Boolean)
    .reduce((acc, word) => acc + (/^[,.;:!?)]/.test(word) ? word : `${acc ? " " : ""}${word}`), "");
}

function joinSemanticText(...parts) {
  return textContent(parts.map(textContent).filter(Boolean).join(" "));
}

function textContent(value) {
  return String(value ?? "")
    .replace(/\s+/g, " ")
    .replace(/\s+([,.;:!?])/g, "$1")
    .trim();
}
