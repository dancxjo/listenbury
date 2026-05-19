const MAX_DELETED_SNIPPETS = 8;

const TOPIC_CATALOG = [
  {
    key: "memory-manuscript",
    heading: "INT. LISTENBURY RUNTIME - MEMORY, DECOUPAGE, AND MANUSCRIPT",
    chapterTitle: "Memory, Découpage, and Manuscript",
    action:
      "Pete and the user shape live traces into something that can be read later as narrative memory.",
    keywords: [/\bmemory\b/i, /\bd[eé]coupage\b/i, /\bscene\b/i, /\bepisode\b/i, /\bchapter\b/i, /\bmanuscript\b/i],
  },
  {
    key: "phonology-workbench",
    heading: "INT. LISTENBURY RUNTIME - PHONOLOGY WORKBENCH",
    chapterTitle: "Phonology and the Native Piper Spine",
    action:
      "The conversation narrows to speech mechanics, phonology, and the shape of Pete's own mouth and ear.",
    keywords: [/\bphonolog/i, /\bpiper\b/i, /\bmouth\b/i, /\bhear\b/i, /\bvowel\b/i, /\bconsonant\b/i],
  },
  {
    key: "cuda-bring-up",
    heading: "INT. LISTENBURY RUNTIME - CUDA BRING-UP",
    chapterTitle: "Building the Mouth",
    action:
      "A technical bring-up scene gathers around the GPU path and the machinery needed to make Pete speak.",
    keywords: [/\bcuda\b/i, /\bgpu\b/i, /\bkernel\b/i, /\bdriver\b/i, /\bbring[- ]?up\b/i],
  },
  {
    key: "wavedeck-inspection",
    heading: "INT. LISTENBURY RUNTIME - WAVEDECK INSPECTION",
    chapterTitle: "The WaveDeck Era",
    action:
      "Pete and the user watch the session machinery in motion, following traces, overlap, and routing decisions.",
    keywords: [/\bwave ?deck\b/i, /\boverlap\b/i, /\brouting\b/i, /\btrace\b/i, /\bspan\b/i, /\btimeline\b/i, /\breplay\b/i],
  },
  {
    key: "quiet-grief",
    heading: "INT. LISTENBURY RUNTIME - QUIET GRIEF",
    chapterTitle: "Teaching Pete to Hear Himself",
    action:
      "The room softens. The talk turns inward and careful, with room for loss, apology, or tenderness.",
    keywords: [/\bgrief\b/i, /\bgriev/i, /\bsad\b/i, /\bloss\b/i, /\bmiss\b/i, /\bsorry\b/i, /\bhurt\b/i],
  },
  {
    key: "interruption-handoff",
    heading: "INT. LISTENBURY RUNTIME - INTERRUPTION HANDOFF",
    chapterTitle: "The WaveDeck Era",
    action:
      "An interruption cuts across the session and the narrative briefly reorients around overlap, yielding, and cancellation.",
    keywords: [/\binterrupt/i, /\boverlap\b/i, /\byield/i, /\bcancel/i],
  },
];

const FALLBACK_TOPIC = {
  key: "live-session",
  heading: "INT. LISTENBURY RUNTIME - PRESENT",
  chapterTitle: "The WaveDeck Era",
  action: "Pete and the user remain in the live runtime, gathering the next beat of the session as it arrives.",
};

export function createNarrativeSession() {
  return {
    turns: new Map(),
    proposition: null,
    propositionDeleted: [],
    sourceEvents: [],
    eventCount: 0,
    nextSourceSequence: 1,
  };
}

export function reduceNarrativeEvent(session, event) {
  if (!event || typeof event !== "object") {
    return;
  }

  session.eventCount += 1;
  const sourceEvent = normalizeSourceEvent(event, session.nextSourceSequence++);
  session.sourceEvents.push(sourceEvent);

  const turn = getTurn(session, eventTurnKey(event), eventTurnNumber(event));
  turn.sourceEventIds.push(sourceEvent.sourceId);
  turn.eventKinds.add(sourceEvent.kind);
  turn.startedAtMs = minNumber(turn.startedAtMs, numericMs(sourceEvent.elapsed_ms));
  turn.endedAtMs = maxNumber(turn.endedAtMs, numericMs(sourceEvent.elapsed_ms));
  registerEventSignals(turn, sourceEvent);

  if (sourceEvent.kind === "transcript_proposition" && textContent(sourceEvent.text)) {
    const next = textContent(sourceEvent.text);
    recordDeletedText(session.propositionDeleted, deletedTextBetween(session.proposition?.text ?? "", next), sourceEvent.elapsed_ms);
    session.proposition = {
      text: next,
      source: textContent(sourceEvent.artifact?.source) || "refinement",
      elapsedMs: sourceEvent.elapsed_ms ?? 0,
      sourceId: sourceEvent.sourceId,
    };
    return;
  }

  if (sourceEvent.kind === "transcript_candidate") {
    applyTranscriptCandidate(turn, sourceEvent);
    return;
  }

  if (sourceEvent.kind === "asr_timed_word_stream") {
    applyAsrWordStream(turn, sourceEvent);
    return;
  }

  if (sourceEvent.kind === "transcript" && textContent(sourceEvent.text)) {
    const next = textContent(sourceEvent.text);
    const deleted = deletedTextBetween(currentUserText(turn), next);
    if (deleted.length) {
      turn.flags.revised = true;
    }
    recordDeletedText(turn.userDeleted, deleted, sourceEvent.elapsed_ms);
    turn.userFinal = next;
    turn.userStable = next;
    turn.userUnstable = "";
    turn.userCandidateText = next;
    return;
  }

  if (sourceEvent.kind === "tts_timed_word_stream_revision") {
    applyTtsWordStreamRevision(turn, sourceEvent);
    return;
  }

  if (
    isLlmTextEvent(sourceEvent.kind) &&
    (textContent(sourceEvent.text) || speechUnitIdFromEvent(sourceEvent))
  ) {
    applyLlmTextEvent(turn, sourceEvent);
    return;
  }

  if (sourceEvent.kind === "playback_started" && textContent(sourceEvent.text)) {
    finalizeLlmText(turn, textContent(sourceEvent.text));
    return;
  }

  if (sourceEvent.kind === "playback_finished" && turn.llmProspective) {
    finalizeLlmText(turn, turn.llmProspective);
  }
}

export function buildNarrativeEpisode(session, options = {}) {
  const turns = [...session.turns.values()]
    .filter((turn) => turnHasNarrativeMaterial(turn))
    .sort((left, right) => left.id - right.id);

  const scenes = buildScenes(turns);
  const startedAtMs = turns.length ? turns[0].startedAtMs ?? 0 : 0;
  const endedAtMs = turns.length ? turns[turns.length - 1].endedAtMs ?? startedAtMs : startedAtMs;
  const leadScene = scenes[0] ?? null;
  const episodeNumber = options.episodeNumber ?? 1;
  const title =
    options.title ??
    (leadScene ? `Episode ${episodeNumber}: ${shortHeading(leadScene.heading)}` : `Episode ${episodeNumber}: Live Session`);
  const summary =
    scenes.length > 0
      ? summarizeEpisode(scenes)
      : "No committed narrative beats have arrived yet. The screenplay remains live and provisional.";
  const sourceEventIds = unique(scenes.flatMap((scene) => scene.sourceEventIds));

  const metadata = {
    episodeNumber,
    startedAtMs,
    endedAtMs,
    sessionTurns: turns.length,
    eventCount: session.eventCount,
    live: options.live ?? true,
    sessionLabel: options.sessionLabel ?? "Live session",
    displayPolicy: {
      committed: "Rendered as plain screenplay dialogue by default.",
      prospective: "Rendered inline with prospective styling while still live.",
      deleted: "Rendered inline with strike-through styling for traceability.",
      cancelled: "Rendered as action lines and deleted fragments when available.",
    },
  };

  const episode = {
    id: options.id ?? `episode-${episodeNumber}`,
    type: "episode",
    title,
    summary,
    metadata,
    sourceEventIds,
    waveDeckHref: options.waveDeckHref ?? "/",
    replayHref: options.replayHref ?? null,
    scenes,
    sceneList: scenes.map((scene) => ({ id: scene.id, heading: scene.heading, summary: scene.summary })),
    children: scenes,
  };
  episode.screenplayBody = renderEpisodeScreenplay(episode);
  return episode;
}

export function assembleNarrativeManuscript(episodes, options = {}) {
  const normalizedEpisodes = episodes.filter(Boolean);
  const chapterMap = new Map();

  for (const episode of normalizedEpisodes) {
    const chapterTitle = episode.chapterTitle ?? dominantChapterTitle(episode.scenes);
    if (!chapterMap.has(chapterTitle)) {
      chapterMap.set(chapterTitle, {
        id: `chapter-${slugify(chapterTitle)}`,
        type: "chapter",
        title: chapterTitle,
        summary: "",
        episodes: [],
        children: [],
      });
    }
    const chapter = chapterMap.get(chapterTitle);
    chapter.episodes.push(episode);
    chapter.children = chapter.episodes;
  }

  const chapters = [...chapterMap.values()].map((chapter) => ({
    ...chapter,
    summary:
      chapter.episodes.length === 1
        ? chapter.episodes[0].summary
        : `${chapter.episodes.length} episodes continue the same narrative arc.`,
  }));

  return {
    id: options.id ?? "manuscript-live",
    type: "manuscript",
    title: options.title ?? "The Life of Pete Listenbury",
    summary:
      normalizedEpisodes.length === 0
        ? "The manuscript is waiting for its first episode."
        : `${normalizedEpisodes.length} episode${normalizedEpisodes.length === 1 ? "" : "s"} arranged into ${chapters.length} chapter${chapters.length === 1 ? "" : "s"}.`,
    chapters,
    episodes: normalizedEpisodes,
    children: chapters,
  };
}

export function renderEpisodeScreenplay(episode) {
  const lines = [episode.title, "", `Summary: ${episode.summary}`, ""];
  lines.push(
    `Session: ${episode.metadata.sessionLabel}`,
    `Window: ${formatMs(episode.metadata.startedAtMs)} → ${formatMs(episode.metadata.endedAtMs)}`,
    `Events: ${episode.metadata.eventCount}`,
    "",
  );

  for (const scene of episode.scenes) {
    lines.push(scene.heading, scene.action, `Source trace IDs: ${scene.sourceEventIds.join(", ") || "none"}`, "");
    for (const beat of scene.beats) {
      if (beat.role) {
        lines.push(beat.role, stringifySegments(beat.segments), "");
      } else {
        lines.push(beat.text, "");
      }
    }
  }

  return lines.join("\n").trim();
}

export function turnHasNarrativeMaterial(turn) {
  return turnHasUserDialogue(turn) || turnHasLlmDialogue(turn) || turn.flags.interruption || turn.flags.cancelled;
}

function getTurn(session, turnKey, turnNumber) {
  if (!session.turns.has(turnKey)) {
    session.turns.set(turnKey, {
      key: turnKey,
      id: turnNumber,
      sourceEventIds: [],
      eventKinds: new Set(),
      startedAtMs: null,
      endedAtMs: null,
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
      flags: {
        interruption: false,
        cancelled: false,
        revised: false,
        prospective: false,
      },
      speechUnitsById: new Map(),
    });
  }
  return session.turns.get(turnKey);
}

function normalizeSourceEvent(event, sequence) {
  const turn = eventTurnNumber(event);
  const turnKey = eventTurnKey(event);
  const elapsedMs = numericMs(event.elapsed_ms);
  return {
    ...event,
    sourceId:
      event.source_id ??
      event.id ??
      `turn-${turnKey}:${String(event.kind ?? "event")}:${Number.isFinite(elapsedMs) ? elapsedMs : "na"}:${sequence}`,
  };
}

function registerEventSignals(turn, event) {
  const text = joinSemanticText(event.text, transcriptCandidateText(event.artifact), wordStreamText(event.artifact?.words));
  if (text) {
    turn.flags.prospective ||= event.kind === "transcript_candidate" || event.kind === "speculative_speech_updated";
  }

  if (["interruption_detected", "overlap_started", "yield_started", "yield_ended"].includes(event.kind)) {
    turn.flags.interruption = true;
  }

  if (["speech_unit_cancelled", "tts_timed_word_stream_revision"].includes(event.kind)) {
    const cancelledWords = Array.isArray(event.artifact?.words)
      ? event.artifact.words.every((word) => String(word?.commitment ?? "") === "Cancelled")
      : false;
    if (event.kind === "speech_unit_cancelled" || cancelledWords || String(event.reason ?? "").toLowerCase().includes("cancel")) {
      turn.flags.cancelled = true;
    }
  }
}

function applyTranscriptCandidate(turn, event) {
  const artifact = event.artifact && typeof event.artifact === "object" ? event.artifact : null;
  if (!artifact) {
    if (String(event.text ?? "").includes("candidate_cancelled")) {
      turn.flags.cancelled = true;
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
  const deleted = deletedTextBetween(turn.userCandidateText, next);
  if (deleted.length) {
    turn.flags.revised = true;
  }
  recordDeletedText(turn.userDeleted, deleted, event.elapsed_ms);
  turn.userStable = stable;
  turn.userUnstable = unstable;
  turn.userCandidateText = next;
  turn.flags.prospective ||= Boolean(unstable);
}

function applyAsrWordStream(turn, event) {
  const words = Array.isArray(event.artifact?.words) ? event.artifact.words : [];
  if (!words.length) {
    return;
  }

  const next = joinWords(words.map((word) => word?.text));
  const deleted = deletedTextBetween(joinWords(turn.userWords.map((word) => word.text)), next);
  if (deleted.length) {
    turn.flags.revised = true;
  }
  recordDeletedText(turn.userDeleted, deleted, event.elapsed_ms);
  turn.userWords = words
    .filter((word) => textContent(word?.text))
    .map((word) => ({
      text: textContent(word.text),
      commitment: String(word.commitment ?? ""),
      id: word.id ?? null,
    }));
  turn.userCandidateText = turn.userFinal || next;
  turn.flags.prospective ||= turn.userWords.some((word) => isProspectiveCommitment(word.commitment));
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
    turn.flags.cancelled = true;
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
      id: word.id ?? null,
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
  const speechUnitId = speechUnitIdFromEvent(event);
  const text = speechUnitText(turn, event);
  if (!text) {
    return;
  }
  if (speechUnitId) {
    turn.speechUnitsById.set(speechUnitId, text);
  }

  if (event.kind === "speech_unit_cancelled") {
    turn.flags.cancelled = true;
    recordDeletedText(turn.llmDeleted, [text], event.elapsed_ms);
    removeLlmFragment(turn, text);
    if (speechUnitId) {
      turn.speechUnitsById.delete(speechUnitId);
    }
    return;
  }

  if (event.kind === "speculative_speech_updated") {
    turn.flags.prospective = true;
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

function speechUnitIdFromEvent(event) {
  return normalizedId(event?.speech_unit_id ?? event?.artifact?.speech_unit_id);
}

function speechUnitText(turn, event) {
  const direct = textContent(event?.text);
  if (direct) {
    return direct;
  }
  const speechUnitId = speechUnitIdFromEvent(event);
  if (!speechUnitId) {
    return "";
  }
  return textContent(turn.speechUnitsById.get(speechUnitId));
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

function buildScenes(turns) {
  const grouped = [];
  for (const turn of turns) {
    const sceneKey = classifyTopic(turn);
    const current = grouped[grouped.length - 1];
    if (!current || shouldStartNewScene(current, turn, sceneKey)) {
      grouped.push({ key: sceneKey, turns: [turn] });
    } else {
      current.turns.push(turn);
      if (current.key === FALLBACK_TOPIC.key && sceneKey !== FALLBACK_TOPIC.key) {
        current.key = sceneKey;
      }
    }
  }

  const merged = mergeWeakScenes(grouped);
  return merged.map((group, index) => buildScene(group.turns, index, group.key));
}

function shouldStartNewScene(currentGroup, turn, nextSceneKey) {
  if (nextSceneKey === currentGroup.key) {
    return false;
  }
  const currentTurns = currentGroup.turns;
  const currentStrong = currentTurns.some((entry) => classifyTopic(entry) !== FALLBACK_TOPIC.key || entry.flags.interruption || entry.flags.cancelled);
  const nextStrong = nextSceneKey !== FALLBACK_TOPIC.key || turn.flags.interruption || turn.flags.cancelled;
  return currentStrong || nextStrong || currentTurns.length >= 2;
}

function mergeWeakScenes(groups) {
  const merged = [];
  for (const group of groups) {
    const weak = group.turns.length <= 1 && group.key === FALLBACK_TOPIC.key;
    const previous = merged[merged.length - 1];
    if (weak && previous) {
      previous.turns.push(...group.turns);
      continue;
    }
    if (previous && previous.key === group.key) {
      previous.turns.push(...group.turns);
      continue;
    }
    merged.push({ key: group.key, turns: [...group.turns] });
  }
  return merged;
}

function buildScene(turns, index, sceneKey) {
  const topic = topicForKey(sceneKey);
  const beats = turns.flatMap((turn) => turnToBeats(turn));
  const sourceEventIds = unique(turns.flatMap((turn) => turn.sourceEventIds));
  const summary = summarizeScene(turns, topic, beats);
  const scene = {
    id: `scene-${index + 1}`,
    type: "scene",
    topicKey: sceneKey,
    heading: topic.heading,
    action: describeSceneAction(turns, topic),
    summary,
    turnIds: turns.map((turn) => turn.id),
    sourceEventIds,
    beats,
    children: beats,
    weak: turns.length <= 1 && sceneKey === FALLBACK_TOPIC.key,
    chapterTitle: topic.chapterTitle,
  };
  return scene;
}

function turnToBeats(turn) {
  const beats = [];
  const sourceEventIds = [...turn.sourceEventIds];
  const topicKey = classifyTopic(turn);

  if (turn.flags.revised && turn.userDeleted.length) {
    beats.push({
      id: `beat-turn-${turn.id}-revision`,
      type: "beat",
      kind: "transcript_revision",
      role: null,
      text: `The transcript revises itself, striking ${deletedSummary(turn.userDeleted)} from the earlier listening pass.`,
      sourceEventIds,
      topicKey,
      turnId: turn.id,
      children: [],
    });
  }

  if (turnHasUserDialogue(turn)) {
    beats.push({
      id: `beat-turn-${turn.id}-user`,
      type: "beat",
      kind: "user_dialogue",
      role: "USER",
      text: stringifySegments(userSegments(turn)),
      segments: userSegments(turn),
      sourceEventIds,
      topicKey,
      turnId: turn.id,
      children: [],
    });
  }

  if (turn.flags.interruption) {
    beats.push({
      id: `beat-turn-${turn.id}-interruption`,
      type: "beat",
      kind: "interruption",
      role: null,
      text: "The runtime catches an interruption and briefly yields before the exchange can settle.",
      sourceEventIds,
      topicKey: "interruption-handoff",
      turnId: turn.id,
      children: [],
    });
  }

  if (turnHasLlmDialogue(turn)) {
    beats.push({
      id: `beat-turn-${turn.id}-pete`,
      type: "beat",
      kind: "llm_dialogue",
      role: "PETE",
      text: stringifySegments(llmSegments(turn)),
      segments: llmSegments(turn),
      sourceEventIds,
      topicKey,
      turnId: turn.id,
      children: [],
    });
  }

  if (turn.flags.cancelled && turn.llmDeleted.length) {
    beats.push({
      id: `beat-turn-${turn.id}-cancelled`,
      type: "beat",
      kind: "cancellation",
      role: null,
      text: `A prepared reply is cancelled before it can fully land: ${deletedSummary(turn.llmDeleted)}.`,
      sourceEventIds,
      topicKey: "interruption-handoff",
      turnId: turn.id,
      children: [],
    });
  }

  return beats;
}

function classifyTopic(turn) {
  const interruptionOnly = turn.flags.interruption || turn.flags.cancelled;
  const body = joinSemanticText(
    turn.userFinal,
    turn.userCandidateText,
    joinWords(turn.userWords.map((word) => word.text)),
    turn.llmFinal,
    turn.llmProspective,
    joinWords(turn.llmWords.map((word) => word.text)),
    ...turn.userDeleted.map((entry) => entry.text),
    ...turn.llmDeleted.map((entry) => entry.text),
    [...turn.eventKinds].join(" "),
  );

  for (const topic of TOPIC_CATALOG) {
    if (topic.key === "interruption-handoff" && !interruptionOnly) {
      continue;
    }
    if (topic.key !== "interruption-handoff" && interruptionOnly && !body) {
      continue;
    }
    if (topic.keywords.some((pattern) => pattern.test(body))) {
      return topic.key;
    }
  }

  if (interruptionOnly) {
    return "interruption-handoff";
  }
  return FALLBACK_TOPIC.key;
}

function topicForKey(key) {
  return TOPIC_CATALOG.find((topic) => topic.key === key) ?? FALLBACK_TOPIC;
}

function summarizeScene(turns, topic, beats) {
  const userLine = turns.map((turn) => currentUserText(turn)).find(Boolean);
  const peteLine = turns.map((turn) => textContent(turn.llmFinal || turn.llmProspective)).find(Boolean);
  const notes = [];
  if (turns.some((turn) => turn.flags.revised)) {
    notes.push("the transcript revises itself");
  }
  if (turns.some((turn) => turn.flags.cancelled)) {
    notes.push("cancelled speech remains visible");
  }
  if (turns.some((turn) => turn.flags.interruption)) {
    notes.push("an interruption changes the cadence");
  }

  const lead = stripTerminalPunctuation(userLine || peteLine || "The live session continues");
  const noteText = notes.length ? ` ${capitalize(joinWithCommas(notes))}.` : "";
  return `${shortHeading(topic.heading)} holds ${beats.length} beat${beats.length === 1 ? "" : "s"} around ${lead}.${noteText}`;
}

function summarizeEpisode(scenes) {
  if (scenes.length === 1) {
    return scenes[0].summary;
  }
  return scenes.map((scene) => shortHeading(scene.heading)).join(" → ");
}

function describeSceneAction(turns, topic) {
  const notes = [];
  if (turns.some((turn) => turn.flags.revised)) {
    notes.push("Transcript revisions stay attached to the scene.");
  }
  if (turns.some((turn) => turn.flags.cancelled)) {
    notes.push("Cancelled speech remains traceable.");
  }
  if (turns.some((turn) => turn.flags.interruption)) {
    notes.push("An interruption cuts through the exchange.");
  }
  return [topic.action, ...notes].join(" ").trim();
}

function dominantChapterTitle(scenes) {
  const counts = new Map();
  for (const scene of scenes) {
    counts.set(scene.chapterTitle, (counts.get(scene.chapterTitle) ?? 0) + 1);
  }
  return [...counts.entries()].sort((left, right) => right[1] - left[1])[0]?.[0] ?? FALLBACK_TOPIC.chapterTitle;
}

function shortHeading(heading) {
  return heading.replace(/^INT\. LISTENBURY RUNTIME - /, "").replace(/^INT\.\/EXT\. LISTENBURY RUNTIME - /, "");
}

function deletedSummary(entries) {
  return entries
    .slice(-MAX_DELETED_SNIPPETS)
    .map((entry) => `“${entry.text}”`)
    .join(", ");
}

function turnHasUserDialogue(turn) {
  return Boolean(
    turn.userFinal || turn.userStable || turn.userUnstable || turn.userWords.length || turn.userDeleted.length,
  );
}

function turnHasLlmDialogue(turn) {
  return Boolean(turn.llmFinal || turn.llmProspective || turn.llmDeleted.length);
}

function currentUserText(turn) {
  return turn.userFinal || joinWords(turn.userWords.map((word) => word.text)) || turn.userCandidateText;
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
  return compactSegments(segments);
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
  return compactSegments(segments);
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

function stringifySegments(segments) {
  return compactSegments(segments)
    .map((segment) => textContent(segment.text))
    .filter(Boolean)
    .join(" ")
    .replace(/\s+([,.;:!?])/g, "$1")
    .trim();
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

function transcriptCandidateText(candidate) {
  if (!candidate || typeof candidate !== "object") return "";
  return joinSemanticText(candidate.stable_text, candidate.unstable_text);
}

function wordStreamText(words) {
  return joinWords((words ?? []).map((word) => word?.text));
}

function textContent(value) {
  return String(value ?? "")
    .replace(/\s+/g, " ")
    .replace(/\s+([,.;:!?])/g, "$1")
    .trim();
}

function eventTurnNumber(event) {
  const turn = Number(event?.turn ?? 0);
  return Number.isFinite(turn) ? turn : 0;
}

function eventTurnKey(event) {
  const turnId = normalizedId(event?.turn_id);
  if (turnId) {
    return `tid:${turnId}`;
  }
  return `turn:${eventTurnNumber(event)}`;
}

function normalizedId(value) {
  if (value == null) {
    return null;
  }
  if (typeof value === "string" || typeof value === "number") {
    return String(value);
  }
  if (typeof value === "object" && value !== null && "0" in value) {
    return String(value[0]);
  }
  return JSON.stringify(value);
}

function unique(values) {
  return [...new Set(values.filter(Boolean))];
}

function slugify(text) {
  return text.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "");
}

function formatMs(ms) {
  const value = numericMs(ms);
  return Number.isFinite(value) ? `${value}ms` : "unknown";
}

function numericMs(value) {
  return Number.isFinite(value) ? Number(value) : null;
}

function minNumber(left, right) {
  if (!Number.isFinite(left)) return right;
  if (!Number.isFinite(right)) return left;
  return Math.min(left, right);
}

function maxNumber(left, right) {
  if (!Number.isFinite(left)) return right;
  if (!Number.isFinite(right)) return left;
  return Math.max(left, right);
}

function capitalize(text) {
  return text ? `${text[0].toUpperCase()}${text.slice(1)}` : text;
}

function stripTerminalPunctuation(text) {
  return String(text ?? "").replace(/[.?!]+$/g, "");
}

function joinWithCommas(parts) {
  if (parts.length <= 1) {
    return parts[0] ?? "";
  }
  if (parts.length === 2) {
    return `${parts[0]} and ${parts[1]}`;
  }
  return `${parts.slice(0, -1).join(", ")}, and ${parts[parts.length - 1]}`;
}
