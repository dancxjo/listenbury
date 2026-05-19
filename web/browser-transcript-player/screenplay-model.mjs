import { SpanModality } from "./shared-span-model.mjs";
import { resolveSlugline, toTitleCase } from "./scene-heading.mjs";
import {
  MAX_DELETED_SNIPPETS,
  textContent,
  normalizedId,
  joinWords,
  joinSemanticText,
  speechUnitIdFromEvent,
  transcriptCandidateText,
  wordStreamText,
  tokenize,
  deletedTextBetween,
  recordDeletedText,
  applyTranscriptCandidate,
  applyAsrWordStream,
  applyTtsWordStreamRevision,
  applyLlmTextEvent,
  commitLlmText,
  setLlmProspective,
  finalizeLlmText,
} from "./shared/events/reducers.mjs";
import {
  isLlmTextEvent,
  isProspectiveCommitment,
  isPlayedCommitment,
} from "./shared/events/schema.mjs";
import {
  currentUserText,
  prospectiveTail,
  turnHasUserDialogue,
  turnHasLlmDialogue,
} from "./shared/events/selectors.mjs";

const TOPIC_CATALOG = [
  {
    key: "memory-manuscript",
    topicLabel: "MEMORY, DECOUPAGE, AND MANUSCRIPT",
    chapterTitle: "Memory, Découpage, and Manuscript",
    action:
      "Pete and another voice shape live traces into something that can be read later as narrative memory.",
    keywords: [/\bmemory\b/i, /\bd[eé]coupage\b/i, /\bscene\b/i, /\bepisode\b/i, /\bchapter\b/i, /\bmanuscript\b/i],
  },
  {
    key: "phonology-workbench",
    topicLabel: "PHONOLOGY WORKBENCH",
    chapterTitle: "Phonology and the Native Piper Spine",
    action:
      "The conversation narrows to speech mechanics, phonology, and the shape of Pete's own mouth and ear.",
    keywords: [/\bphonolog/i, /\bpiper\b/i, /\bmouth\b/i, /\bhear\b/i, /\bvowel\b/i, /\bconsonant\b/i],
  },
  {
    key: "cuda-bring-up",
    topicLabel: "CUDA BRING-UP",
    chapterTitle: "Building the Mouth",
    action:
      "A technical bring-up scene gathers around the GPU path and the machinery needed to make Pete speak.",
    keywords: [/\bcuda\b/i, /\bgpu\b/i, /\bkernel\b/i, /\bdriver\b/i, /\bbring[- ]?up\b/i],
  },
  {
    key: "wavedeck-inspection",
    topicLabel: "WAVEDECK INSPECTION",
    chapterTitle: "The WaveDeck Era",
    action:
      "Pete and another voice watch the session machinery in motion, following traces, overlap, and routing decisions.",
    keywords: [/\bwave ?deck\b/i, /\boverlap\b/i, /\brouting\b/i, /\btrace\b/i, /\bspan\b/i, /\btimeline\b/i, /\breplay\b/i],
  },
  {
    key: "quiet-grief",
    topicLabel: "QUIET GRIEF",
    chapterTitle: "Teaching Pete to Hear Himself",
    action:
      "The room softens. The talk turns inward and careful, with room for loss, apology, or tenderness.",
    keywords: [/\bgrief\b/i, /\bgriev/i, /\bsad\b/i, /\bloss\b/i, /\bmiss\b/i, /\bsorry\b/i, /\bhurt\b/i],
  },
  {
    key: "interruption-handoff",
    topicLabel: "INTERRUPTION HANDOFF",
    chapterTitle: "The WaveDeck Era",
    action:
      "An interruption cuts across the session and the narrative briefly reorients around overlap, yielding, and cancellation.",
    keywords: [/\binterrupt/i, /\boverlap\b/i, /\byield/i, /\bcancel/i],
  },
];

const FALLBACK_TOPIC = {
  key: "live-session",
  topicLabel: null,
  chapterTitle: "The WaveDeck Era",
  action: "Pete and other voices remain in the live session, gathering the next beat as it arrives.",
};

const PETE_CUE = "PETE";
const UNKNOWN_VOICE_PREFIX = "UNKNOWN VOICE #";

export function createNarrativeSession() {
  return {
    turns: new Map(),
    unknownVoiceOrdinalByKey: new Map(),
    nextUnknownVoiceOrdinal: 1,
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
  maybeCaptureVoiceCue(session, turn, sourceEvent);

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

  const locationContext = options.locationContext ?? {};
  const scenes = buildScenes(turns, locationContext);
  const startedAtMs = turns.length ? turns[0].startedAtMs ?? 0 : 0;
  const endedAtMs = turns.length ? turns[turns.length - 1].endedAtMs ?? startedAtMs : startedAtMs;
  const leadScene = scenes[0] ?? null;
  const episodeNumber = options.episodeNumber ?? 1;
  const title =
    options.title ??
    (leadScene
      ? `Episode ${episodeNumber}: ${leadScene.topicLabel ?? shortHeading(leadScene.heading)}`
      : `Episode ${episodeNumber}: Live Session`);
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
    lines.push(scene.heading);
    if (scene.topicLabel) {
      lines.push(`Soft-note: Topic: ${toTitleCase(scene.topicLabel)}.`);
    }
    lines.push(scene.action, "");
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
      userVoiceCue: null,
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

function maybeCaptureVoiceCue(session, turn, event) {
  if (!isHumanDialogueEvent(event.kind)) {
    return;
  }
  turn.userVoiceCue = resolveVoiceCue(session, event) ?? turn.userVoiceCue ?? nextUnknownVoiceCue(session, "unattributed-human");
}

function isHumanDialogueEvent(kind) {
  return ["transcript_candidate", "asr_timed_word_stream", "transcript"].includes(kind);
}

function resolveVoiceCue(session, event) {
  const entries = eventVoiceAttributions(event);
  for (const entry of entries) {
    const cue = cueFromVoiceEntry(session, entry);
    if (cue) {
      return cue;
    }
  }
  return null;
}

function eventVoiceAttributions(event) {
  if (Array.isArray(event.voice_attributions)) {
    return event.voice_attributions;
  }
  if (event.voice_id != null || event.voice_label != null) {
    return [{ voice_id: event.voice_id, voice_label: event.voice_label }];
  }
  return [];
}

function cueFromVoiceEntry(session, entry) {
  const voiceId = normalizedId(entry?.voice_id);
  const label = entry?.voice_label;
  if (label && typeof label === "object" && typeof label.ordinal === "number") {
    return `${UNKNOWN_VOICE_PREFIX}${label.ordinal}`;
  }
  const textLabel = textContent(
    typeof label === "string"
      ? label
      : label?.named ?? label?.cluster ?? label?.kind ?? label?.label ?? null,
  );
  if (!textLabel) {
    return nextUnknownVoiceCue(session, voiceId ? `voice-id:${voiceId}` : "unattributed-human");
  }
  const normalized = textLabel.toUpperCase();
  if (normalized === "PETE") {
    return PETE_CUE;
  }
  if (normalized.includes("UNKNOWN")) {
    return nextUnknownVoiceCue(session, voiceId ? `voice-id:${voiceId}` : `voice-label:${normalized}`);
  }
  if (normalized.includes("BACKGROUND")) {
    return "BACKGROUND VOICE";
  }
  if (normalized.includes("ENVIRONMENT")) {
    return "ENVIRONMENT";
  }
  return normalized;
}

function nextUnknownVoiceCue(session, key) {
  const stableKey = textContent(key) || "unattributed-human";
  if (!session.unknownVoiceOrdinalByKey.has(stableKey)) {
    session.unknownVoiceOrdinalByKey.set(stableKey, session.nextUnknownVoiceOrdinal++);
  }
  return `${UNKNOWN_VOICE_PREFIX}${session.unknownVoiceOrdinalByKey.get(stableKey)}`;
}

function resolveTurnVoiceCue(turn) {
  return turn.userVoiceCue || `${UNKNOWN_VOICE_PREFIX}1`;
}

function buildScenes(turns, locationContext = {}) {
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
  return merged.map((group, index) => buildScene(group.turns, index, group.key, locationContext));
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

function buildScene(turns, index, sceneKey, locationContext = {}) {
  const topic = topicForKey(sceneKey);
  const sceneContext = {
    ...locationContext,
    timestampMs: turns[0]?.startedAtMs ?? locationContext.timestampMs ?? null,
  };
  const heading = resolveSlugline(sceneContext);
  const beats = consolidateConsecutiveDialogueBeats(turns.flatMap((turn) => turnToBeats(turn)));
  const sourceEventIds = unique(turns.flatMap((turn) => turn.sourceEventIds));
  const summary = summarizeScene(turns, topic, beats);
  const scene = {
    id: `scene-${index + 1}`,
    type: "scene",
    topicKey: sceneKey,
    topicLabel: topic.topicLabel,
    heading,
    action: describeSceneAction(turns, topic),
    summary,
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

  if (turnHasUserDialogue(turn)) {
    beats.push({
      id: `beat-turn-${turn.id}-voice`,
      type: "beat",
      kind: "voice_dialogue",
      role: resolveTurnVoiceCue(turn),
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
      role: PETE_CUE,
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

function consolidateConsecutiveDialogueBeats(beats) {
  const consolidated = [];

  for (const beat of beats) {
    const previous = consolidated[consolidated.length - 1];
    if (previous?.role && previous.role === beat.role) {
      const segments = [
        ...(previous.segments ?? [{ text: previous.text }]),
        ...(beat.segments ?? [{ text: beat.text }]),
      ];
      previous.id = `${previous.id}+${beat.id}`;
      previous.kind = previous.kind === beat.kind ? previous.kind : "dialogue";
      previous.text = stringifySegments(segments);
      previous.segments = compactSegments(segments);
      previous.sourceEventIds = unique([
        ...(previous.sourceEventIds ?? []),
        ...(beat.sourceEventIds ?? []),
      ]);
      previous.turnId ??= beat.turnId;
      previous.children = [...(previous.children ?? []), ...(beat.children ?? [])];
      continue;
    }

    consolidated.push({ ...beat });
  }

  return consolidated;
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
  if (turns.some((turn) => turn.flags.cancelled)) {
    notes.push("cancelled speech remains visible");
  }
  if (turns.some((turn) => turn.flags.interruption)) {
    notes.push("an interruption changes the cadence");
  }

  const sceneLabel = topic.topicLabel ?? shortHeading(topic.key);
  const lead = stripTerminalPunctuation(userLine || peteLine || "The live session continues");
  const noteText = notes.length ? ` ${capitalize(joinWithCommas(notes))}.` : "";
  return `${sceneLabel} holds ${beats.length} beat${beats.length === 1 ? "" : "s"} around ${lead}.${noteText}`;
}

function summarizeEpisode(scenes) {
  if (scenes.length === 1) {
    return scenes[0].summary;
  }
  return scenes.map((scene) => scene.topicLabel ?? shortHeading(scene.heading)).join(" → ");
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
  const topicNote = topic.topicLabel ? `Topic: ${toTitleCase(topic.topicLabel)}.` : null;
  return [topic.action, ...notes, topicNote].filter(Boolean).join(" ").trim();
}

function dominantChapterTitle(scenes) {
  const counts = new Map();
  for (const scene of scenes) {
    counts.set(scene.chapterTitle, (counts.get(scene.chapterTitle) ?? 0) + 1);
  }
  return [...counts.entries()].sort((left, right) => right[1] - left[1])[0]?.[0] ?? FALLBACK_TOPIC.chapterTitle;
}

function shortHeading(heading) {
  // Extract the place name from a slugline like "INT. LIVING ROOM - NIGHT" → "LIVING ROOM"
  const match = heading.match(/^(?:INT\.|EXT\.|INT\.\/EXT\.) (.+?) - /);
  return match ? match[1] : heading;
}

function deletedSummary(entries) {
  return entries
    .slice(-MAX_DELETED_SNIPPETS)
    .map((entry) => `“${entry.text}”`)
    .join(", ");
}

function userSegments(turn) {
  const segments = [];

  if (turn.userFinal) {
    segments.push({
      text: turn.userFinal,
      spanMetadata: aggregateDialogueSpanMetadata(turn.userWords, turn.id, "asr_timed_word_stream"),
    });
  } else if (turn.userWords.length) {
    for (const word of turn.userWords) {
      segments.push({
        text: word.text,
        className: isProspectiveCommitment(word.commitment) ? "prospective-asr" : "",
        word: true,
        spanMetadata: dialogueSpanMetadata(word, turn.id, "asr_timed_word_stream"),
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
    segments.push({
      text: finalText,
      spanMetadata: aggregateDialogueSpanMetadata(turn.llmWords, turn.id, "tts_timed_word_stream_revision"),
    });
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
        spanMetadata: dialogueSpanMetadata(word, turn.id, "tts_timed_word_stream_revision"),
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
    const spanMetadata = normalizeSegmentSpanMetadata(segment.spanMetadata);
    if (previous && previous.className === segment.className && segment.word) {
      previous.text += nextText;
      if (spanMetadata.length) {
        previous.spanMetadata = previous.spanMetadata ?? [];
        previous.spanMetadata.push(...spanMetadata);
      }
    } else {
      result.push({
        text: nextText,
        className: segment.className || "",
        spanMetadata: spanMetadata.length ? spanMetadata : undefined,
      });
    }
  }
  return result;
}

function normalizeSegmentSpanMetadata(spanMetadata) {
  if (!spanMetadata) {
    return [];
  }
  return Array.isArray(spanMetadata) ? spanMetadata.filter(Boolean) : [spanMetadata].filter(Boolean);
}

function dialogueSpanMetadata(word, turnId, source) {
  const startMs = Number(word?.timing?.start_ms);
  const endMs = Number(word?.timing?.end_ms);
  if (!Number.isFinite(startMs) || !Number.isFinite(endMs)) {
    return null;
  }
  return {
    id:
      word?.span_id != null
        ? String(word.span_id)
        : `turn-${turnId ?? 0}:${source || "stream"}:${word?.id ?? safeSpanIdToken(word?.text)}`,
    modality: SpanModality.Word,
    start_ms: startMs,
    end_ms: endMs,
    source,
    turn: turnId,
    text: word?.text ?? "",
  };
}

function aggregateDialogueSpanMetadata(words, turnId, source) {
  return (words ?? [])
    .map((word) => dialogueSpanMetadata(word, turnId, source))
    .filter(Boolean);
}

function safeSpanIdToken(value) {
  const text = String(value ?? "word").trim().toLowerCase();
  const normalized = text.replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
  return normalized || "word";
}

function stringifySegments(segments) {
  return compactSegments(segments)
    .map((segment) => textContent(segment.text))
    .filter(Boolean)
    .join(" ")
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
