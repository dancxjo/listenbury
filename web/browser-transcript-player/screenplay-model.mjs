import { SpanModality } from "./shared-span-model.mjs";
import { resolveSlugline } from "./scene-heading.mjs";
import {
  MAX_DELETED_SNIPPETS,
  textContent,
  normalizedId,
  joinWords,
  joinSemanticText,
  syntheticUnitIdFromEvent,
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

const FALLBACK_TOPIC = {
  key: "live-session",
  topicLabel: null,
  chapterTitle: "Live Session",
  action: "The live exchange continues as a screenplay beat.",
};

const PETE_CUE = "PETE";
const UNKNOWN_VOICE_PREFIX = "UNKNOWN VOICE #";
const DEFAULT_EPISODE_GAP_MS = 10 * 60 * 1000;
const STAGE_INSTRUCTION_EVENT_KINDS = new Set([
  "stage_instruction",
  "screenplay_stage_instruction",
  "pericope_update",
  "pete_stage_updated",
  "set_stage",
  "auditory_scene_observation",
]);
const SCENE_CUT_EVENT_KINDS = new Set(["scene_cut", "screenplay_scene_cut"]);
const EPISODE_CUT_EVENT_KINDS = new Set(["episode_cut", "screenplay_episode_cut"]);

export function createNarrativeSession() {
  return {
    turns: new Map(),
    unknownVoiceOrdinalByKey: new Map(),
    nextUnknownVoiceOrdinal: 1,
    proposition: null,
    propositionDeleted: [],
    sourceEvents: [],
    stageInstructions: [],
    narrativeCuts: [],
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
  captureStageInstruction(session, sourceEvent);
  captureNarrativeCut(session, sourceEvent);

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

  if (sourceEvent.kind === "source_attributed_transcript") {
    applySourceAttributedTranscript(session, turn, sourceEvent);
    return;
  }

  if (sourceEvent.kind === "tts_timed_word_stream_revision") {
    applyTtsWordStreamRevision(turn, sourceEvent);
    return;
  }

  if (
    isLlmTextEvent(sourceEvent.kind) &&
    (textContent(sourceEvent.text) || syntheticUnitIdFromEvent(sourceEvent))
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

export function buildNarrativeTimeline(session, options = {}) {
  const turns = sortedNarrativeTurns(session);
  if (!turns.length) {
    return [
      buildNarrativeEpisodeFromTurns(session, [], {
        ...options,
        id: options.id ?? `${options.idPrefix ?? "episode"}-1`,
        episodeNumber: options.episodeNumber ?? 1,
      }),
    ];
  }

  const groups = groupTurnsIntoEpisodes(turns, session.narrativeCuts, options);
  return groups.map((group, index) =>
    buildNarrativeEpisodeFromTurns(session, group.turns, {
      ...options,
      id: options.id ?? `${options.idPrefix ?? "episode"}-${index + 1}`,
      episodeNumber: (options.episodeNumber ?? 1) + index,
      episodeCut: group.cut ?? null,
    }),
  );
}

export function buildNarrativeEpisode(session, options = {}) {
  return buildNarrativeEpisodeFromTurns(session, sortedNarrativeTurns(session), options);
}

function buildNarrativeEpisodeFromTurns(session, turns, options = {}) {
  const locationContext = options.locationContext ?? {};
  const episodeCuts = options.episodeCut ? [options.episodeCut] : [];
  const episodeNarrativeCuts = narrativeCutsForWindow(
    session.narrativeCuts,
    turns[0]?.startedAtMs ?? null,
    turns[turns.length - 1]?.endedAtMs ?? null,
  );
  const scenes = buildScenes(turns, locationContext, {
    narrativeCuts: [...episodeNarrativeCuts, ...episodeCuts],
    stageInstructions: stageInstructionsForWindow(
      session.stageInstructions,
      turns[0]?.startedAtMs ?? null,
      turns[turns.length - 1]?.endedAtMs ?? null,
    ),
  });
  const startedAtMs = turns.length ? turns[0].startedAtMs ?? 0 : 0;
  const endedAtMs = turns.length ? turns[turns.length - 1].endedAtMs ?? startedAtMs : startedAtMs;
  const episodeNumber = options.episodeNumber ?? 1;
  const title = options.title ?? `Episode ${episodeNumber}: Live Session`;
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
      cutAuthority: options.cutAuthority ?? "explicit runtime cuts only",
      episodeCut: options.episodeCut ?? null,
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
    progressiveSummary: summarizeProgressively(scenes),
    stageInstruction: currentStageInstruction(scenes),
    metadata,
    sourceEventIds,
    waveDeckHref: options.waveDeckHref ?? "/",
    replayHref: options.replayHref ?? null,
    scenes,
    sceneList: scenes.map((scene) => ({ id: scene.id, heading: scene.heading, summary: scene.summary })),
    timeline: buildEpisodeMemoryTimeline(scenes),
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
    `Cuts: ${episode.metadata.cutAuthority}`,
    "",
  );
  if (episode.stageInstruction?.text) {
    lines.push(`Current screenplay beat: ${episode.stageInstruction.text}`, "");
  }
  if (episode.timeline?.length) {
    lines.push("Episodic memory timeline:");
    for (const item of episode.timeline) {
      lines.push(`${formatMs(item.startedAtMs)}–${formatMs(item.endedAtMs)} ${item.label}: ${item.summary}`);
    }
    lines.push("");
  }

  for (const scene of episode.scenes) {
    lines.push(scene.heading);
    if (scene.stageInstruction?.text) {
      lines.push(`Screenplay beat: ${scene.stageInstruction.text}`);
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

function sortedNarrativeTurns(session) {
  return [...session.turns.values()]
    .filter((turn) => turnHasNarrativeMaterial(turn))
    .sort((left, right) => (left.startedAtMs ?? left.id) - (right.startedAtMs ?? right.id));
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
      userStartedAtMs: null,
      userEndedAtMs: null,
      llmFinal: "",
      llmProspective: "",
      llmFragments: [],
      llmWords: [],
      llmDeleted: [],
      llmStartedAtMs: null,
      llmEndedAtMs: null,
      interruptionStartedAtMs: null,
      interruptionEndedAtMs: null,
      cancellationAtMs: null,
      flags: {
        interruption: false,
        cancelled: false,
        revised: false,
        prospective: false,
      },
      syntheticUnitsById: new Map(),
    });
  }
  return session.turns.get(turnKey);
}

function captureStageInstruction(session, event) {
  const instruction = stageInstructionFromEvent(event);
  if (!instruction) {
    return;
  }
  session.stageInstructions.push(instruction);
}

function captureNarrativeCut(session, event) {
  const cut = narrativeCutFromEvent(event);
  if (!cut) {
    return;
  }
  session.narrativeCuts.push(cut);
}

function stageInstructionFromEvent(event) {
  const runtimeSubtype = runtimeSubtypeFromEvent(event);
  const eventKind = String(event.kind ?? "");
  const runtimeKind = String(runtimeSubtype?.kind ?? "");
  const isStageEvent =
    STAGE_INSTRUCTION_EVENT_KINDS.has(eventKind) ||
    STAGE_INSTRUCTION_EVENT_KINDS.has(runtimeKind) ||
    textContent(event.artifact?.stage_instruction) ||
    textContent(runtimeSubtype?.artifact?.stage_instruction);
  if (!isStageEvent) {
    return null;
  }

  const text = firstText(
    event.artifact?.stage_instruction,
    event.artifact?.instruction,
    event.artifact?.description,
    runtimeSubtype?.artifact?.stage_instruction,
    runtimeSubtype?.artifact?.instruction,
    runtimeSubtype?.artifact?.description,
    event.instruction,
    runtimeSubtype?.instruction,
    runtimeSubtype?.text,
    event.text,
  );
  if (!text) {
    return null;
  }
  const elapsedMs = numericMs(event.elapsed_ms);
  return {
    id: event.sourceId,
    type: "stage_instruction",
    text,
    summary: firstText(
      event.artifact?.summary,
      runtimeSubtype?.artifact?.summary,
      event.summary,
      runtimeSubtype?.summary,
      text,
    ),
    startedAtMs: elapsedMs,
    endedAtMs: elapsedMs,
    source: runtimeSubtype ? "runtime_event" : "trace_event",
    sourceEventIds: [event.sourceId],
  };
}

function narrativeCutFromEvent(event) {
  const runtimeSubtype = runtimeSubtypeFromEvent(event);
  const eventKind = String(event.kind ?? "");
  const runtimeKind = String(runtimeSubtype?.kind ?? "");
  const explicitLevel = firstText(
    event.level,
    event.cut_level,
    event.artifact?.level,
    event.artifact?.cut_level,
    runtimeSubtype?.artifact?.level,
    runtimeSubtype?.artifact?.cut_level,
  ).toLowerCase();
  let level = null;
  if (explicitLevel === "episode" || EPISODE_CUT_EVENT_KINDS.has(eventKind) || EPISODE_CUT_EVENT_KINDS.has(runtimeKind)) {
    level = "episode";
  } else if (explicitLevel === "scene" || SCENE_CUT_EVENT_KINDS.has(eventKind) || SCENE_CUT_EVENT_KINDS.has(runtimeKind)) {
    level = "scene";
  } else if (eventKind === "narrative_cut" || eventKind === "screenplay_cut" || runtimeKind === "narrative_cut") {
    level = "scene";
  }
  if (!level) {
    return null;
  }
  const atMs = numericMs(event.elapsed_ms);
  return {
    id: event.sourceId,
    type: "narrative_cut",
    level,
    atMs,
    reason: firstText(event.reason, event.artifact?.reason, runtimeSubtype?.reason) || `${level} cut`,
    source: runtimeSubtype ? "runtime_event" : "trace_event",
    sourceEventIds: [event.sourceId],
  };
}

function runtimeSubtypeFromEvent(event) {
  const runtimeKind = event.runtime_event?.kind;
  if (runtimeKind?.event && typeof runtimeKind.event === "object") {
    return runtimeKind.event;
  }
  if (runtimeKind?.kind || runtimeKind?.text || runtimeKind?.artifact) {
    return runtimeKind;
  }
  return null;
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
  const atMs = numericMs(event.elapsed_ms);
  if (text) {
    turn.flags.prospective ||= event.kind === "transcript_candidate" || event.kind === "speculative_synthetic_unit_updated";
  }

  if (isHumanDialogueEvent(event.kind) || event.kind === "speech_started" || event.kind === "speech_stopped") {
    const range = eventWordTimingRange(event);
    turn.userStartedAtMs = minNumber(turn.userStartedAtMs, range.startedAtMs ?? atMs);
    turn.userEndedAtMs = maxNumber(turn.userEndedAtMs, range.endedAtMs ?? atMs);
  }

  if (isLlmTextEvent(event.kind) || event.kind === "playback_started" || event.kind === "playback_finished") {
    const range = eventArtifactTimingRange(event);
    turn.llmStartedAtMs = minNumber(turn.llmStartedAtMs, range.startedAtMs ?? atMs);
    turn.llmEndedAtMs = maxNumber(turn.llmEndedAtMs, range.endedAtMs ?? atMs);
  }

  if (["interruption_detected", "overlap_started", "yield_started", "yield_ended"].includes(event.kind)) {
    turn.flags.interruption = true;
    const range = eventArtifactTimingRange(event);
    turn.interruptionStartedAtMs = minNumber(turn.interruptionStartedAtMs, range.startedAtMs ?? atMs);
    turn.interruptionEndedAtMs = maxNumber(turn.interruptionEndedAtMs, range.endedAtMs ?? atMs);
  }

  if (["synthetic_unit_cancelled", "tts_timed_word_stream_revision"].includes(event.kind)) {
    const cancelledWords = Array.isArray(event.artifact?.words)
      ? event.artifact.words.every((word) => String(word?.commitment ?? "") === "Cancelled")
      : false;
    if (event.kind === "synthetic_unit_cancelled" || cancelledWords || String(event.reason ?? "").toLowerCase().includes("cancel")) {
      turn.flags.cancelled = true;
      turn.cancellationAtMs = minNumber(turn.cancellationAtMs, atMs);
    }
  }
}

function eventWordTimingRange(event) {
  const words = Array.isArray(event.artifact?.words) ? event.artifact.words : [];
  let startedAtMs = null;
  let endedAtMs = null;
  for (const word of words) {
    const startMs = numericMs(word?.timing?.start_ms);
    const endMs = numericMs(word?.timing?.end_ms);
    startedAtMs = minNumber(startedAtMs, startMs);
    endedAtMs = maxNumber(endedAtMs, endMs);
  }
  return { startedAtMs, endedAtMs };
}

function eventArtifactTimingRange(event) {
  return {
    startedAtMs: numericMs(event.artifact?.start_ms ?? event.artifact?.clip_start_ms),
    endedAtMs: numericMs(event.artifact?.end_ms ?? event.artifact?.clip_end_ms),
  };
}

function maybeCaptureVoiceCue(session, turn, event) {
  if (!isHumanDialogueEvent(event.kind)) {
    return;
  }
  turn.userVoiceCue = resolveVoiceCue(session, event) ?? turn.userVoiceCue ?? nextUnknownVoiceCue(session, "unattributed-human");
}

function isHumanDialogueEvent(kind) {
  return ["transcript_candidate", "asr_timed_word_stream", "transcript", "source_attributed_transcript"].includes(kind);
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

/**
 * Derive a screenplay character-cue string from a `source_label` field as
 * serialised by the Rust `SourceLabel` enum.
 *
 * The Rust serialisation produces one of:
 *   { "NamedVoice": "Pete" }
 *   { "UnknownVoice": { "ordinal": 1 } }
 *   { "BackgroundVoice": { "ordinal": 2 } }
 *   { "Playback": "Television" }
 *   "RoomNoise"
 *   { "Custom": "Television Voice" }
 * …or a plain string for simple variants.
 */
function cueFromSourceLabel(label) {
  if (!label) {
    return null;
  }
  if (typeof label === "string") {
    const up = label.toUpperCase();
    if (up === "ROOMNOISE" || up === "ROOM_NOISE") {
      return "ROOM NOISE";
    }
    return up;
  }
  if (typeof label === "object") {
    if (label.NamedVoice) {
      return `${label.NamedVoice.trim().toUpperCase()} VOICE`;
    }
    if (label.UnknownVoice && typeof label.UnknownVoice.ordinal === "number") {
      return `${UNKNOWN_VOICE_PREFIX}${label.UnknownVoice.ordinal}`;
    }
    if (label.BackgroundVoice && typeof label.BackgroundVoice.ordinal === "number") {
      return `BACKGROUND VOICE #${label.BackgroundVoice.ordinal}`;
    }
    if (label.Playback) {
      return `${label.Playback.trim().toUpperCase()} PLAYBACK`;
    }
    if (label.Custom) {
      return label.Custom.trim().toUpperCase();
    }
  }
  return null;
}

/**
 * Apply a `source_attributed_transcript` event to the current turn.
 *
 * The event shape mirrors the Rust `SourceAttributedTranscript` struct:
 *   {
 *     kind: "source_attributed_transcript",
 *     text: string,
 *     source_label: SourceLabel,   // serialised Rust enum
 *     transcript_confidence: number,
 *     attribution_confidence: number,
 *     overlap: AcousticMixtureId | null,
 *     range: { start: { millis: number }, end: { millis: number } },
 *   }
 *
 * The `source_label` drives the character cue; `text` populates the
 * transcript dialogue; low `transcript_confidence` or a present `overlap`
 * renders the text as `[indistinct]`.
 */
function applySourceAttributedTranscript(session, turn, event) {
  const label = event.source_label ?? event.artifact?.source_label;
  const cue = cueFromSourceLabel(label);
  if (cue) {
    turn.userVoiceCue = cue;
  } else {
    turn.userVoiceCue ??= nextUnknownVoiceCue(session, "unattributed-human");
  }

  const rawText = textContent(event.text);
  const transcriptConfidence = typeof event.transcript_confidence === "number"
    ? event.transcript_confidence
    : 1.0;
  const isOverlapped = event.overlap != null;
  const isIndistinct = transcriptConfidence < 0.4 || isOverlapped;
  const resolvedText = isIndistinct ? "[indistinct]" : rawText;

  if (!resolvedText) {
    return;
  }

  const deleted = deletedTextBetween(currentUserText(turn), resolvedText);
  if (deleted.length) {
    turn.flags.revised = true;
  }
  recordDeletedText(turn.userDeleted, deleted, event.elapsed_ms);
  turn.userFinal = resolvedText;
  turn.userStable = resolvedText;
  turn.userUnstable = "";
  turn.userCandidateText = resolvedText;
}

function resolveTurnVoiceCue(turn) {
  return turn.userVoiceCue || `${UNKNOWN_VOICE_PREFIX}1`;
}

function groupTurnsIntoEpisodes(turns, narrativeCuts = [], options = {}) {
  const episodeGapMs = options.episodeGapMs ?? DEFAULT_EPISODE_GAP_MS;
  const groups = [];
  for (const turn of turns) {
    const current = groups[groups.length - 1];
    const explicitCut = current ? strongestCutBetween(narrativeCuts, current.endedAtMs, turn.startedAtMs, "episode") : null;
    const gapCut =
      current && !explicitCut && Number.isFinite(current.endedAtMs) && Number.isFinite(turn.startedAtMs) && turn.startedAtMs - current.endedAtMs >= episodeGapMs
        ? {
            id: `gap:${current.endedAtMs}:${turn.startedAtMs}`,
            type: "narrative_cut",
            level: "episode",
            atMs: turn.startedAtMs,
            reason: `silence gap ${turn.startedAtMs - current.endedAtMs}ms`,
            source: "screenplay-model",
            sourceEventIds: [],
          }
        : null;
    const cut = explicitCut ?? gapCut;
    if (!current || cut) {
      groups.push({ turns: [turn], cut, endedAtMs: turn.endedAtMs });
      continue;
    }
    current.turns.push(turn);
    current.endedAtMs = maxNumber(current.endedAtMs, turn.endedAtMs);
  }
  return groups;
}

function buildScenes(turns, locationContext = {}, options = {}) {
  const narrativeCuts = options.narrativeCuts ?? [];
  const grouped = [];
  for (const turn of turns) {
    const current = grouped[grouped.length - 1];
    const explicitCut = current ? strongestCutBetween(narrativeCuts, current.endedAtMs, turn.startedAtMs, "scene") : null;
    if (!current || explicitCut) {
      grouped.push({ key: FALLBACK_TOPIC.key, turns: [turn], cut: explicitCut });
    } else {
      current.turns.push(turn);
      current.endedAtMs = maxNumber(current.endedAtMs, turn.endedAtMs);
    }
    grouped[grouped.length - 1].endedAtMs = maxNumber(grouped[grouped.length - 1].endedAtMs, turn.endedAtMs);
  }

  return grouped.map((group, index) =>
    buildScene(group.turns, index, group.key, locationContext, {
      cut: group.cut ?? null,
      stageInstructions: stageInstructionsForWindow(
        options.stageInstructions ?? [],
        group.turns[0]?.startedAtMs ?? null,
        group.turns[group.turns.length - 1]?.endedAtMs ?? null,
      ),
    }),
  );
}

function buildScene(turns, index, sceneKey, locationContext = {}, options = {}) {
  const topic = topicForKey(sceneKey);
  const sceneContext = {
    ...locationContext,
    timestampMs: turns[0]?.startedAtMs ?? locationContext.timestampMs ?? null,
  };
  const heading = resolveSlugline(sceneContext);
  const beats = consolidateConsecutiveDialogueBeats(sortBeatsForScreenplay(turns.flatMap((turn) => turnToBeats(turn))));
  const sourceEventIds = unique(turns.flatMap((turn) => turn.sourceEventIds));
  const summary = summarizeScene(turns, topic, beats);
  const startedAtMs = turns[0]?.startedAtMs ?? null;
  const endedAtMs = turns[turns.length - 1]?.endedAtMs ?? startedAtMs;
  const explicitStageInstruction = collapseStageInstructions(options.stageInstructions ?? [], startedAtMs, endedAtMs);
  const stageInstruction = explicitStageInstruction ?? deriveStageInstruction(turns, topic, beats, startedAtMs, endedAtMs, sourceEventIds);
  const scene = {
    id: `scene-${index + 1}`,
    type: "scene",
    topicKey: sceneKey,
    topicLabel: topic.topicLabel,
    heading,
    startedAtMs,
    endedAtMs,
    action: describeSceneAction(turns, topic),
    summary,
    stageInstruction,
    cut: options.cut ?? null,
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
  const topicKey = FALLBACK_TOPIC.key;

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
      startedAtMs: turn.userStartedAtMs ?? turn.startedAtMs,
      endedAtMs: turn.userEndedAtMs ?? turn.startedAtMs,
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
      startedAtMs: turn.llmStartedAtMs ?? turn.startedAtMs,
      endedAtMs: turn.llmEndedAtMs ?? turn.endedAtMs,
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
      startedAtMs: turn.interruptionStartedAtMs ?? turn.endedAtMs,
      endedAtMs: turn.interruptionEndedAtMs ?? turn.endedAtMs,
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
      startedAtMs: turn.cancellationAtMs ?? turn.endedAtMs,
      endedAtMs: turn.cancellationAtMs ?? turn.endedAtMs,
      children: [],
    });
  }

  return beats;
}

function sortBeatsForScreenplay(beats) {
  return beats
    .map((beat, index) => ({ ...beat, screenplayOrder: index }))
    .sort(
      (left, right) =>
        (left.startedAtMs ?? Number.POSITIVE_INFINITY) - (right.startedAtMs ?? Number.POSITIVE_INFINITY) ||
        beatTimingWeight(left) - beatTimingWeight(right) ||
        left.screenplayOrder - right.screenplayOrder,
    )
    .map(({ screenplayOrder, ...beat }) => beat);
}

function beatTimingWeight(beat) {
  if (beat.kind === "interruption" || beat.kind === "cancellation") {
    return 0;
  }
  return 1;
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
      previous.startedAtMs = minNumber(previous.startedAtMs, beat.startedAtMs);
      previous.endedAtMs = maxNumber(previous.endedAtMs, beat.endedAtMs);
      previous.children = [...(previous.children ?? []), ...(beat.children ?? [])];
      continue;
    }

    consolidated.push({ ...beat });
  }

  return consolidated;
}

function topicForKey(_key) {
  return FALLBACK_TOPIC;
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

  const lead = stripTerminalPunctuation(userLine || peteLine || "The live session continues");
  const noteText = notes.length ? ` ${capitalize(joinWithCommas(notes))}.` : "";
  return `${beats.length} beat${beats.length === 1 ? "" : "s"} around ${lead}.${noteText}`;
}

function summarizeEpisode(scenes) {
  if (scenes.length === 1) {
    return scenes[0].summary;
  }
  const beatCount = scenes.reduce((total, scene) => total + scene.beats.length, 0);
  return `${scenes.length} explicit scene${scenes.length === 1 ? "" : "s"} with ${beatCount} beat${beatCount === 1 ? "" : "s"}.`;
}

function summarizeProgressively(scenes) {
  if (!scenes.length) {
    return "No episodic memory has formed yet.";
  }
  return scenes
    .map((scene, index) => `Scene ${index + 1}: ${scene.summary}`)
    .join(" ");
}

function currentStageInstruction(scenes) {
  return scenes
    .slice()
    .reverse()
    .find((scene) => scene.stageInstruction?.text)?.stageInstruction ?? null;
}

function buildEpisodeMemoryTimeline(scenes) {
  return scenes.map((scene, index) => ({
    id: `timeline-${scene.id}`,
    type: "episodic_memory_scene",
    label: `Scene ${index + 1}`,
    heading: scene.heading,
    startedAtMs: scene.startedAtMs,
    endedAtMs: scene.endedAtMs,
    topicLabel: scene.topicLabel,
    summary: scene.summary,
    stageInstruction: scene.stageInstruction?.text ?? "",
    cutReason: scene.cut?.reason ?? null,
    sourceEventIds: scene.sourceEventIds,
  }));
}

function collapseStageInstructions(instructions, startedAtMs, endedAtMs) {
  const scoped = stageInstructionsForWindow(instructions, startedAtMs, endedAtMs);
  if (!scoped.length) {
    return null;
  }
  const latest = scoped[scoped.length - 1];
  const sourceEventIds = unique(scoped.flatMap((instruction) => instruction.sourceEventIds ?? []));
  return {
    ...latest,
    id: latest.id ?? `stage:${startedAtMs ?? "na"}:${endedAtMs ?? "na"}`,
    startedAtMs: scoped[0].startedAtMs ?? startedAtMs,
    endedAtMs: latest.endedAtMs ?? endedAtMs,
    sourceEventIds,
  };
}

function deriveStageInstruction(turns, topic, beats, startedAtMs, endedAtMs, sourceEventIds) {
  const leadBeat = stageInstructionLeadBeat(beats);
  const lead = stripTerminalPunctuation(leadBeat?.text ?? topic.action ?? "The live session continues");
  const activeSpeakers = unique(beats.map((beat) => beat.role).filter(Boolean));
  const actorText = activeSpeakers.length
    ? `${joinWithCommas(activeSpeakers)} continue`
    : "The runtime continues";
  const revisionText = turns.some((turn) => turn.flags.prospective)
    ? " while live text is still settling"
    : "";
  const text = `Action: ${actorText} the live scene; latest beat: ${lead}${revisionText}.`;
  return {
    id: `stage-derived:${startedAtMs ?? "na"}:${endedAtMs ?? "na"}`,
    type: "stage_instruction",
    text,
    summary: text,
    startedAtMs,
    endedAtMs,
    source: "screenplay-model",
    sourceEventIds,
  };
}

function stageInstructionLeadBeat(beats) {
  const interruptionBeat = beats
    .slice()
    .reverse()
    .find((beat) => beat.kind === "interruption" || beat.kind === "cancellation");
  if (interruptionBeat) {
    return interruptionBeat;
  }
  return beats
    .slice()
    .reverse()
    .find((beat) => beat.text);
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
  return [topic.action, ...notes].filter(Boolean).join(" ").trim();
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

function narrativeCutsForWindow(cuts, startedAtMs, endedAtMs) {
  return (cuts ?? [])
    .filter((cut) => inWindow(cut.atMs, startedAtMs, endedAtMs))
    .sort((left, right) => (left.atMs ?? 0) - (right.atMs ?? 0));
}

function stageInstructionsForWindow(instructions, startedAtMs, endedAtMs) {
  return (instructions ?? [])
    .filter((instruction) => inWindow(instruction.startedAtMs, startedAtMs, endedAtMs))
    .sort((left, right) => (left.startedAtMs ?? 0) - (right.startedAtMs ?? 0));
}

function strongestCutBetween(cuts, previousEndedAtMs, nextStartedAtMs, minimumLevel = "scene") {
  const ordered = (cuts ?? [])
    .filter((cut) => cutAppliesAtLevel(cut, minimumLevel))
    .filter((cut) => afterPrevious(cut.atMs, previousEndedAtMs) && beforeNext(cut.atMs, nextStartedAtMs))
    .sort((left, right) => levelWeight(right.level) - levelWeight(left.level) || (right.atMs ?? 0) - (left.atMs ?? 0));
  return ordered[0] ?? null;
}

function cutAppliesAtLevel(cut, minimumLevel) {
  return levelWeight(cut?.level) >= levelWeight(minimumLevel);
}

function levelWeight(level) {
  if (level === "episode") return 2;
  if (level === "scene") return 1;
  return 0;
}

function inWindow(value, startedAtMs, endedAtMs) {
  if (!Number.isFinite(value)) {
    return false;
  }
  const afterStart = !Number.isFinite(startedAtMs) || value >= startedAtMs;
  const beforeEnd = !Number.isFinite(endedAtMs) || value <= endedAtMs;
  return afterStart && beforeEnd;
}

function afterPrevious(value, previousEndedAtMs) {
  return !Number.isFinite(previousEndedAtMs) || !Number.isFinite(value) || value >= previousEndedAtMs;
}

function beforeNext(value, nextStartedAtMs) {
  return !Number.isFinite(nextStartedAtMs) || !Number.isFinite(value) || value <= nextStartedAtMs;
}

function unique(values) {
  return [...new Set(values.filter(Boolean))];
}

function firstText(...values) {
  for (const value of values) {
    const text = textContent(value);
    if (text) {
      return text;
    }
  }
  return "";
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
